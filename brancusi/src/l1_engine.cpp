// Layer 1: bsd_* on top of the public sd.cpp API (include/stable-diffusion.h) and the
// public ggml backend API. Do not include upstream internal headers in this file.

#include "bsd.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "ggml-backend.h"
#include "stable-diffusion.h"

namespace fs = std::filesystem;

struct bsd_engine {
    int n_threads = 0;
    sd_ctx_t* ctx = nullptr;
    std::string error;
    std::string error_detail;

    std::mutex log_mutex;
    std::deque<std::string> recent_log;  // Last log lines, for error detail and classification.

    bsd_event_cb_t event_cb = nullptr;
    void* event_user        = nullptr;

    std::atomic<bool> generating{false};
    std::atomic<bool> cancel_requested{false};
};

namespace {

constexpr size_t kRecentLogLines = 40;
constexpr uint64_t kGiB          = 1024ull * 1024ull * 1024ull;
// Headroom for compute buffers on top of the weight files.
constexpr uint64_t kLoadHeadroom = kGiB;

const char* const kComponentNames[BSD_COMPONENT_COUNT] = {
    "model", "diffusion", "high_noise_diffusion", "uncond_diffusion", "vae", "taesd", "clip_l",
    "clip_g", "clip_vision", "t5xxl", "llm", "llm_vision", "control_net", "ip_adapter",
    "photo_maker", "pulid", "motion_module", "audio_vae", "audio_encoder",
    "embeddings_connectors", "tokenizer",
};

const char* const kStatusNames[BSD_STATUS_COUNT] = {
    "ok", "out_of_memory", "missing_file", "bad_model", "cancelled", "unsupported",
    "invalid_argument", "internal", "not_loaded", "busy",
};

// One engine per process (upstream callbacks are process-global), so a global is enough.
bsd_engine* g_engine = nullptr;

bool is_set(const char* s) {
    return s != nullptr && s[0] != '\0';
}

void emit(bsd_engine* e, const bsd_event_t& ev) {
    if (e != nullptr && e->event_cb != nullptr) {
        e->event_cb(&ev, e->event_user);
    }
}

bsd_event_t make_event(bsd_event_type_t type) {
    bsd_event_t ev{};
    ev.size = sizeof(ev);
    ev.type = type;
    return ev;
}

void emit_stage(bsd_engine* e, const char* stage) {
    bsd_event_t ev = make_event(BSD_EVENT_STAGE);
    ev.stage       = stage;
    emit(e, ev);
}

bsd_status_t fail(bsd_engine* e, bsd_status_t status, std::string message, std::string detail = "") {
    e->error        = std::move(message);
    e->error_detail = std::move(detail);
    return status;
}

std::string recent_log_text(bsd_engine* e) {
    std::lock_guard<std::mutex> lock(e->log_mutex);
    std::string out;
    for (const auto& line : e->recent_log) {
        out += line;
        if (out.empty() || out.back() != '\n') {
            out += '\n';
        }
    }
    return out;
}

void clear_recent_log(bsd_engine* e) {
    std::lock_guard<std::mutex> lock(e->log_mutex);
    e->recent_log.clear();
}

void on_sd_log(sd_log_level_t level, const char* text, void*) {
    bsd_engine* e = g_engine;
    if (e == nullptr || text == nullptr) {
        return;
    }
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    if (line.empty()) {
        return;
    }
    if (level >= SD_LOG_INFO) {
        std::lock_guard<std::mutex> lock(e->log_mutex);
        e->recent_log.push_back(line);
        if (e->recent_log.size() > kRecentLogLines) {
            e->recent_log.pop_front();
        }
    }
    const char* stage       = nullptr;
    const char* timing_name = nullptr;
    float timing_ms         = 0;
    if (bsd_detect_stage(line.c_str(), &stage, &timing_name, &timing_ms)) {
        if (stage != nullptr) {
            emit_stage(e, stage);
        }
        if (timing_name != nullptr) {
            bsd_event_t ev  = make_event(BSD_EVENT_METRICS);
            ev.timing_name  = timing_name;
            ev.timing_ms    = timing_ms;
            emit(e, ev);
        }
    }
    bsd_event_t ev = make_event(BSD_EVENT_LOG);
    ev.level       = static_cast<bsd_log_level_t>(level);
    ev.text        = line.c_str();
    emit(e, ev);
}

void on_sd_preview(int step, int frame_count, sd_image_t* frames, bool is_noisy, void*) {
    bsd_engine* e = g_engine;
    if (e == nullptr || frames == nullptr || frame_count <= 0 || frames[0].data == nullptr) {
        return;
    }
    bsd_image_t image{frames[0].width, frames[0].height, frames[0].channel, frames[0].data};
    bsd_event_t ev = make_event(BSD_EVENT_PREVIEW);
    ev.step        = step;
    ev.steps       = sd_get_preview_info().total_steps;
    ev.preview     = &image;
    ev.is_noisy    = is_noisy ? 1 : 0;
    emit(e, ev);
}

void on_sd_progress(int step, int steps, float time, void*) {
    bsd_engine* e = g_engine;
    if (e == nullptr) {
        return;
    }
    bsd_event_t ev  = make_event(BSD_EVENT_PROGRESS);
    ev.step         = step;
    ev.steps        = steps;
    ev.sec_per_step = time;
    emit(e, ev);
}

// Size of a model source on disk. A directory or a safetensors index counts all its files.
uint64_t source_bytes(const std::string& path) {
    std::error_code ec;
    fs::path p = fs::u8path(path);
    if (fs::is_directory(p, ec)) {
        uint64_t total = 0;
        for (const auto& entry : fs::recursive_directory_iterator(p, ec)) {
            if (entry.is_regular_file(ec)) {
                total += entry.file_size(ec);
            }
        }
        return total;
    }
    if (p.extension() == ".json") {
        uint64_t total = 0;
        for (const auto& entry : fs::directory_iterator(p.parent_path(), ec)) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".safetensors") {
                total += entry.file_size(ec);
            }
        }
        return total;
    }
    uint64_t size = fs::file_size(p, ec);
    return ec ? 0 : size;
}

std::string gib(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / static_cast<double>(kGiB));
    return buf;
}

const char* device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accel";
        default:
            return "other";
    }
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parse_sigmas(const char* text, std::vector<float>& out) {
    std::string s = text;
    if (s == "qwen-image-2.1-turbo") {
        s = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0";
    }
    if (!s.empty() && s.front() == '[') {
        s.erase(0, 1);
    }
    if (!s.empty() && s.back() == ']') {
        s.pop_back();
    }
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        char* end   = nullptr;
        float value = std::strtof(item.c_str(), &end);
        if (end == item.c_str()) {
            return false;
        }
        out.push_back(value);
    }
    return !out.empty();
}

}  // namespace

extern "C" {

int bsd_api_version(void) {
    return BSD_API_VERSION;
}

const char* bsd_status_name(bsd_status_t status) {
    if (status < 0 || status >= BSD_STATUS_COUNT) {
        return "unknown";
    }
    return kStatusNames[status];
}

const char* bsd_component_name(bsd_component_t c) {
    if (c < 0 || c >= BSD_COMPONENT_COUNT) {
        return "unknown";
    }
    return kComponentNames[c];
}

bsd_status_t bsd_classify_failure(const char* log_text) {
    if (log_text == nullptr) {
        return BSD_ERR_INTERNAL;
    }
    const std::string text = lower(log_text);
    auto has               = [&](const char* needle) { return text.find(needle) != std::string::npos; };

    if (has("out of memory") || has("outofmemory") || has("failed to allocate") ||
        has("not enough memory") || has("insufficient memory") || has("alloc_buffer") ||
        has("erroroutofdevicememory") || has("cudamalloc failed")) {
        return BSD_ERR_OUT_OF_MEMORY;
    }
    if (has("no such file") || has("cannot inspect model source") || has("file not found") ||
        (has("file ") && has(" not found")) || has("failed to open")) {
        return BSD_ERR_MISSING_FILE;
    }
    if (has("get sd version from file failed") || has("unknown model") || has("unsupported model") ||
        has("unknown format") || has("shape mismatch") || has("missing tensor") ||
        has("invalid model") || has("failed to detect")) {
        return BSD_ERR_BAD_MODEL;
    }
    return BSD_ERR_INTERNAL;
}

int bsd_detect_stage(const char* line, const char** stage, const char** timing_name, float* timing_ms) {
    if (line == nullptr || stage == nullptr || timing_name == nullptr || timing_ms == nullptr) {
        return 0;
    }
    *stage       = nullptr;
    *timing_name = nullptr;
    *timing_ms   = 0;
    const std::string s(line);
    auto taking = [&](const char* name) {
        const size_t at = s.find("taking ");
        if (at == std::string::npos) {
            return false;
        }
        *timing_name = name;
        *timing_ms   = std::strtof(s.c_str() + at + 7, nullptr) * 1000.0f;
        return true;
    };
    struct Rule {
        const char* prefix;
        const char* stage;   // Stage that starts.
        const char* timing;  // Stage that ends ("taking Ns").
    };
    // The first matching prefix wins.
    static const Rule rules[] = {
        {"generate_image completed", nullptr, nullptr},
        {"generate_image ", "encode_prompt", nullptr},
        {"get_learned_condition completed", nullptr, "encode_prompt"},
        {"generating image: ", "sample", nullptr},
        {"sampling completed", nullptr, "sample"},
        {"decoding ", "decode", nullptr},
        {"decode_first_stage completed", nullptr, "decode"},
        {"loading tensors completed", nullptr, "load"},
    };
    for (const auto& rule : rules) {
        if (s.rfind(rule.prefix, 0) != 0) {
            continue;
        }
        *stage = rule.stage;
        if (rule.timing != nullptr) {
            taking(rule.timing);
        }
        return 1;
    }
    return 0;
}

bsd_engine_t* bsd_create(const bsd_config_t* config) {
    if (g_engine != nullptr) {
        return nullptr;  // One engine per process.
    }
    auto* e      = new bsd_engine();
    e->n_threads = (config != nullptr && config->n_threads > 0) ? config->n_threads : sd_get_num_physical_cores();
    g_engine     = e;
    sd_set_log_callback(on_sd_log, nullptr);
    sd_set_progress_callback(on_sd_progress, nullptr);
    return e;
}

void bsd_destroy(bsd_engine_t* e) {
    if (e == nullptr) {
        return;
    }
    if (e->ctx != nullptr) {
        free_sd_ctx(e->ctx);
    }
    sd_set_log_callback(nullptr, nullptr);
    sd_set_progress_callback(nullptr, nullptr);
    if (g_engine == e) {
        g_engine = nullptr;
    }
    delete e;
}

const char* bsd_last_error(const bsd_engine_t* e) {
    return e != nullptr ? e->error.c_str() : "no engine";
}

const char* bsd_last_error_detail(const bsd_engine_t* e) {
    return e != nullptr ? e->error_detail.c_str() : "";
}

void bsd_set_event_cb(bsd_engine_t* e, bsd_event_cb_t cb, void* user) {
    if (e == nullptr) {
        return;
    }
    e->event_cb   = cb;
    e->event_user = user;
}

bsd_status_t bsd_memory(bsd_engine_t* e, bsd_memory_t* out) {
    if (e == nullptr || out == nullptr) {
        return BSD_ERR_INVALID_ARGUMENT;
    }
    *out              = bsd_memory_t{};
    out->size         = sizeof(*out);
    const size_t count = std::min<size_t>(ggml_backend_dev_count(), BSD_MAX_DEVICES);
    for (size_t i = 0; i < count; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        bsd_device_memory_t& d = out->devices[out->device_count++];
        std::snprintf(d.name, sizeof(d.name), "%s", ggml_backend_dev_name(dev));
        std::snprintf(d.description, sizeof(d.description), "%s", ggml_backend_dev_description(dev));
        std::snprintf(d.type, sizeof(d.type), "%s", device_type_name(ggml_backend_dev_type(dev)));
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        d.free_bytes  = free_bytes;
        d.total_bytes = total_bytes;
    }
    return BSD_OK;
}

bsd_status_t bsd_info(bsd_engine_t* e, bsd_info_t* out) {
    if (e == nullptr || out == nullptr) {
        return BSD_ERR_INVALID_ARGUMENT;
    }
    *out              = bsd_info_t{};
    out->size         = sizeof(*out);
    out->api_version  = BSD_API_VERSION;
    out->sd_version   = sd_version();
    out->sd_commit    = sd_commit();
    out->system_info  = sd_get_system_info();
    out->capabilities = BSD_CAP_CANCEL | BSD_CAP_MEMORY;
    out->loaded       = e->ctx != nullptr ? 1 : 0;
    if (e->ctx != nullptr) {
        out->model_version = sd_get_model_version_name(e->ctx);
        if (sd_ctx_supports_image_generation(e->ctx)) {
            out->capabilities |= BSD_CAP_IMAGE;
        }
    }
    return BSD_OK;
}

bsd_status_t bsd_unload(bsd_engine_t* e) {
    if (e == nullptr) {
        return BSD_ERR_INVALID_ARGUMENT;
    }
    if (e->generating) {
        return fail(e, BSD_ERR_BUSY, "A job is running. Cancel it before you unload the model.");
    }
    if (e->ctx != nullptr) {
        free_sd_ctx(e->ctx);
        e->ctx = nullptr;
    }
    return BSD_OK;
}

bsd_status_t bsd_load(bsd_engine_t* e, const bsd_components_t* c, const bsd_load_options_t* o) {
    if (e == nullptr || c == nullptr) {
        return BSD_ERR_INVALID_ARGUMENT;
    }
    if (e->generating) {
        return fail(e, BSD_ERR_BUSY, "A job is running. Cancel it before you load a model.");
    }
    bsd_load_options_t opts{};
    if (o != nullptr) {
        opts = *o;
    }

    // 1. Every part must exist.
    uint64_t needed = 0;
    bool any        = false;
    for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
        const char* path = c->paths[i];
        if (!is_set(path)) {
            continue;
        }
        any = true;
        std::error_code ec;
        if (!fs::exists(fs::u8path(path), ec)) {
            return fail(e, BSD_ERR_MISSING_FILE,
                        std::string("Model file not found for part '") + kComponentNames[i] + "': " + path);
        }
        needed += source_bytes(path);
    }
    if (!any) {
        return fail(e, BSD_ERR_INVALID_ARGUMENT, "No model parts given.");
    }
    if (!is_set(c->paths[BSD_COMPONENT_MODEL]) && !is_set(c->paths[BSD_COMPONENT_DIFFUSION])) {
        return fail(e, BSD_ERR_INVALID_ARGUMENT, "Give a 'model' or a 'diffusion' part.");
    }

    // 2. Free memory must be large enough for the weight files.
    if (!opts.skip_memory_check) {
        bsd_memory_t mem{};
        bsd_memory(e, &mem);
        uint64_t gpu_free = 0, cpu_free = 0;
        std::string gpu_name;
        for (int i = 0; i < mem.device_count; i++) {
            const auto& d = mem.devices[i];
            if (std::strcmp(d.type, "cpu") == 0) {
                cpu_free = std::max<uint64_t>(cpu_free, d.free_bytes);
            } else if (d.free_bytes > gpu_free) {
                gpu_free = d.free_bytes;
                gpu_name = d.name;
            }
        }
        const std::string pb = lower(is_set(opts.params_backend) ? opts.params_backend : "");
        const bool uses_host = pb.find("cpu") != std::string::npos || pb.find("disk") != std::string::npos;
        uint64_t available   = gpu_free;
        std::string where    = gpu_name;
        if (gpu_free == 0) {
            available = cpu_free;
            where     = "CPU";
        } else if (uses_host) {
            available += cpu_free;
            where += " + CPU";
        }
        if (available > 0 && needed + kLoadHeadroom > available) {
            return fail(e, BSD_ERR_OUT_OF_MEMORY,
                        "Not enough memory to load this model. It needs about " + gib(needed + kLoadHeadroom) +
                            ". " + gib(available) + " is free on " + where +
                            ". Close other apps, use a smaller quantization, or put the text encoder on the CPU.");
        }
    }

    // 3. Replace the current context.
    if (e->ctx != nullptr) {
        free_sd_ctx(e->ctx);
        e->ctx = nullptr;
    }

    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    auto path = [&](bsd_component_t which) { return is_set(c->paths[which]) ? c->paths[which] : nullptr; };
    p.model_path                      = path(BSD_COMPONENT_MODEL);
    p.diffusion_model_path            = path(BSD_COMPONENT_DIFFUSION);
    p.high_noise_diffusion_model_path = path(BSD_COMPONENT_HIGH_NOISE_DIFFUSION);
    p.uncond_diffusion_model_path     = path(BSD_COMPONENT_UNCOND_DIFFUSION);
    p.vae_path                        = path(BSD_COMPONENT_VAE);
    p.taesd_path                      = path(BSD_COMPONENT_TAESD);
    p.clip_l_path                     = path(BSD_COMPONENT_CLIP_L);
    p.clip_g_path                     = path(BSD_COMPONENT_CLIP_G);
    p.clip_vision_path                = path(BSD_COMPONENT_CLIP_VISION);
    p.t5xxl_path                      = path(BSD_COMPONENT_T5XXL);
    p.llm_path                        = path(BSD_COMPONENT_LLM);
    p.llm_vision_path                 = path(BSD_COMPONENT_LLM_VISION);
    p.control_net_path                = path(BSD_COMPONENT_CONTROL_NET);
    p.ip_adapter_path                 = path(BSD_COMPONENT_IP_ADAPTER);
    p.photo_maker_path                = path(BSD_COMPONENT_PHOTO_MAKER);
    p.pulid_weights_path              = path(BSD_COMPONENT_PULID);
    p.motion_module_path              = path(BSD_COMPONENT_MOTION_MODULE);
    p.audio_vae_path                  = path(BSD_COMPONENT_AUDIO_VAE);
    p.audio_encoder_path              = path(BSD_COMPONENT_AUDIO_ENCODER);
    p.embeddings_connectors_path      = path(BSD_COMPONENT_EMBEDDINGS_CONNECTORS);
    p.tokenizer                       = path(BSD_COMPONENT_TOKENIZER);
    p.n_threads                       = e->n_threads;
    p.enable_mmap                     = opts.enable_mmap != 0;
    p.eager_load                      = opts.eager_load != 0;
    p.tae_preview_only                = opts.tae_preview_only != 0;
    if (is_set(opts.backend)) p.backend = opts.backend;
    if (is_set(opts.params_backend)) p.params_backend = opts.params_backend;
    if (is_set(opts.max_vram)) p.max_vram = opts.max_vram;
    if (is_set(opts.flash_attn_mode)) p.flash_attn_mode = opts.flash_attn_mode;
    if (is_set(opts.model_args)) p.model_args = opts.model_args;
    if (is_set(opts.weight_type)) {
        sd_type_t t = str_to_sd_type(opts.weight_type);
        if (t == SD_TYPE_COUNT) {
            return fail(e, BSD_ERR_INVALID_ARGUMENT, std::string("Unknown weight type: ") + opts.weight_type);
        }
        p.wtype = t;
    }

    clear_recent_log(e);
    emit_stage(e, "load");
    e->ctx = new_sd_ctx(&p);
    if (e->ctx == nullptr) {
        const std::string log = recent_log_text(e);
        const bsd_status_t st = bsd_classify_failure(log.c_str());
        const char* msg       = st == BSD_ERR_OUT_OF_MEMORY ? "Out of memory while loading the model."
                                : st == BSD_ERR_MISSING_FILE ? "A model file could not be read."
                                : st == BSD_ERR_BAD_MODEL    ? "The model files are not a supported model, or the parts do not match."
                                                             : "The model failed to load.";
        return fail(e, st, msg, log);
    }
    return BSD_OK;
}

void bsd_cancel(bsd_engine_t* e, bsd_cancel_mode_t mode) {
    if (e == nullptr || e->ctx == nullptr || !e->generating) {
        return;
    }
    e->cancel_requested = true;
    sd_cancel_generation(e->ctx, mode == BSD_CANCEL_NEW_LATENTS ? SD_CANCEL_NEW_LATENTS : SD_CANCEL_ALL);
}

bsd_status_t bsd_generate_image(bsd_engine_t* e, const bsd_image_params_t* p, bsd_result_t** out) {
    if (e == nullptr || p == nullptr || out == nullptr) {
        return BSD_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (e->ctx == nullptr) {
        return fail(e, BSD_ERR_NOT_LOADED, "The engine has no model. Load a model first.");
    }
    if (!sd_ctx_supports_image_generation(e->ctx)) {
        return fail(e, BSD_ERR_UNSUPPORTED, "The loaded model cannot generate images.");
    }
    if (p->width <= 0 || p->height <= 0 || p->prompt == nullptr) {
        return fail(e, BSD_ERR_INVALID_ARGUMENT, "Give a prompt, a width and a height.");
    }

    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt          = p->prompt;
    g.negative_prompt = p->negative_prompt != nullptr ? p->negative_prompt : "";
    g.width           = p->width;
    g.height          = p->height;
    g.batch_count     = p->batch_count > 0 ? p->batch_count : 1;
    if (p->steps > 0) g.sample_params.sample_steps = p->steps;
    if (p->cfg_scale > 0) g.sample_params.guidance.txt_cfg = p->cfg_scale;
    if (p->flow_shift != 0) g.sample_params.flow_shift = p->flow_shift;
    if (is_set(p->sample_method)) {
        sample_method_t m = str_to_sample_method(p->sample_method);
        if (m == SAMPLE_METHOD_COUNT) {
            return fail(e, BSD_ERR_INVALID_ARGUMENT, std::string("Unknown sample method: ") + p->sample_method);
        }
        g.sample_params.sample_method = m;
    }
    if (is_set(p->scheduler)) {
        scheduler_t s = str_to_scheduler(p->scheduler);
        if (s == SCHEDULER_COUNT) {
            return fail(e, BSD_ERR_INVALID_ARGUMENT, std::string("Unknown scheduler: ") + p->scheduler);
        }
        g.sample_params.scheduler = s;
    }
    std::vector<float> sigmas;
    if (is_set(p->sigmas)) {
        if (!parse_sigmas(p->sigmas, sigmas)) {
            return fail(e, BSD_ERR_INVALID_ARGUMENT, std::string("Bad sigmas: ") + p->sigmas);
        }
        g.sample_params.custom_sigmas       = sigmas.data();
        g.sample_params.custom_sigmas_count = static_cast<int>(sigmas.size());
    }
    int64_t seed = p->seed;
    if (seed < 0) {
        std::random_device rd;
        seed = static_cast<int64_t>(rd() & 0x7fffffff);
    }
    g.seed = seed;

    preview_t preview = PREVIEW_NONE;
    if (is_set(p->preview_mode) && std::strcmp(p->preview_mode, "none") != 0) {
        preview = str_to_preview(p->preview_mode);
        if (preview == PREVIEW_COUNT) {
            return fail(e, BSD_ERR_INVALID_ARGUMENT, std::string("Unknown preview mode: ") + p->preview_mode);
        }
    }
    sd_set_preview_callback(preview == PREVIEW_NONE ? nullptr : on_sd_preview, preview,
                            p->preview_interval > 0 ? p->preview_interval : 2, true, false, nullptr);

    clear_recent_log(e);
    e->cancel_requested = false;
    sd_cancel_generation(e->ctx, SD_CANCEL_RESET);
    e->generating = true;

    sd_image_generation_result_t* res = nullptr;
    const bool ok                     = generate_image(e->ctx, &g, &res);
    e->generating                     = false;

    if (e->cancel_requested && (!ok || res == nullptr || res->count == 0)) {
        if (res != nullptr) free_sd_image_generation_result(res);
        return fail(e, BSD_ERR_CANCELLED, "The job stopped because of a cancel request.");
    }
    if (!ok || res == nullptr) {
        const std::string log = recent_log_text(e);
        const bsd_status_t st = bsd_classify_failure(log.c_str());
        return fail(e, st, st == BSD_ERR_OUT_OF_MEMORY ? "Out of memory while generating the image." : "Image generation failed.", log);
    }

    auto* r   = static_cast<bsd_result_t*>(std::calloc(1, sizeof(bsd_result_t)));
    r->images = static_cast<bsd_image_t*>(std::calloc(std::max(res->count, 1), sizeof(bsd_image_t)));
    r->seeds  = static_cast<int64_t*>(std::calloc(std::max(res->count, 1), sizeof(int64_t)));
    for (int i = 0; i < res->count; i++) {
        sd_image_t& src = res->items[i].image;
        if (src.data == nullptr) {
            continue;
        }
        const size_t bytes = static_cast<size_t>(src.width) * src.height * src.channel;
        bsd_image_t& dst   = r->images[r->count];
        dst.width          = src.width;
        dst.height         = src.height;
        dst.channels       = src.channel;
        dst.data           = static_cast<uint8_t*>(std::malloc(bytes));
        std::memcpy(dst.data, src.data, bytes);
        r->seeds[r->count] = seed + i;
        r->count++;
    }
    free_sd_image_generation_result(res);
    if (r->count == 0) {
        bsd_free_result(r);
        if (e->cancel_requested) {
            return fail(e, BSD_ERR_CANCELLED, "The job stopped because of a cancel request.");
        }
        return fail(e, BSD_ERR_INTERNAL, "Generation returned no images.", recent_log_text(e));
    }
    *out = r;
    return BSD_OK;
}

void bsd_free_result(bsd_result_t* r) {
    if (r == nullptr) {
        return;
    }
    for (int i = 0; i < r->count; i++) {
        std::free(r->images[i].data);
    }
    std::free(r->images);
    std::free(r->seeds);
    std::free(r);
}

}  // extern "C"
