// brancusi-engine: runs bsd_* calls for the Brancusi core.
//
// Transport: JSON lines. Commands on stdin, protocol messages on stdout, raw text on stderr.
// The engine exits when stdin closes. See brancusi-desktop _docs/sd-cpp-interface.md section 3.4.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define bsd_dup _dup
#define bsd_dup2 _dup2
#define bsd_fdopen _fdopen
#else
#include <unistd.h>
#define bsd_dup dup
#define bsd_dup2 dup2
#define bsd_fdopen fdopen
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include "bsd.h"

#ifndef BRANCUSI_ENGINE_VERSION
#define BRANCUSI_ENGINE_VERSION "unknown"
#endif
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// ---- Output: one writer thread, so callbacks never block on the pipe. ----

class Writer {
public:
    explicit Writer(FILE* out) : out_(out), thread_([this] { run(); }) {}

    ~Writer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void send(const json& message) {
        std::string line = message.dump(-1, ' ', false, json::error_handler_t::replace);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(line));
        }
        cv_.notify_one();
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return closing_ || !queue_.empty(); });
            while (!queue_.empty()) {
                std::string line = std::move(queue_.front());
                queue_.pop_front();
                lock.unlock();
                std::fputs(line.c_str(), out_);
                std::fputc('\n', out_);
                std::fflush(out_);
                lock.lock();
            }
            if (closing_) {
                return;
            }
        }
    }

    FILE* out_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    bool closing_ = false;
    std::thread thread_;
};

// ---- Long work (load, unload, generate, convert, inspect) runs on one worker thread. ----

class Worker {
public:
    Worker() : thread_([this] { run(); }) {}

    ~Worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    // True when no work runs or waits. Only the thread that posts work can rely on it.
    bool idle() {
        std::lock_guard<std::mutex> lock(mutex_);
        return !busy_ && !task_;
    }

    // Returns false if work is already running or waiting.
    bool try_post(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (busy_ || task_) {
            return false;
        }
        task_ = std::move(task);
        cv_.notify_one();
        return true;
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return closing_ || task_; });
            if (!task_) {
                return;
            }
            auto task = std::move(task_);
            task_     = nullptr;
            busy_     = true;
            lock.unlock();
            task();
            lock.lock();
            busy_ = false;
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> task_;
    bool busy_    = false;
    bool closing_ = false;
    std::thread thread_;
};

using Clock = std::chrono::steady_clock;

struct JobStats {
    int preview_max_px = 256;
    Clock::time_point last_preview{};
    Clock::time_point last_memory{};
    json timings = json::object();
    json stages  = json::array();  // Stages with exact times from Layer 2 hooks.
    uint64_t min_free_bytes = UINT64_MAX;
    uint64_t total_bytes    = 0;
};

struct State {
    bsd_engine_t* engine = nullptr;
    Writer* writer       = nullptr;
    // The command and job that own the events from the engine callback.
    std::mutex current_mutex;
    std::string current_id;
    std::string current_job;
    int log_level = BSD_LOG_INFO;
    // Only the worker thread uses this (engine callbacks run on the worker thread).
    JobStats stats;
};

constexpr auto kPreviewMinGap = std::chrono::milliseconds(250);  // At most 4 previews per second.
constexpr auto kMemoryMinGap  = std::chrono::seconds(1);

State g;

std::string str_or(const json& j, const char* key, const std::string& fallback = "") {
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : fallback;
}

const char* c_or_null(const std::string& s) {
    return s.empty() ? nullptr : s.c_str();
}

void set_current(const std::string& id, const std::string& job) {
    std::lock_guard<std::mutex> lock(g.current_mutex);
    g.current_id  = id;
    g.current_job = job;
}

json tagged(json message) {
    std::lock_guard<std::mutex> lock(g.current_mutex);
    if (!g.current_id.empty()) message["id"] = g.current_id;
    if (!g.current_job.empty()) message["job"] = g.current_job;
    return message;
}

json error_message(const std::string& id, bsd_status_t status, const std::string& message,
                   const std::string& detail = "", const std::string& job = "") {
    json m = {{"event", "error"}, {"code", bsd_status_name(status)}, {"message", message}};
    if (!id.empty()) m["id"] = id;
    if (!job.empty()) m["job"] = job;
    if (!detail.empty()) m["detail"] = detail;
    return m;
}

json engine_error(const std::string& id, bsd_status_t status, const std::string& job = "") {
    return error_message(id, status, bsd_last_error(g.engine), bsd_last_error_detail(g.engine), job);
}

json devices_json() {
    bsd_memory_t mem{};
    mem.size = sizeof(mem);
    bsd_memory(g.engine, &mem);
    json devices = json::array();
    for (int i = 0; i < mem.device_count; i++) {
        const auto& d = mem.devices[i];
        devices.push_back({{"name", d.name},
                           {"description", d.description},
                           {"type", d.type},
                           {"free_mib", d.free_bytes / (1024 * 1024)},
                           {"total_mib", d.total_bytes / (1024 * 1024)}});
    }
    return devices;
}

json capabilities_json(uint64_t caps) {
    json out = json::array();
    const std::pair<uint64_t, const char*> names[] = {
        {BSD_CAP_IMAGE, "image"},     {BSD_CAP_VIDEO, "video"},           {BSD_CAP_CANCEL, "cancel"},
        {BSD_CAP_PREVIEW, "preview"}, {BSD_CAP_MEMORY, "memory"},         {BSD_CAP_UPSCALE, "upscale"},
        {BSD_CAP_SET_COMPONENT, "set_component"}, {BSD_CAP_RUN_STAGE, "run_stage"},
        {BSD_CAP_AUTO_FIT, "auto_fit"}, {BSD_CAP_PART_MEMORY, "part_memory"}, {BSD_CAP_STAGE_TIMING, "stage_timing"},
    };
    for (const auto& [flag, name] : names) {
        if (caps & flag) out.push_back(name);
    }
    return out;
}

// Weight memory of each part of the loaded model. Null if no model is loaded or a job runs.
json parts_json() {
    bsd_parts_t parts{};
    parts.size = sizeof(parts);
    if (bsd_parts(g.engine, &parts) != BSD_OK) return nullptr;
    json out = json::array();
    for (int i = 0; i < parts.count; i++) {
        const bsd_part_memory_t& p = parts.parts[i];
        out.push_back({{"name", p.name},
                       {"backend", p.backend},
                       {"bytes", p.bytes},
                       {"bytes_gpu", p.bytes_gpu},
                       {"bytes_cpu", p.bytes_cpu},
                       {"on_disk", p.on_disk}});
    }
    return out;
}

json info_json() {
    bsd_info_t info{};
    info.size = sizeof(info);
    bsd_info(g.engine, &info);
    json caps = capabilities_json(info.capabilities);
    for (const char* command : {"convert", "inspect", "schedulers", "hardware", "peak_bytes"}) caps.push_back(command);
    json m = {{"version", BRANCUSI_ENGINE_VERSION},
              {"bsd_api_version", info.api_version},
              {"sd_version", info.sd_version ? info.sd_version : ""},
              {"sd_commit", info.sd_commit ? info.sd_commit : ""},
              {"system_info", info.system_info ? info.system_info : ""},
              {"capabilities", caps},
              {"loaded", info.loaded != 0},
              {"devices", devices_json()}};
    if (info.model_version) m["model_version"] = info.model_version;
    if (info.loaded) {
        m["default_sample_method"] = info.default_sample_method ? info.default_sample_method : "";
        m["default_scheduler"]     = info.default_scheduler ? info.default_scheduler : "";
        m["image_generation"]      = (info.capabilities & BSD_CAP_IMAGE) != 0;
        m["video_generation"]      = (info.capabilities & BSD_CAP_VIDEO) != 0;
        m["params_backend"]        = info.params_backend ? info.params_backend : "";
    }
    return m;
}

// "SSE3 = 1 | AVX = 0 | " to {"SSE3": true, "AVX": false}. The engine ignores items without "=".
json system_features(const std::string& text) {
    json out = json::object();
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('|', start);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(start, end - start);
        const size_t eq        = item.find('=');
        if (eq != std::string::npos) {
            auto trim = [](std::string v) {
                const size_t a = v.find_first_not_of(" \t\n");
                const size_t b = v.find_last_not_of(" \t\n");
                return a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
            };
            std::string key = item.substr(0, eq);
            const size_t head = key.find_last_of(":\n");  // Drop the "System Info:" title.
            if (head != std::string::npos) key = key.substr(head + 1);
            key = trim(key);
            const std::string value = trim(item.substr(eq + 1));
            if (!key.empty()) out[key] = value != "0";
        }
        start = end + 1;
    }
    return out;
}

json hardware_json() {
    bsd_info_t info{};
    info.size = sizeof(info);
    bsd_info(g.engine, &info);
    const std::string system_info = info.system_info ? info.system_info : "";
    return {{"system_info", system_info},
            {"features", system_features(system_info)},
            {"physical_cores", info.physical_cores},
            {"n_threads", info.n_threads},
            {"devices", devices_json()}};
}

json table_json(bsd_table_t table) {
    json out = json::array();
    for (int32_t i = 0; i < bsd_table_size(table); i++) {
        const char* name = bsd_table_name(table, i);
        if (name != nullptr && name[0] != '\0') out.push_back(name);
    }
    return out;
}

json schedulers_json() {
    return {{"samplers", table_json(BSD_TABLE_SAMPLE_METHOD)},
            {"schedulers", table_json(BSD_TABLE_SCHEDULER)},
            {"types", table_json(BSD_TABLE_WEIGHT_TYPE)}};
}

const char kBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const std::string& in) {
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t n = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
        out += kBase64[(n >> 18) & 63];
        out += kBase64[(n >> 12) & 63];
        out += kBase64[(n >> 6) & 63];
        out += kBase64[n & 63];
    }
    if (i < in.size()) {
        uint32_t n = uint8_t(in[i]) << 16;
        if (i + 1 < in.size()) n |= uint8_t(in[i + 1]) << 8;
        out += kBase64[(n >> 18) & 63];
        out += kBase64[(n >> 12) & 63];
        out += i + 1 < in.size() ? kBase64[(n >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// Shrink to max_px on the long side (box filter), then encode as JPEG.
std::string preview_jpeg(const bsd_image_t& img, int max_px) {
    const int w = static_cast<int>(img.width), h = static_cast<int>(img.height), c = static_cast<int>(img.channels);
    const int scale = std::max(1, (std::max(w, h) + max_px - 1) / max_px);
    const int ow = std::max(1, w / scale), oh = std::max(1, h / scale);
    std::vector<uint8_t> small(static_cast<size_t>(ow) * oh * 3);
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            for (int ch = 0; ch < 3; ch++) {
                int sum = 0, n = 0;
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        const int sx = x * scale + dx, sy = y * scale + dy;
                        if (sx < w && sy < h) {
                            sum += img.data[(static_cast<size_t>(sy) * w + sx) * c + std::min(ch, c - 1)];
                            n++;
                        }
                    }
                }
                small[(static_cast<size_t>(y) * ow + x) * 3 + ch] = static_cast<uint8_t>(n ? sum / n : 0);
            }
        }
    }
    std::string bytes;
    stbi_write_jpg_to_func(
        [](void* ctx, void* data, int size) { static_cast<std::string*>(ctx)->append(static_cast<char*>(data), size); },
        &bytes, ow, oh, 3, small.data(), 80);
    return bytes;
}

// The main GPU (the first one), or the CPU if there is no GPU. NULL if no device reports memory.
const bsd_device_memory_t* main_device(const bsd_memory_t& mem) {
    const bsd_device_memory_t* best = nullptr;
    for (int i = 0; i < mem.device_count; i++) {
        const auto& d = mem.devices[i];
        if (d.total_bytes == 0) continue;
        const bool gpu = std::strcmp(d.type, "cpu") != 0;
        if (best == nullptr || (gpu && std::strcmp(best->type, "cpu") == 0)) best = &d;
    }
    return best;
}

// Reads the main device memory at a short interval while a job runs. The events come only at
// stage changes and steps, so they can miss the peak of a lazy load or a VAE decode.
class PeakSampler {
public:
    PeakSampler() : thread_([this] { run(); }) {}

    ~PeakSampler() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    uint64_t peak_bytes() const { return peak_.load(); }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        do {
            lock.unlock();
            sample();
            lock.lock();
        } while (!cv_.wait_for(lock, kInterval, [this] { return stopping_; }));
        lock.unlock();
        sample();
    }

    void sample() {
        bsd_memory_t mem{};
        mem.size = sizeof(mem);
        bsd_memory(g.engine, &mem);
        const bsd_device_memory_t* d = main_device(mem);
        if (d == nullptr) return;
        const uint64_t used = d->total_bytes > d->free_bytes ? d->total_bytes - d->free_bytes : 0;
        uint64_t prev       = peak_.load();
        while (used > prev && !peak_.compare_exchange_weak(prev, used)) {
        }
    }

    static constexpr auto kInterval = std::chrono::milliseconds(50);
    std::atomic<uint64_t> peak_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::thread thread_;
};

// Send device memory for the main GPU (or the CPU). Track the lowest free memory for peak use.
void send_memory(bool force) {
    const auto now = Clock::now();
    if (!force && now - g.stats.last_memory < kMemoryMinGap) return;
    g.stats.last_memory = now;
    bsd_memory_t mem{};
    mem.size = sizeof(mem);
    bsd_memory(g.engine, &mem);
    const bsd_device_memory_t* best = main_device(mem);
    if (best == nullptr) return;
    g.stats.min_free_bytes = std::min<uint64_t>(g.stats.min_free_bytes, best->free_bytes);
    g.stats.total_bytes    = best->total_bytes;
    g.writer->send(tagged({{"event", "metrics"},
                           {"device", best->name},
                           {"used_mib", (best->total_bytes - best->free_bytes) / (1024 * 1024)},
                           {"free_mib", best->free_bytes / (1024 * 1024)}}));
}

void on_event(const bsd_event_t* ev, void*) {
    switch (ev->type) {
        case BSD_EVENT_LOG: {
            if (ev->level < g.log_level) return;
            static const char* levels[] = {"debug", "verbose", "info", "warn", "error"};
            g.writer->send(tagged({{"event", "log"}, {"level", levels[ev->level]}, {"text", ev->text}}));
            return;
        }
        case BSD_EVENT_PROGRESS:
            g.writer->send(tagged({{"event", "progress"},
                                   {"step", ev->step},
                                   {"steps", ev->steps},
                                   {"sec_per_step", ev->sec_per_step}}));
            send_memory(false);
            return;
        case BSD_EVENT_STAGE: {
            // A stage end from a Layer 2 hook comes again in the metrics event with the times.
            if (ev->stage_phase == BSD_STAGE_END) return;
            json m = {{"event", "stage"}, {"stage", ev->stage}};
            if (ev->stage_phase == BSD_STAGE_BEGIN) m["start_us"] = ev->start_us;
            g.writer->send(tagged(m));
            send_memory(true);
            return;
        }
        case BSD_EVENT_METRICS: {
            if (ev->timing_name == nullptr) return;
            // Loading can happen more than once in a job (lazy load). Add the times.
            const std::string key = std::string(ev->timing_name) + "_ms";
            const double prev     = g.stats.timings.value(key, 0.0);
            g.stats.timings[key]  = std::round(prev + ev->timing_ms);
            json m                = {{"event", "metrics"}, {"timings", {{key, std::round(ev->timing_ms)}}}};
            if (ev->stage_phase == BSD_STAGE_END) {
                const json span = {{"stage", ev->timing_name},
                                   {"start_us", ev->start_us},
                                   {"end_us", ev->end_us},
                                   {"ms", static_cast<double>(ev->end_us - ev->start_us) / 1000.0}};
                g.stats.stages.push_back(span);
                m["span"] = span;
            }
            g.writer->send(tagged(m));
            return;
        }
        case BSD_EVENT_PREVIEW: {
            const auto now = Clock::now();
            if (ev->preview == nullptr || now - g.stats.last_preview < kPreviewMinGap) return;
            g.stats.last_preview = now;
            const std::string jpeg = preview_jpeg(*ev->preview, g.stats.preview_max_px);
            if (jpeg.empty()) return;
            g.writer->send(tagged({{"event", "preview"},
                                   {"step", ev->step},
                                   {"steps", ev->steps},
                                   {"mime", "image/jpeg"},
                                   {"data", base64(jpeg)}}));
            return;
        }
        default:
            return;
    }
}

long long ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

// ---- Commands ----

// Model parts of a command: part name to path. The strings stay alive while this object lives.
struct Parts {
    std::string paths[BSD_COMPONENT_COUNT];
    bsd_components_t components{};
    std::string bad_key;  // Set if a key is not a part name or a value is not a string.

    explicit Parts(const json& comps) {
        components.size = sizeof(components);
        if (!comps.is_object()) {
            bad_key = "components";
            return;
        }
        for (auto it = comps.begin(); it != comps.end(); ++it) {
            int index = -1;
            for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
                if (it.key() == bsd_component_name(static_cast<bsd_component_t>(i))) index = i;
            }
            if (index < 0 || !it->is_string()) {
                bad_key = it.key();
                return;
            }
            paths[index]            = it->get<std::string>();
            components.paths[index] = paths[index].c_str();
        }
    }
};

json part_info_json(const bsd_part_info_t& p, bool sizes) {
    json types = json::object();
    for (int t = 0; t < p.type_count; t++) types[p.types[t].name] = p.types[t].tensors;
    json m = {{"tensors", p.tensors}, {"types", types}};
    if (p.type_count > 0) m["type"] = p.types[0].name;
    if (sizes) {
        m["params"] = p.params;
        m["bytes"]  = p.bytes;
    }
    return m;
}

void do_inspect(const std::string& id, const json& cmd) {
    set_current(id, "");
    Parts parts(cmd.value("components", json::object()));
    if (!parts.bad_key.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown model part: " + parts.bad_key));
        set_current("", "");
        return;
    }
    const std::string weight_type = str_or(cmd, "weight_type");
    const auto start              = std::chrono::steady_clock::now();
    bsd_inspect_t r{};
    r.size                = sizeof(r);
    const bsd_status_t st = bsd_inspect(g.engine, &parts.components, c_or_null(weight_type), &r);
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st));
        set_current("", "");
        return;
    }
    json part_map = json::object();
    for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
        if (r.parts[i].present) part_map[bsd_component_name(static_cast<bsd_component_t>(i))] = part_info_json(r.parts[i], true);
    }
    json contents = json::object();
    for (int i = 0; i < BSD_CONTENT_COUNT; i++) {
        if (r.contents[i].present) contents[bsd_content_name(static_cast<bsd_content_t>(i))] = part_info_json(r.contents[i], false);
    }
    g.writer->send({{"id", id},
                    {"event", "inspect"},
                    {"family", r.family},
                    {"known", r.family[0] != '\0'},
                    {"parts", part_map},
                    {"contents", contents},
                    {"tensors", r.tensors},
                    {"params", r.params},
                    {"mem_bytes", r.mem_bytes},
                    {"mem_mib", r.mem_bytes / (1024 * 1024)},
                    {"ms", ms_since(start)}});
    set_current("", "");
}

void do_load(const std::string& id, const json& cmd) {
    set_current(id, "");
    Parts parts(cmd.value("components", json::object()));
    if (!parts.bad_key.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown model part: " + parts.bad_key));
        set_current("", "");
        return;
    }
    bsd_components_t& components = parts.components;

    const json o = cmd.value("options", json::object());
    const std::string backend = str_or(o, "backend"), params_backend = str_or(o, "params_backend"),
                      max_vram = str_or(o, "max_vram"), flash = str_or(o, "flash_attn"),
                      model_args = str_or(o, "model_args"), weight_type = str_or(o, "weight_type");
    bsd_load_options_t opts{};
    opts.size              = sizeof(opts);
    opts.backend           = c_or_null(backend);
    opts.params_backend    = c_or_null(params_backend);
    opts.max_vram          = c_or_null(max_vram);
    opts.flash_attn_mode   = c_or_null(flash);
    opts.model_args        = c_or_null(model_args);
    opts.weight_type       = c_or_null(weight_type);
    opts.enable_mmap       = o.value("mmap", false) ? 1 : 0;
    opts.eager_load        = o.value("eager_load", false) ? 1 : 0;
    opts.tae_preview_only  = o.value("tae_preview_only", false) ? 1 : 0;
    opts.skip_memory_check = o.value("skip_memory_check", false) ? 1 : 0;
    opts.auto_fit          = BSD_AUTO_FIT_DEFAULT;
    if (o.contains("auto_fit") && !o["auto_fit"].is_null()) {
        const json& af = o["auto_fit"];
        if (af.is_boolean()) {
            opts.auto_fit = af.get<bool>() ? BSD_AUTO_FIT_ON : BSD_AUTO_FIT_OFF;
        } else if (af.is_string() && (af == "on" || af == "off")) {
            opts.auto_fit = af == "on" ? BSD_AUTO_FIT_ON : BSD_AUTO_FIT_OFF;
        } else {
            g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "auto_fit must be true, false, \"on\" or \"off\"."));
            set_current("", "");
            return;
        }
    }

    const auto start        = std::chrono::steady_clock::now();
    const bsd_status_t st   = bsd_load(g.engine, &components, &opts);
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st));
    } else {
        json m    = info_json();
        m["id"]    = id;
        m["event"] = "loaded";
        m["ms"]    = ms_since(start);
        if (json parts = parts_json(); !parts.is_null()) m["parts"] = parts;
        g.writer->send(m);
    }
    set_current("", "");
}

// Layer 2: load, replace or remove one part of the loaded model. The reply is `loaded`.
void do_set_component(const std::string& id, const json& cmd) {
    set_current(id, "");
    const std::string name = str_or(cmd, "component");
    int index              = -1;
    for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
        if (name == bsd_component_name(static_cast<bsd_component_t>(i))) index = i;
    }
    const bool has_path = cmd.contains("path") && !cmd["path"].is_null();
    if (index < 0 || (has_path && !cmd["path"].is_string())) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT,
                                     index < 0 ? "Unknown model part: " + name : "path must be a string or null."));
        set_current("", "");
        return;
    }
    const std::string path = has_path ? cmd["path"].get<std::string>() : "";
    const auto start       = std::chrono::steady_clock::now();
    const bsd_status_t st  = bsd_set_component(g.engine, static_cast<bsd_component_t>(index), c_or_null(path));
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st));
    } else {
        json m         = info_json();
        m["id"]        = id;
        m["event"]     = "loaded";
        m["ms"]        = ms_since(start);
        m["component"] = name;
        m["path"]      = has_path ? json(path) : json(nullptr);
        if (json parts = parts_json(); !parts.is_null()) m["parts"] = parts;
        g.writer->send(m);
    }
    set_current("", "");
}

void do_generate(const std::string& id, const json& cmd) {
    const std::string job        = str_or(cmd, "job", id);
    const std::string output_dir = str_or(cmd, "output_dir");
    const json p                 = cmd.value("params", json::object());
    set_current(id, job);

    if (output_dir.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "output_dir is required.", "", job));
        set_current("", "");
        return;
    }
    const std::string prompt = str_or(p, "prompt"), negative = str_or(p, "negative_prompt"),
                      method = str_or(p, "sample_method"), scheduler = str_or(p, "scheduler"),
                      sigmas = str_or(p, "sigmas");
    const json preview             = cmd.value("preview", json::object());
    const std::string preview_mode = str_or(preview, "mode", "none");
    g.stats                        = JobStats{};
    g.stats.preview_max_px         = std::clamp(preview.value("max_px", 256), 32, 1024);
    bsd_image_params_t params{};
    params.size            = sizeof(params);
    params.prompt          = prompt.c_str();
    params.negative_prompt = negative.c_str();
    params.width           = p.value("width", 0);
    params.height          = p.value("height", 0);
    params.steps           = p.value("steps", 0);
    params.seed            = p.value("seed", -1LL);
    params.cfg_scale       = p.value("cfg_scale", 0.0f);
    params.sample_method   = c_or_null(method);
    params.scheduler       = c_or_null(scheduler);
    params.sigmas          = c_or_null(sigmas);
    params.batch_count     = p.value("batch_count", 1);
    params.flow_shift      = p.value("flow_shift", 0.0f);
    params.preview_mode    = preview_mode.c_str();
    params.preview_interval = preview.value("interval", 2);

    const auto start      = std::chrono::steady_clock::now();
    bsd_result_t* result  = nullptr;
    PeakSampler sampler;
    const bsd_status_t st = bsd_generate_image(g.engine, &params, &result);
    sampler.stop();
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st, job));
        set_current("", "");
        return;
    }

    std::error_code ec;
    fs::create_directories(fs::u8path(output_dir), ec);
    json outputs = json::array();
    for (int i = 0; i < result->count; i++) {
        const bsd_image_t& img = result->images[i];
        const fs::path path    = fs::u8path(output_dir) / (std::to_string(i) + ".png");
        const std::string path_str = path.u8string();
        if (!stbi_write_png(path_str.c_str(), img.width, img.height, img.channels, img.data, img.width * img.channels)) {
            g.writer->send(error_message(id, BSD_ERR_INTERNAL, "Could not write the image to " + path_str, "", job));
            bsd_free_result(result);
            set_current("", "");
            return;
        }
        outputs.push_back({{"path", path_str}, {"width", img.width}, {"height", img.height}, {"seed", result->seeds[i]}});
    }
    bsd_free_result(result);
    json timings        = g.stats.timings;
    timings["total_ms"] = ms_since(start);
    json done = {{"id", id}, {"event", "done"}, {"job", job}, {"outputs", outputs}, {"timings", timings}};
    if (!g.stats.stages.empty()) done["stages"] = g.stats.stages;
    if (json parts = parts_json(); !parts.is_null()) done["parts"] = parts;
    uint64_t peak = sampler.peak_bytes();
    if (g.stats.min_free_bytes != UINT64_MAX && g.stats.total_bytes > g.stats.min_free_bytes) {
        peak = std::max<uint64_t>(peak, g.stats.total_bytes - g.stats.min_free_bytes);
    }
    if (peak > 0) {
        done["peak_bytes"] = peak;
        done["peak_mib"]   = peak / (1024 * 1024);
    }
    g.writer->send(done);
    set_current("", "");
}

void do_convert(const std::string& id, const json& cmd) {
    set_current(id, "");
    g.stats = JobStats{};
    const std::string input = str_or(cmd, "input"), output = str_or(cmd, "output"), type = str_or(cmd, "type");
    std::string rules;
    if (cmd.contains("rules") && !cmd["rules"].is_null()) {
        const json& r = cmd["rules"];
        bool ok       = r.is_string() || r.is_array();
        if (r.is_string()) {
            rules = r.get<std::string>();
        } else if (r.is_array()) {
            for (const auto& item : r) {
                if (!item.is_string()) {
                    ok = false;
                    break;
                }
                if (!rules.empty()) {
                    rules += ",";
                }
                rules += item.get<std::string>();
            }
        }
        if (!ok) {
            g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "rules must be a string or a list of strings."));
            set_current("", "");
            return;
        }
    }
    const int32_t convert_name = cmd.value("convert_name", false) ? 1 : 0;
    // Either one `input` file, or `components` (part name to path) to merge into one file.
    const bool has_parts = cmd.contains("components");
    if (has_parts == !input.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Give either input or components."));
        set_current("", "");
        return;
    }
    Parts parts(has_parts ? cmd["components"] : json::object());
    if (!parts.bad_key.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown model part: " + parts.bad_key));
        set_current("", "");
        return;
    }
    const auto start      = std::chrono::steady_clock::now();
    const bsd_status_t st = has_parts
                                ? bsd_convert_parts(g.engine, &parts.components, c_or_null(output), c_or_null(type),
                                                    c_or_null(rules), convert_name)
                                : bsd_convert(g.engine, c_or_null(input), c_or_null(output), c_or_null(type),
                                              c_or_null(rules), convert_name);
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st));
    } else {
        std::error_code ec;
        const auto bytes = fs::file_size(fs::u8path(output), ec);
        g.writer->send({{"id", id},
                        {"event", "ok"},
                        {"output", output},
                        {"bytes", ec ? 0 : static_cast<uint64_t>(bytes)},
                        {"ms", ms_since(start)}});
    }
    set_current("", "");
}

void handle(const std::string& line, Worker& worker) {
    json cmd;
    try {
        cmd = json::parse(line);
    } catch (const std::exception& ex) {
        g.writer->send(error_message("", BSD_ERR_INVALID_ARGUMENT, std::string("Bad JSON: ") + ex.what()));
        return;
    }
    if (!cmd.is_object()) {
        g.writer->send(error_message("", BSD_ERR_INVALID_ARGUMENT, "A command must be a JSON object."));
        return;
    }
    const std::string id   = str_or(cmd, "id");
    const std::string name = str_or(cmd, "cmd");

    if (name == "info") {
        json m    = info_json();
        m["id"]    = id;
        m["event"] = "info";
        g.writer->send(m);
    } else if (name == "memory") {
        json m = {{"id", id}, {"event", "memory"}, {"devices", devices_json()}};
        // Only this thread posts work, so an idle worker stays idle while the call runs.
        if (worker.idle()) {
            if (json parts = parts_json(); !parts.is_null()) m["parts"] = parts;
        }
        g.writer->send(m);
    } else if (name == "hardware") {
        json m    = hardware_json();
        m["id"]    = id;
        m["event"] = "hardware";
        g.writer->send(m);
    } else if (name == "schedulers") {
        json m    = schedulers_json();
        m["id"]    = id;
        m["event"] = "schedulers";
        g.writer->send(m);
    } else if (name == "cancel") {
        const std::string job = str_or(cmd, "job");
        std::string running_job;
        {
            std::lock_guard<std::mutex> lock(g.current_mutex);
            running_job = g.current_job;
        }
        if (!running_job.empty() && (job.empty() || job == running_job)) {
            bsd_cancel(g.engine, str_or(cmd, "mode") == "new_latents" ? BSD_CANCEL_NEW_LATENTS : BSD_CANCEL_ALL);
        }
        g.writer->send({{"id", id}, {"event", "ok"}});
    } else if (name == "load" || name == "unload" || name == "generate" || name == "convert" || name == "inspect" ||
               name == "set_component") {
        std::function<void()> task;
        if (name == "inspect") {
            task = [id, cmd] { do_inspect(id, cmd); };
        } else if (name == "convert") {
            task = [id, cmd] { do_convert(id, cmd); };
        } else if (name == "load") {
            task = [id, cmd] { do_load(id, cmd); };
        } else if (name == "set_component") {
            task = [id, cmd] { do_set_component(id, cmd); };
        } else if (name == "generate") {
            task = [id, cmd] { do_generate(id, cmd); };
        } else {
            task = [id] {
                const bsd_status_t st = bsd_unload(g.engine);
                g.writer->send(st == BSD_OK ? json{{"id", id}, {"event", "unloaded"}} : engine_error(id, st));
            };
        }
        if (!worker.try_post(std::move(task))) {
            g.writer->send(error_message(id, BSD_ERR_BUSY, "The engine is busy. Wait for the current command, or cancel it.",
                                         "", str_or(cmd, "job")));
        }
    } else if (name == "log_level") {
        static const char* levels[] = {"debug", "verbose", "info", "warn", "error"};
        const std::string level     = str_or(cmd, "level", "info");
        for (int i = 0; i < 5; i++) {
            if (level == levels[i]) g.log_level = i;
        }
        g.writer->send({{"id", id}, {"event", "ok"}});
    } else {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown command: " + name));
    }
}

}  // namespace

int main() {
    // Only protocol messages go to stdout. Keep the real stdout for the protocol, and send
    // anything else that writes to fd 1 (upstream printf, ggml) to stderr.
    std::fflush(stdout);
    const int proto_fd = bsd_dup(1);
    bsd_dup2(2, 1);
#ifdef _WIN32
    _setmode(proto_fd, _O_BINARY);
#endif
    FILE* proto = bsd_fdopen(proto_fd, "w");
    if (proto == nullptr) {
        std::fprintf(stderr, "brancusi-engine: cannot open the protocol stream\n");
        return 1;
    }

    Writer writer(proto);
    g.writer = &writer;

    bsd_config_t config{};
    config.size = sizeof(config);
    g.engine    = bsd_create(&config);
    if (g.engine == nullptr) {
        writer.send(error_message("", BSD_ERR_INTERNAL, "Could not create the engine."));
        return 1;
    }
    bsd_set_event_cb(g.engine, on_event, nullptr);

    json hello     = info_json();
    hello["event"] = "hello";
    writer.send(hello);

    {
        Worker worker;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.find("\"shutdown\"") != std::string::npos) {
                try {
                    if (json::parse(line).value("cmd", "") == "shutdown") break;
                } catch (...) {
                }
            }
            handle(line, worker);
        }
        // stdin closed or shutdown: stop the running job, then let the worker finish.
        bsd_cancel(g.engine, BSD_CANCEL_ALL);
    }
    bsd_destroy(g.engine);
    g.engine = nullptr;
    return 0;
}
