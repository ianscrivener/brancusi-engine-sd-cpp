// Unit tests for bsd_* functions that need no model.
// Set BSD_TEST_SD15 to a single-file SD 1.5 model to also test inspect on a real file.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "bsd.h"
#include "bsd_hook.h"

namespace fs = std::filesystem;

static int failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                \
        }                                                              \
    } while (0)

static void test_classify_failure() {
    CHECK(bsd_classify_failure("ggml_metal: failed to allocate buffer of size 9000 MB") == BSD_ERR_OUT_OF_MEMORY);
    CHECK(bsd_classify_failure("CUDA error: out of memory") == BSD_ERR_OUT_OF_MEMORY);
    CHECK(bsd_classify_failure("vk::Device::allocateMemory: ErrorOutOfDeviceMemory") == BSD_ERR_OUT_OF_MEMORY);
    CHECK(bsd_classify_failure("[W] file /m/x.gguf not found") == BSD_ERR_MISSING_FILE);
    CHECK(bsd_classify_failure("cannot inspect model source") == BSD_ERR_MISSING_FILE);
    CHECK(bsd_classify_failure("get sd version from file failed: '/m/x.safetensors'") == BSD_ERR_BAD_MODEL);
    CHECK(bsd_classify_failure("[W] unknown format /m/x.bin") == BSD_ERR_BAD_MODEL);
    CHECK(bsd_classify_failure("something else went wrong") == BSD_ERR_INTERNAL);
    CHECK(bsd_classify_failure(nullptr) == BSD_ERR_INTERNAL);
}

static void test_detect_stage() {
    const char* stage  = nullptr;
    const char* timing = nullptr;
    float ms           = 0;
    CHECK(bsd_detect_stage("generate_image 512x512 --- image.cpp:869", &stage, &timing, &ms) == 1);
    CHECK(stage != nullptr && std::strcmp(stage, "encode_prompt") == 0);
    CHECK(bsd_detect_stage("get_learned_condition completed, taking 27.31s --- image.cpp:580", &stage, &timing, &ms) == 1);
    CHECK(stage == nullptr && timing != nullptr && std::strcmp(timing, "encode_prompt") == 0);
    CHECK(ms > 27309 && ms < 27311);
    CHECK(bsd_detect_stage("generating image: 1/1 - seed 7", &stage, &timing, &ms) == 1);
    CHECK(std::strcmp(stage, "sample") == 0);
    CHECK(bsd_detect_stage("sampling completed, taking 25.68s", &stage, &timing, &ms) == 1);
    CHECK(std::strcmp(timing, "sample") == 0);
    CHECK(bsd_detect_stage("decoding 1 latents", &stage, &timing, &ms) == 1);
    CHECK(std::strcmp(stage, "decode") == 0);
    CHECK(bsd_detect_stage("decode_first_stage completed, taking 3.74s", &stage, &timing, &ms) == 1);
    CHECK(std::strcmp(timing, "decode") == 0);
    CHECK(bsd_detect_stage("loading tensors completed, taking 1.02s (read: 0.82s)", &stage, &timing, &ms) == 1);
    CHECK(std::strcmp(timing, "load") == 0 && ms > 1019 && ms < 1021);
    CHECK(bsd_detect_stage("sampling using Euler method", &stage, &timing, &ms) == 0);
    CHECK(bsd_detect_stage("generate_image completed in 56.74s", &stage, &timing, &ms) == 1);
    CHECK(stage == nullptr && timing == nullptr);
}

static void test_names() {
    CHECK(std::strcmp(bsd_status_name(BSD_ERR_OUT_OF_MEMORY), "out_of_memory") == 0);
    CHECK(std::strcmp(bsd_status_name(BSD_ERR_BUSY), "busy") == 0);
    CHECK(std::strcmp(bsd_component_name(BSD_COMPONENT_LLM), "llm") == 0);
    CHECK(std::strcmp(bsd_component_name(BSD_COMPONENT_TOKENIZER), "tokenizer") == 0);
    CHECK(std::strcmp(bsd_content_name(BSD_CONTENT_TEXT_ENCODER), "text_encoder") == 0);
}

static bool table_has(bsd_table_t table, const char* name) {
    for (int32_t i = 0; i < bsd_table_size(table); i++) {
        const char* n = bsd_table_name(table, i);
        if (n != nullptr && std::strcmp(n, name) == 0) return true;
    }
    return false;
}

static void test_tables() {
    CHECK(bsd_table_size(BSD_TABLE_SAMPLE_METHOD) > 0);
    CHECK(table_has(BSD_TABLE_SAMPLE_METHOD, "euler"));
    CHECK(table_has(BSD_TABLE_SCHEDULER, "karras"));
    CHECK(table_has(BSD_TABLE_WEIGHT_TYPE, "q8_0"));
    CHECK(table_has(BSD_TABLE_WEIGHT_TYPE, "q4_K"));
    CHECK(!table_has(BSD_TABLE_WEIGHT_TYPE, "DEPRECATED"));
    CHECK(bsd_table_name(BSD_TABLE_SCHEDULER, -1) == nullptr);
    CHECK(bsd_table_name(BSD_TABLE_SCHEDULER, bsd_table_size(BSD_TABLE_SCHEDULER)) == nullptr);
}

struct SeenEvent {
    bsd_event_type_t type;
    std::string name;
    int32_t phase;
    int64_t start_us, end_us;
    float timing_ms;
};

static void collect_event(const bsd_event_t* ev, void* user) {
    if (ev->type != BSD_EVENT_STAGE && ev->type != BSD_EVENT_METRICS) return;
    const char* name = ev->type == BSD_EVENT_STAGE ? ev->stage : ev->timing_name;
    static_cast<std::vector<SeenEvent>*>(user)->push_back(
        {ev->type, name != nullptr ? name : "", ev->stage_phase, ev->start_us, ev->end_us, ev->timing_ms});
}

static void test_stage_hooks(bsd_engine_t* e) {
    std::vector<SeenEvent> seen;
    bsd_set_event_cb(e, collect_event, &seen);
    {
        bsd_stage_scope scope("sample");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        scope.end();
        scope.end();  // A second end does nothing.
    }
    {
        bsd_stage_scope scope("decode");  // The destructor closes the stage.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    bsd_set_event_cb(e, nullptr, nullptr);
    // sample: begin, end, metrics. decode: begin, end, metrics.
    CHECK(seen.size() == 6);
    if (seen.size() == 6) {
        CHECK(seen[0].type == BSD_EVENT_STAGE && seen[0].name == "sample" && seen[0].phase == BSD_STAGE_BEGIN);
        CHECK(seen[0].start_us > 0 && seen[0].end_us == 0);
        CHECK(seen[1].type == BSD_EVENT_STAGE && seen[1].phase == BSD_STAGE_END);
        CHECK(seen[1].start_us == seen[0].start_us && seen[1].end_us - seen[1].start_us >= 20000);
        CHECK(seen[2].type == BSD_EVENT_METRICS && seen[2].name == "sample" && seen[2].phase == BSD_STAGE_END);
        CHECK(seen[2].timing_ms >= 20.0f && seen[2].timing_ms < 200.0f);
        CHECK(static_cast<int64_t>(seen[2].timing_ms * 1000.0f) <= (seen[2].end_us - seen[2].start_us) + 1000);
        CHECK(seen[3].name == "decode" && seen[3].start_us >= seen[1].end_us);
        CHECK(seen[4].phase == BSD_STAGE_END && seen[5].name == "decode");
    }
    {
        // An unknown stage name gives no event.
        std::vector<SeenEvent> none;
        bsd_set_event_cb(e, collect_event, &none);
        bsd_stage_scope scope("load");
        scope.end();
        bsd_set_event_cb(e, nullptr, nullptr);
        CHECK(none.empty());
    }
}

static void test_load_errors() {
    bsd_config_t config{};
    config.size     = sizeof(config);
    bsd_engine_t* e = bsd_create(&config);
    CHECK(e != nullptr);
    CHECK(bsd_create(&config) == nullptr);  // One engine per process.

    bsd_components_t c{};
    c.size = sizeof(c);
    CHECK(bsd_load(e, &c, nullptr) == BSD_ERR_INVALID_ARGUMENT);

    c.paths[BSD_COMPONENT_DIFFUSION] = "/no/such/model.gguf";
    CHECK(bsd_load(e, &c, nullptr) == BSD_ERR_MISSING_FILE);
    CHECK(std::strstr(bsd_last_error(e), "diffusion") != nullptr);

    bsd_image_params_t p{};
    p.size   = sizeof(p);
    p.prompt = "x";
    p.width = p.height = 64;
    bsd_result_t* r    = nullptr;
    CHECK(bsd_generate_image(e, &p, &r) == BSD_ERR_NOT_LOADED);

    bsd_memory_t mem{};
    CHECK(bsd_memory(e, &mem) == BSD_OK);
    CHECK(mem.device_count > 0);

    CHECK(bsd_convert(e, nullptr, "/tmp/o.gguf", "q8_0", nullptr, 0) == BSD_ERR_INVALID_ARGUMENT);
    CHECK(bsd_convert(e, "/m/x.safetensors", "/tmp/o.gguf", "nope", nullptr, 0) == BSD_ERR_INVALID_ARGUMENT);
    CHECK(bsd_convert(e, "/does/not/exist.safetensors", "/tmp/o.gguf", "q8_0", nullptr, 0) == BSD_ERR_MISSING_FILE);

    bsd_components_t parts{};
    parts.size                      = sizeof(parts);
    CHECK(bsd_convert_parts(e, &parts, "/tmp/o.gguf", "q8_0", nullptr, 0) == BSD_ERR_INVALID_ARGUMENT);
    parts.paths[BSD_COMPONENT_LLM] = "/m/llm.safetensors";
    CHECK(bsd_convert_parts(e, &parts, "/tmp/o.gguf", "q8_0", nullptr, 0) == BSD_ERR_UNSUPPORTED);
    parts.paths[BSD_COMPONENT_LLM]  = nullptr;
    parts.paths[BSD_COMPONENT_VAE]  = "/does/not/exist.safetensors";
    CHECK(bsd_convert_parts(e, &parts, "/tmp/o.gguf", "q8_0", nullptr, 0) == BSD_ERR_MISSING_FILE);

    bsd_inspect_t ins{};
    CHECK(bsd_inspect(e, &c, nullptr, &ins) == BSD_ERR_MISSING_FILE);
    bsd_components_t none{};
    none.size = sizeof(none);
    CHECK(bsd_inspect(e, &none, nullptr, &ins) == BSD_ERR_INVALID_ARGUMENT);

    if (const char* sd15 = std::getenv("BSD_TEST_SD15")) {
        bsd_components_t one{};
        one.size                       = sizeof(one);
        one.paths[BSD_COMPONENT_MODEL] = sd15;
        CHECK(bsd_inspect(e, &one, "q8_0", &ins) == BSD_OK);
        CHECK(std::strcmp(ins.family, "SD 1.x") == 0);
        CHECK(ins.parts[BSD_COMPONENT_MODEL].present == 1);
        CHECK(ins.parts[BSD_COMPONENT_MODEL].tensors > 1000);
        CHECK(ins.params > 1000000000ull);
        CHECK(ins.contents[BSD_CONTENT_DIFFUSION].present == 1);
        CHECK(ins.contents[BSD_CONTENT_TEXT_ENCODER].present == 1);
        CHECK(ins.contents[BSD_CONTENT_VAE].present == 1);
        CHECK(ins.mem_bytes > 0);
        CHECK(bsd_inspect(e, &one, "nope", &ins) == BSD_ERR_INVALID_ARGUMENT);
        std::printf("bsd-tests: inspect on %s passed\n", sd15);
    }

    // Layer 2: parts and set_component need a loaded model.
    CHECK(bsd_set_component(e, BSD_COMPONENT_VAE, "/m/vae.safetensors") == BSD_ERR_NOT_LOADED);
    CHECK(bsd_set_component(e, BSD_COMPONENT_VAE, nullptr) == BSD_ERR_NOT_LOADED);
    CHECK(bsd_set_component(nullptr, BSD_COMPONENT_VAE, nullptr) == BSD_ERR_INVALID_ARGUMENT);
    bsd_parts_t parts_out{};
    CHECK(bsd_parts(e, &parts_out) == BSD_ERR_NOT_LOADED);
    CHECK(bsd_parts(e, nullptr) == BSD_ERR_INVALID_ARGUMENT);
    test_stage_hooks(e);

    bsd_info_t info{};
    CHECK(bsd_info(e, &info) == BSD_OK);
    CHECK(info.api_version == BSD_API_VERSION);
    CHECK(info.loaded == 0);
    CHECK(info.physical_cores > 0);
    CHECK(info.n_threads > 0);
    CHECK(info.default_sample_method == nullptr);
    CHECK((info.capabilities & BSD_CAP_SET_COMPONENT) != 0);
    CHECK((info.capabilities & BSD_CAP_PART_MEMORY) != 0);
    CHECK((info.capabilities & BSD_CAP_STAGE_TIMING) != 0);

    bsd_destroy(e);
}

static constexpr uint64_t kMiB = 1024ull * 1024ull;
static constexpr uint64_t kGiB = 1024ull * kMiB;

// Sparse files of a set size. The plan reads only the file sizes.
static std::string fake_file(const fs::path& dir, const char* name, uint64_t bytes) {
    const fs::path path = dir / name;
    std::FILE* f        = std::fopen(path.string().c_str(), "wb");
    if (f != nullptr) std::fclose(f);
    std::error_code ec;
    fs::resize_file(path, bytes, ec);
    return path.string();
}

static void test_auto_fit() {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "bsd-tests-auto-fit";
    fs::create_directories(dir, ec);
    const std::string te = fake_file(dir, "te.gguf", 300 * kMiB);
    const std::string dit = fake_file(dir, "dit.gguf", 700 * kMiB);
    const std::string vae = fake_file(dir, "vae.safetensors", 30 * kMiB);

    bsd_components_t c{};
    c.size                           = sizeof(c);
    c.paths[BSD_COMPONENT_LLM]       = te.c_str();
    c.paths[BSD_COMPONENT_DIFFUSION] = dit.c_str();
    c.paths[BSD_COMPONENT_VAE]       = vae.c_str();
    const uint64_t head              = kGiB;

    // Kept parts, plus the largest disk part, plus 1 GiB.
    CHECK(bsd_load_bytes(&c, nullptr) == head + 1030 * kMiB);
    CHECK(bsd_load_bytes(&c, "te=disk") == head + 1030 * kMiB);
    CHECK(bsd_load_bytes(&c, "te=disk,diffusion=disk") == head + 730 * kMiB);
    CHECK(bsd_load_bytes(&c, "disk") == head + 700 * kMiB);
    CHECK(bsd_load_bytes(&c, "all=disk") == head + 700 * kMiB);
    // A module entry overrides the default in any order, as in upstream.
    CHECK(bsd_load_bytes(&c, "diffusion=MTL0,disk") == head + 1000 * kMiB);
    CHECK(bsd_load_bytes(&c, "disk,diffusion=MTL0") == head + 1000 * kMiB);
    CHECK(bsd_load_bytes(&c, "text_encoder=disk") == head + 1030 * kMiB);
    CHECK(bsd_load_bytes(&c, "te=disk,DiT=disk") == head + 730 * kMiB);

    char plan[128];
    uint64_t needed = 0;
    CHECK(bsd_fit_params_backend(&c, nullptr, head + 2 * kGiB, 0, plan, sizeof(plan), &needed) == 1);
    CHECK(std::strcmp(plan, "") == 0 && needed == head + 1030 * kMiB);
    CHECK(bsd_fit_params_backend(&c, nullptr, head + 800 * kMiB, 0, plan, sizeof(plan), &needed) == 1);
    CHECK(std::strcmp(plan, "te=disk,diffusion=disk") == 0 && needed == head + 730 * kMiB);
    CHECK(bsd_fit_params_backend(&c, nullptr, head + 710 * kMiB, 0, plan, sizeof(plan), &needed) == 1);
    CHECK(std::strcmp(plan, "te=disk,diffusion=disk,vae=disk") == 0 && needed == head + 700 * kMiB);
    CHECK(bsd_fit_params_backend(&c, nullptr, head + 100 * kMiB, 0, plan, sizeof(plan), &needed) == 0);
    CHECK(std::strcmp(plan, "te=disk,diffusion=disk,vae=disk") == 0 && needed == head + 700 * kMiB);
    // A module that the caller sets by name stays as it is.
    CHECK(bsd_fit_params_backend(&c, "diffusion=MTL0", head + 800 * kMiB, 0, plan, sizeof(plan), &needed) == 0);
    CHECK(std::strcmp(plan, "diffusion=MTL0,te=disk,vae=disk") == 0 && needed == head + 1000 * kMiB);
    // Host memory counts when a module goes to disk (not on unified memory: the caller gives 0).
    CHECK(bsd_fit_params_backend(&c, nullptr, head + 500 * kMiB, 300 * kMiB, plan, sizeof(plan), &needed) == 1);
    CHECK(std::strcmp(plan, "te=disk,diffusion=disk") == 0);
    CHECK(bsd_fit_params_backend(&c, nullptr, head, 0, plan, 8, &needed) == 0);
    CHECK(plan[0] == '\0');

    // No VAE part: the plan does not add vae=disk.
    c.paths[BSD_COMPONENT_VAE] = nullptr;
    CHECK(bsd_fit_params_backend(&c, nullptr, head, 0, plan, sizeof(plan), &needed) == 0);
    CHECK(std::strcmp(plan, "te=disk,diffusion=disk") == 0);
    c.paths[BSD_COMPONENT_VAE] = vae.c_str();

    bsd_config_t config{};
    config.size     = sizeof(config);
    bsd_engine_t* e = bsd_create(&config);
    CHECK(e != nullptr);
    bsd_info_t info{};
    CHECK(bsd_info(e, &info) == BSD_OK);
    CHECK((info.capabilities & BSD_CAP_AUTO_FIT) != 0);
    CHECK(info.params_backend == nullptr);

    bsd_load_options_t o{};
    o.size     = sizeof(o);
    o.auto_fit = 7;
    CHECK(bsd_load(e, &c, &o) == BSD_ERR_INVALID_ARGUMENT);

    bsd_memory_t mem{};
    bsd_memory(e, &mem);
    bool gpu = false;
    for (int i = 0; i < mem.device_count; i++) gpu = gpu || std::strcmp(mem.devices[i].type, "cpu") != 0;
#if defined(__APPLE__) && defined(__aarch64__)
    // Unified memory: host memory adds 0. A 0.1 GiB budget is less than the headroom, so no plan
    // fits, and the load stops before upstream.
    if (gpu) {
        o.auto_fit = BSD_AUTO_FIT_ON;
        o.max_vram = "0.1";
        CHECK(bsd_load(e, &c, &o) == BSD_ERR_OUT_OF_MEMORY);
        CHECK(std::strstr(bsd_last_error(e), "te=disk,diffusion=disk,vae=disk") != nullptr);
        o.auto_fit = BSD_AUTO_FIT_DEFAULT;
        CHECK(bsd_load(e, &c, &o) == BSD_ERR_OUT_OF_MEMORY);
        CHECK(std::strstr(bsd_last_error(e), "te=disk") == nullptr);
    }
#else
    (void)gpu;
#endif
    bsd_destroy(e);
    fs::remove_all(dir, ec);
}

int main() {
    test_classify_failure();
    test_detect_stage();
    test_names();
    test_tables();
    test_load_errors();
    test_auto_fit();
    if (failures == 0) std::printf("bsd-tests: all passed\n");
    return failures == 0 ? 0 : 1;
}
