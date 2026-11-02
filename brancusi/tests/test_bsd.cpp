// Unit tests for bsd_* functions that need no model.
// Set BSD_TEST_SD15 to a single-file SD 1.5 model to also test inspect on a real file.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bsd.h"

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

    bsd_info_t info{};
    CHECK(bsd_info(e, &info) == BSD_OK);
    CHECK(info.api_version == BSD_API_VERSION);
    CHECK(info.loaded == 0);
    CHECK(info.physical_cores > 0);
    CHECK(info.n_threads > 0);
    CHECK(info.default_sample_method == nullptr);

    bsd_destroy(e);
}

int main() {
    test_classify_failure();
    test_detect_stage();
    test_names();
    test_tables();
    test_load_errors();
    if (failures == 0) std::printf("bsd-tests: all passed\n");
    return failures == 0 ? 0 : 1;
}
