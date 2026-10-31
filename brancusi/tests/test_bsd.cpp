// Unit tests for bsd_* functions that need no model.
#include <cstdio>
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

    bsd_info_t info{};
    CHECK(bsd_info(e, &info) == BSD_OK);
    CHECK(info.api_version == BSD_API_VERSION);
    CHECK(info.loaded == 0);

    bsd_destroy(e);
}

int main() {
    test_classify_failure();
    test_detect_stage();
    test_names();
    test_load_errors();
    if (failures == 0) std::printf("bsd-tests: all passed\n");
    return failures == 0 ? 0 : 1;
}
