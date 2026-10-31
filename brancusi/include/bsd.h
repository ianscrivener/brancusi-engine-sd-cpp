/*
 * bsd.h - Brancusi SD C API.
 *
 * The contract between a host (brancusi-engine, a test harness, a binding) and the
 * Brancusi wrapper on stable-diffusion.cpp. See brancusi-desktop _docs/sd-cpp-interface.md.
 *
 * Rules:
 *  1. Plain C ABI. Opaque handles. Integer types with a set size.
 *  2. Each struct has `size` as its first member. Set it to sizeof(the struct).
 *     Fields are added only at the end of a struct.
 *  3. No upstream structs in this header.
 *  4. Each call returns bsd_status_t. bsd_last_error() returns the message.
 *  5. bsd_info() reports capability flags. A missing feature is a flag set to 0.
 *  6. One event callback receives every event type.
 *  7. bsd_cancel() is safe from any thread while bsd_generate_image() runs.
 */
#ifndef BSD_H
#define BSD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BSD_API_VERSION 1

typedef struct bsd_engine bsd_engine_t;

typedef enum {
    BSD_OK = 0,
    BSD_ERR_OUT_OF_MEMORY,
    BSD_ERR_MISSING_FILE,
    BSD_ERR_BAD_MODEL,
    BSD_ERR_CANCELLED,
    BSD_ERR_UNSUPPORTED,
    BSD_ERR_INVALID_ARGUMENT,
    BSD_ERR_INTERNAL,
    BSD_ERR_NOT_LOADED,
    BSD_ERR_BUSY,
    BSD_STATUS_COUNT
} bsd_status_t;

/* Model parts. These match the path fields in upstream sd_ctx_params_t. */
typedef enum {
    BSD_COMPONENT_MODEL = 0,
    BSD_COMPONENT_DIFFUSION,
    BSD_COMPONENT_HIGH_NOISE_DIFFUSION,
    BSD_COMPONENT_UNCOND_DIFFUSION,
    BSD_COMPONENT_VAE,
    BSD_COMPONENT_TAESD,
    BSD_COMPONENT_CLIP_L,
    BSD_COMPONENT_CLIP_G,
    BSD_COMPONENT_CLIP_VISION,
    BSD_COMPONENT_T5XXL,
    BSD_COMPONENT_LLM,
    BSD_COMPONENT_LLM_VISION,
    BSD_COMPONENT_CONTROL_NET,
    BSD_COMPONENT_IP_ADAPTER,
    BSD_COMPONENT_PHOTO_MAKER,
    BSD_COMPONENT_PULID,
    BSD_COMPONENT_MOTION_MODULE,
    BSD_COMPONENT_AUDIO_VAE,
    BSD_COMPONENT_AUDIO_ENCODER,
    BSD_COMPONENT_EMBEDDINGS_CONNECTORS,
    BSD_COMPONENT_TOKENIZER,
    BSD_COMPONENT_COUNT
} bsd_component_t;

typedef enum {
    BSD_CANCEL_ALL = 0,     /* Stop the current generation as soon as possible. */
    BSD_CANCEL_NEW_LATENTS, /* Finish the current image, skip the rest of the batch. */
} bsd_cancel_mode_t;

typedef enum {
    BSD_EVENT_LOG = 0,
    BSD_EVENT_PROGRESS,
    BSD_EVENT_PREVIEW,
    BSD_EVENT_STAGE,
    BSD_EVENT_METRICS,
} bsd_event_type_t;

typedef enum {
    BSD_LOG_DEBUG = 0,
    BSD_LOG_VERBOSE,
    BSD_LOG_INFO,
    BSD_LOG_WARN,
    BSD_LOG_ERROR,
} bsd_log_level_t;

/* Capability flags in bsd_info_t.capabilities. */
#define BSD_CAP_IMAGE         (1ull << 0)
#define BSD_CAP_VIDEO         (1ull << 1)
#define BSD_CAP_CANCEL        (1ull << 2)
#define BSD_CAP_PREVIEW       (1ull << 3)
#define BSD_CAP_MEMORY        (1ull << 4)
#define BSD_CAP_UPSCALE       (1ull << 5)
#define BSD_CAP_SET_COMPONENT (1ull << 6) /* Layer 2 */
#define BSD_CAP_RUN_STAGE     (1ull << 7) /* Layer 2 */

typedef struct {
    uint32_t size;
    int32_t n_threads; /* 0 or less: number of physical cores. */
} bsd_config_t;

typedef struct {
    uint32_t size;
    /* Indexed by bsd_component_t. NULL or "" means not set. */
    const char* paths[BSD_COMPONENT_COUNT];
} bsd_components_t;

typedef struct {
    uint32_t size;
    const char* backend;         /* For example "MTL0" or "cuda0". NULL: auto. */
    const char* params_backend;  /* For example "te=cpu,vae=MTL0". NULL: same as backend. */
    const char* max_vram;        /* Per-device GiB budget. NULL: live free memory. */
    const char* flash_attn_mode; /* "on", "off", "auto" or module assignments. NULL: auto. */
    const char* model_args;      /* Upstream --model-args. NULL: none. */
    const char* weight_type;     /* Load-time quantization, for example "q8_0". NULL: keep file types. */
    int32_t enable_mmap;         /* 0 or 1. */
    int32_t eager_load;          /* 0 or 1. Load all weights now, not on first use. */
    int32_t tae_preview_only;    /* 0 or 1. Use TAESD for previews only. */
    int32_t skip_memory_check;   /* 0 or 1. Do not compare file sizes with free memory before the load. */
} bsd_load_options_t;

typedef struct {
    uint32_t size;
    const char* prompt;
    const char* negative_prompt; /* NULL: empty. */
    int32_t width;
    int32_t height;
    int32_t steps;            /* 0 or less: model default. */
    int64_t seed;             /* Less than 0: random. */
    float cfg_scale;          /* 0 or less: model default. */
    const char* sample_method;/* For example "euler". NULL: model default. */
    const char* scheduler;    /* For example "simple". NULL: model default. */
    const char* sigmas;       /* Comma-separated list or a preset name. NULL: none. */
    int32_t batch_count;      /* 0 or less: 1. */
    float flow_shift;         /* 0: model default. */
} bsd_image_params_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t* data;
} bsd_image_t;

typedef struct {
    int32_t count;
    bsd_image_t* images;
    int64_t* seeds;
} bsd_result_t;

typedef struct {
    uint32_t size;
    bsd_event_type_t type;
    /* BSD_EVENT_LOG */
    bsd_log_level_t level;
    const char* text;
    /* BSD_EVENT_PROGRESS and BSD_EVENT_PREVIEW */
    int32_t step;
    int32_t steps;
    float sec_per_step;
    /* BSD_EVENT_PREVIEW */
    const bsd_image_t* preview;
    int32_t is_noisy;
    /* BSD_EVENT_STAGE: "load", "generate", ... */
    const char* stage;
} bsd_event_t;

typedef void (*bsd_event_cb_t)(const bsd_event_t* event, void* user);

#define BSD_MAX_DEVICES 16

typedef struct {
    char name[64];
    char description[128];
    char type[16]; /* "cpu", "gpu", "igpu", "accel" */
    uint64_t free_bytes;
    uint64_t total_bytes;
} bsd_device_memory_t;

typedef struct {
    uint32_t size;
    int32_t device_count;
    bsd_device_memory_t devices[BSD_MAX_DEVICES];
} bsd_memory_t;

typedef struct {
    uint32_t size;
    int32_t api_version;
    const char* sd_version;
    const char* sd_commit;
    const char* system_info;
    uint64_t capabilities;
    int32_t loaded;
    const char* model_version; /* NULL when no model is loaded. */
} bsd_info_t;

int bsd_api_version(void);
const char* bsd_status_name(bsd_status_t status);

bsd_engine_t* bsd_create(const bsd_config_t* config);
void bsd_destroy(bsd_engine_t* engine);
const char* bsd_last_error(const bsd_engine_t* engine);
/* Extra detail for the last error, for example the last log lines. Can be "". */
const char* bsd_last_error_detail(const bsd_engine_t* engine);

/* Models */
bsd_status_t bsd_load(bsd_engine_t* e, const bsd_components_t* c, const bsd_load_options_t* o);
bsd_status_t bsd_unload(bsd_engine_t* e);

/* Work */
bsd_status_t bsd_generate_image(bsd_engine_t* e, const bsd_image_params_t* p, bsd_result_t** out);
void bsd_cancel(bsd_engine_t* e, bsd_cancel_mode_t mode);
void bsd_free_result(bsd_result_t* r);

/* Information and metrics */
bsd_status_t bsd_info(bsd_engine_t* e, bsd_info_t* out);
bsd_status_t bsd_memory(bsd_engine_t* e, bsd_memory_t* out);
void bsd_set_event_cb(bsd_engine_t* e, bsd_event_cb_t cb, void* user);

/* Helpers, exposed for tests. */
const char* bsd_component_name(bsd_component_t c);
/* Map recent upstream log text to a status. Used when upstream gives no other signal. */
bsd_status_t bsd_classify_failure(const char* log_text);

#ifdef __cplusplus
}
#endif

#endif /* BSD_H */
