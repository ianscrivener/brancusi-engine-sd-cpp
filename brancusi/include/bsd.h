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
#define BSD_CAP_AUTO_FIT      (1ull << 8) /* bsd_load_options_t.auto_fit */
#define BSD_CAP_PART_MEMORY   (1ull << 9) /* Layer 2: bsd_parts() */
#define BSD_CAP_STAGE_TIMING  (1ull << 10) /* Layer 2: exact stage events (start_us, end_us) */

/* Values of bsd_load_options_t.auto_fit. */
typedef enum {
    BSD_AUTO_FIT_DEFAULT = 0, /* Upstream auto-fit: on when params_backend is not set. No engine plan. */
    BSD_AUTO_FIT_ON,          /* Engine plan: add `<module>=disk` entries until the parts fit. Upstream auto-fit off. */
    BSD_AUTO_FIT_OFF,         /* Upstream auto-fit off. No engine plan. */
} bsd_auto_fit_t;

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
    /* Added after the first release. */
    int32_t auto_fit;            /* bsd_auto_fit_t. */
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
    const char* preview_mode; /* "proj", "tae" or "vae". NULL or "none": no previews. */
    int32_t preview_interval; /* Preview every N steps. 0 or less: 2. */
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
    /* BSD_EVENT_STAGE: "load", "encode_prompt", "sample", "decode" */
    const char* stage;
    /* BSD_EVENT_METRICS: a finished stage and its duration. */
    const char* timing_name;
    float timing_ms;
    /* Added after the first release. BSD_EVENT_STAGE and BSD_EVENT_METRICS from Layer 2 hooks:
     * the start and the end of a stage on one monotonic clock, in microseconds. For a stage that
     * starts, stage_phase is BSD_STAGE_BEGIN and end_us is 0. For a stage that ends, it is
     * BSD_STAGE_END. A stage from the log text (Layer 1) has stage_phase 0 and no times. */
    int32_t stage_phase;
    int64_t start_us;
    int64_t end_us;
} bsd_event_t;

#define BSD_STAGE_BEGIN 1
#define BSD_STAGE_END   2

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
    /* Added after the first release. */
    int32_t physical_cores;
    int32_t n_threads;
    const char* default_sample_method; /* NULL when no model is loaded. */
    const char* default_scheduler;     /* NULL when no model is loaded. */
    /* Params backend of the loaded model: as given, or as auto-fit set it. NULL when not set. */
    const char* params_backend;
} bsd_info_t;

#define BSD_MAX_PARTS 16

/* Weight memory of one loaded part. */
typedef struct {
    char name[32];        /* "te", "diffusion", "vae", "taesd", "control_net" and others. */
    char backend[32];     /* Params backend: a device name, or "disk". */
    uint64_t bytes;       /* All weights of the part. */
    uint64_t bytes_gpu;   /* In device memory now (also a staged copy for the compute backend). */
    uint64_t bytes_cpu;   /* In host memory now. */
    uint64_t on_disk;     /* Not in memory now: loads from the file when needed. */
} bsd_part_memory_t;

typedef struct {
    uint32_t size;
    int32_t count;
    bsd_part_memory_t parts[BSD_MAX_PARTS];
} bsd_parts_t;

/* Name tables of the engine. See bsd_table_name(). */
typedef enum {
    BSD_TABLE_SAMPLE_METHOD = 0,
    BSD_TABLE_SCHEDULER,
    BSD_TABLE_WEIGHT_TYPE,
    BSD_TABLE_COUNT
} bsd_table_t;

/* What a model file holds, found from the tensor names. */
typedef enum {
    BSD_CONTENT_DIFFUSION = 0,
    BSD_CONTENT_TEXT_ENCODER,
    BSD_CONTENT_VAE,
    BSD_CONTENT_COUNT
} bsd_content_t;

#define BSD_MAX_PART_TYPES 8

typedef struct {
    char name[16]; /* Weight type name, for example "f16". */
    uint32_t tensors;
} bsd_type_count_t;

typedef struct {
    int32_t present;      /* 1 if the part is given (or found, for contents). */
    uint32_t tensors;
    uint64_t params;      /* Number of weights. 0 for contents. */
    uint64_t bytes;       /* Weight bytes in the file types. 0 for contents. */
    int32_t type_count;   /* Number of entries in types, most tensors first. */
    bsd_type_count_t types[BSD_MAX_PART_TYPES];
} bsd_part_info_t;

typedef struct {
    uint32_t size;
    char family[64];      /* Upstream version name, for example "SD 1.x". "" if unknown. */
    uint32_t tensors;
    uint64_t params;
    uint64_t mem_bytes;   /* Estimated weight memory with the load-time weight type. */
    bsd_part_info_t parts[BSD_COMPONENT_COUNT];   /* Indexed by bsd_component_t. */
    bsd_part_info_t contents[BSD_CONTENT_COUNT];  /* Indexed by bsd_content_t. */
} bsd_inspect_t;

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

/*
 * Write model parts into one GGUF file (or safetensors, by the output extension). Calls upstream
 * convert_with_components(). Only these parts are allowed: model, diffusion, clip_l, clip_g,
 * t5xxl and vae. Other parts return BSD_ERR_UNSUPPORTED. The other arguments and rules are the
 * same as for bsd_convert(). In `rules`, the tensor names have the part prefix, for example
 * "model.diffusion_model.", "text_encoders.clip_l.transformer." or "vae.". With convert_name,
 * the rules match the converted names, for example "first_stage_model." for the VAE.
 */
bsd_status_t bsd_convert_parts(bsd_engine_t* e, const bsd_components_t* parts, const char* output,
                               const char* type, const char* rules, int32_t convert_name);

/*
 * Layer 2. Read the file headers of the given parts, not the weights. Needs no loaded model.
 * `weight_type` is an optional load-time type for mem_bytes. NULL: keep the file types.
 * Returns BSD_ERR_MISSING_FILE or BSD_ERR_BAD_MODEL if a part cannot be read. An unknown
 * model family is not an error: family is "".
 */
bsd_status_t bsd_inspect(bsd_engine_t* e, const bsd_components_t* parts, const char* weight_type,
                         bsd_inspect_t* out);

/*
 * Write one model file as GGUF (or safetensors, by the output extension) with a new weight type.
 * Calls upstream convert(). `type` is a type name such as "q8_0". `rules` is an optional list of
 * "regex=type" pairs for single tensors (upstream tensor type rules). Needs no loaded model:
 * returns BSD_ERR_BUSY while a model is loaded. Reports BSD_EVENT_PROGRESS (step = tensors
 * written). It cannot be cancelled: the host stops the process.
 */
bsd_status_t bsd_convert(bsd_engine_t* e, const char* input, const char* output, const char* type,
                         const char* rules, int32_t convert_name);

/*
 * Layer 2. Load, replace or remove one model part on the loaded model, without a model reload.
 * `path` loads or replaces the part. NULL or "" removes it. Now only BSD_COMPONENT_VAE and
 * BSD_COMPONENT_TAESD work. Other parts return BSD_ERR_UNSUPPORTED: the text encoder and the
 * diffusion model share one upstream runner group, so a change rebuilds both. Returns
 * BSD_ERR_NOT_LOADED without a model and BSD_ERR_BUSY while a job runs. If the new part does not
 * fit the model, it returns BSD_ERR_BAD_MODEL and the old part stays. If upstream also cannot
 * rebuild the old part, it returns BSD_ERR_INTERNAL and unloads the model.
 */
bsd_status_t bsd_set_component(bsd_engine_t* e, bsd_component_t c, const char* path);

/*
 * Layer 2. Weight memory of each part of the loaded model. Returns BSD_ERR_NOT_LOADED without a
 * model and BSD_ERR_BUSY while a job runs (the job changes the part state).
 */
bsd_status_t bsd_parts(bsd_engine_t* e, bsd_parts_t* out);

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
const char* bsd_content_name(bsd_content_t c);
/* Number of entries in a name table. */
int32_t bsd_table_size(bsd_table_t table);
/* One name of a table. NULL if index is out of range. "" for an unused slot. */
const char* bsd_table_name(bsd_table_t table, int32_t index);
/*
 * Memory needed to load the given parts with this params_backend: the kept parts, plus the
 * largest `disk` part (a `disk` part loads and releases in turn), plus 1 GiB headroom.
 * A missing part file counts as 0 bytes.
 */
uint64_t bsd_load_bytes(const bsd_components_t* c, const char* params_backend);
/*
 * The auto-fit plan of bsd_load. Adds `te=disk`, then `diffusion=disk`, then `vae=disk` to
 * params_backend until bsd_load_bytes() is not more than the available memory. It does not change
 * a module that params_backend sets by name. `available` is the free memory of the main device.
 * `host_available` is added when the plan puts a module on `disk` or `cpu` (0 on unified memory).
 * Writes the plan to `out` (always NUL-terminated) and the bytes needed to *needed (if not NULL).
 * Returns 1 if the plan fits, 0 if it does not fit or `out` is too small.
 */
int32_t bsd_fit_params_backend(const bsd_components_t* c, const char* params_backend, uint64_t available,
                               uint64_t host_available, char* out, size_t out_size, uint64_t* needed);
/* Map recent upstream log text to a status. Used when upstream gives no other signal. */
bsd_status_t bsd_classify_failure(const char* log_text);
/*
 * Layer 1 stage detection from one upstream log line. Sets *stage when a stage starts.
 * Sets *timing_name and *timing_ms when a stage ends. Returns 1 if the line matched.
 * Layer 2 replaces this for encode_prompt, sample and decode with hooks that give exact times
 * (see the Layer 2 register). This function stays as the fallback and for the load stage.
 */
int bsd_detect_stage(const char* line, const char** stage, const char** timing_name, float* timing_ms);

#ifdef __cplusplus
}
#endif

#endif /* BSD_H */
