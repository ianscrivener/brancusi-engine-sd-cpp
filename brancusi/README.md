# brancusi/

Brancusi code in the stable-diffusion.cpp fork. Upstream code stays unchanged where possible. All our code is in this folder.

Pinned upstream release: `master-1028-37bc88b`.

## Contents

| Path | Contents |
|---|---|
| `include/bsd.h` | The `bsd_*` C API. Plain C ABI, versioned with `BSD_API_VERSION`. |
| `src/l1_*.cpp` | Layer 1. Uses only `include/stable-diffusion.h` and the public ggml backend API. |
| `src/l2.h`, `src/l2_*.cpp` | Layer 2. Uses upstream internals. Built as the `bsd_l2` object library with the upstream include folders and definitions. Now: `l2_inspect.cpp` (`ModelLoader`). |
| `host/main.cpp` | `brancusi-engine`. JSON lines on stdin and stdout. |
| `tests/test_bsd.cpp` | Unit tests. They need no model. |
| `scripts/package.sh` | Makes the zip for one variant. Prints its size and SHA-256. |

## Upstream hooks

Each change to an upstream file has a `BRANCUSI:` comment.

| File | Hook |
|---|---|
| `CMakeLists.txt` | `add_subdirectory(brancusi)` behind the `SD_BUILD_BRANCUSI` option (default `ON`). |

## Build (macOS, Metal)

```sh
cmake -B build -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target brancusi-engine bsd-tests
./build/bin/bsd-tests
```

For other backends, use the upstream CMake options (`SD_CUDA`, `SD_VULKAN`, `SD_HIPBLAS` and others).

## CI and releases

The workflow `.github/workflows/brancusi-engine.yml` builds `brancusi-engine` for each variant in its matrix. It runs `bsd-tests` on the Metal and CPU builds. Now the matrix has only `darwin-arm64-metal`. To make CI faster, the other variants have comment marks.

To make a release, push a tag `brancusi-v<version>`, for example `brancusi-v0.1.0`. The release gets one zip for each variant and a `manifest.json` with the file, size and SHA-256 of each zip. Copy these values into `engine.json` in `brancusi-desktop`.

The upstream `.gitignore` ignores `tests/`. To add a new file in `brancusi/tests/`, use `git add -f`.

## Protocol (API version 1)

One JSON object on each line. stdout carries only protocol messages. The engine sends all other output, including upstream `printf` output, to stderr. The engine exits when stdin closes.

Commands (core to engine):

| `cmd` | Fields | Reply |
|---|---|---|
| `info` | – | `info` |
| `memory` | – | `memory` with `devices` |
| `load` | `components` (part name to path), `options` | `stage`, `log`, then `loaded` or `error` |
| `unload` | – | `unloaded` or `error` |
| `generate` | `job`, `params`, `output_dir`, `preview` | `stage`, `progress`, `preview`, `metrics`, `log`, then `done` or `error` |
| `cancel` | `job`, `mode` (`all` or `new_latents`) | `ok`. The running job then ends with `error` code `cancelled`. |
| `convert` | `input` or `components`, `output`, `type`, `rules` (optional), `convert_name` (optional) | `stage` (`convert`), `progress` (step = tensors written), `log`, then `ok` (`output`, `bytes`, `ms`) or `error` |
| `inspect` | `components` (part name to path), `weight_type` (optional) | `log`, then `inspect` or `error` |
| `hardware` | – | `hardware` |
| `schedulers` | – | `schedulers` |
| `log_level` | `level` (`debug`, `verbose`, `info`, `warn`, `error`) | `ok` |
| `shutdown` | – | The engine stops the running job and exits. |

`convert` writes one file with a new weight type (upstream `convert_with_components()`). Give one `input` file, or `components` to merge parts into one file. `components` can have only `model`, `diffusion`, `clip_l`, `clip_g`, `t5xxl` and `vae`. Other parts (for example `llm`) give `unsupported`. `type` is a type name such as `q8_0`. `rules` is an optional list of `regex=type` pairs for single tensors. The rules match the tensor names with the part prefix (`model.diffusion_model.`, `text_encoders.clip_l.transformer.`, `vae.`). With `convert_name`, they match the converted names (for example `first_stage_model.` for the VAE). A conversion needs no loaded model: with a model loaded, the engine replies `busy`. A conversion cannot be cancelled: stop the engine process.

`inspect` reads the file headers only, not the weights (Layer 2, upstream `ModelLoader`). It needs no loaded model. It runs on the worker thread, so it replies `busy` while a job runs. Run it in a second, short-lived engine process. The reply:

```json
{"event":"inspect","family":"SD 1.x","known":true,"tensors":1145,"params":1066247386,"mem_bytes":4265136104,"mem_mib":4067,"ms":13,
 "parts":{"model":{"tensors":1145,"params":1066247386,"bytes":4264989544,"type":"f32","types":{"f32":1143,"i32":2}}},
 "contents":{"diffusion":{"tensors":686,"type":"f32","types":{"f32":686}},"text_encoder":{"tensors":197,"type":"f32","types":{"f32":196,"i32":1}},"vae":{"tensors":248,"type":"f32","types":{"f32":248}}}}
```

- `family` is the upstream version name. It is `""` and `known` is `false` when upstream cannot find the family (for example a split SD 1.5 UNet).
- `parts` has one entry for each given part. `type` is the type of most tensors.
- `contents` tells what the files hold, from the tensor names: `diffusion`, `text_encoder` and `vae`. Use it to find missing parts.
- `mem_bytes` is the upstream estimate of the weight memory, with `weight_type` if given.

`hardware` returns `system_info`, `features` (the `system_info` flags, for example `{"NEON":true}`), `physical_cores`, `n_threads` and `devices`. `schedulers` returns `samplers`, `schedulers` and `types` (weight type names) from the engine tables. Both commands run at once, also while a job runs.

Part names are the `bsd_component_t` names: `model`, `diffusion`, `vae`, `taesd`, `llm`, `t5xxl`, `clip_l` and others.

`load` options: `backend`, `params_backend`, `max_vram`, `flash_attn`, `model_args`, `weight_type`, `mmap`, `eager_load`, `tae_preview_only`, `skip_memory_check`.

`generate` params: `prompt`, `negative_prompt`, `width`, `height`, `steps`, `seed`, `cfg_scale`, `sample_method`, `scheduler`, `sigmas` (a list, or the preset `qwen-image-2.1-turbo`), `batch_count`, `flow_shift`.

`generate` preview: `mode` (`none`, `proj`, `tae` or `vae`. Default `none`), `interval` (steps, default 2), `max_px` (default 256). The engine sends at most 4 previews each second, as base64 JPEG.

Events during a job:

- `stage`: `load`, `encode_prompt`, `sample` or `decode`. Layer 1 finds the stage in the upstream log text (`bsd_detect_stage`).
- `progress`: `step`, `steps`, `sec_per_step`. Upstream also sends tensor loading as progress. Use it as a sampling step only in the `sample` stage.
- `metrics`: device memory (`device`, `used_mib`, `free_mib`) at each stage change and at most once each second, or one stage duration (`timings`).
- `loaded`: the `info` fields plus `ms`, `default_sample_method`, `default_scheduler`, `image_generation` and `video_generation`. `capabilities` has `image` and `video` when the model supports them.
- `done`: `outputs`, `timings` (`load_ms`, `encode_prompt_ms`, `sample_ms`, `decode_ms`, `total_ms`) and `peak_mib`.

The engine sends `hello` first. It has `bsd_api_version`, `sd_commit`, `capabilities` and `devices`.

Each reply and each event for a command has the same `id`. Job events also have `job`.

Error codes: `out_of_memory`, `missing_file`, `bad_model`, `cancelled`, `unsupported`, `invalid_argument`, `not_loaded`, `busy`, `internal`. An `error` has `message` (safe to show) and can have `detail` (the last engine log lines).

## Memory check

Before a load, the engine adds the file sizes of all parts and adds 1 GiB. It compares the sum with the free memory of the largest GPU device. If `params_backend` puts a module on `cpu` or `disk`, the engine also adds free CPU memory. If the sum is larger, the load stops with `out_of_memory`. Set `skip_memory_check` to load anyway.
