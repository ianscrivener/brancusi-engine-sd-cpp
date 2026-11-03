# brancusi/

Brancusi code in the stable-diffusion.cpp fork. Upstream code stays unchanged where possible. All our code is in this folder.

Pinned upstream release: `master-1028-37bc88b`.

## Contents

| Path | Contents |
|---|---|
| `include/bsd.h` | The `bsd_*` C API. Plain C ABI, versioned with `BSD_API_VERSION`. |
| `include/bsd_hook.h` | The stage hook that upstream `generate_image()` calls (Layer 2). |
| `src/l1_*.cpp` | Layer 1. Uses only `include/stable-diffusion.h` and the public ggml backend API. |
| `src/l2.h`, `src/l2_*.cpp` | Layer 2. Uses upstream internals. Built as the `bsd_l2` object library with the upstream include folders and definitions. Now: `l2_inspect.cpp` (`ModelLoader`) and `l2_parts.cpp` (parts of a live context). |
| `host/main.cpp` | `brancusi-engine`. JSON lines on stdin and stdout. |
| `tests/test_bsd.cpp` | Unit tests. They need no model. |
| `scripts/package.sh` | Makes the zip for one variant. Prints its size and SHA-256. |

## Upstream hooks

Each change to an upstream file has a `BRANCUSI:` comment.

| File | Hook |
|---|---|
| `CMakeLists.txt` | `add_subdirectory(brancusi)` behind the `SD_BUILD_BRANCUSI` option (default `ON`). |
| `src/pipeline/image.cpp` | `bsd_stage_scope` marks the start and the end of `encode_prompt`, `sample` and `decode`. |
| `src/model_manager.h` | `friend struct BsdPartProbe`: gives `l2_parts.cpp` the tensor states for part memory. |
| `src/stable-diffusion.cpp` | `bsd_l2_context()`: gives `l2_parts.cpp` the internals of `sd_ctx_t`. |

## Build (macOS, Metal)

```sh
cmake -B build -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target brancusi-engine bsd-tests
./build/bin/bsd-tests
```

For other backends, use the upstream CMake options (`SD_CUDA`, `SD_VULKAN`, `SD_HIPBLAS` and others).

## CI and releases

The workflow `.github/workflows/brancusi-engine.yml` builds `brancusi-engine` for each variant in its matrix. It runs `bsd-tests` on the Metal and CPU builds. Now the matrix has only `darwin-arm64-metal`. To make CI faster, the other variants have comment marks.

To make a release, run `just bump`. It adds 1 to the patch version (`just bump minor` or `just bump major` for the other parts), adds a CHANGELOG entry, commits, pushes, pushes the tag and watches CI. The tree must be clean. Or push a tag `brancusi-v<version>` by hand, for example `brancusi-v0.1.1`. The tag must match `BRANCUSI_VERSION` in `brancusi/CMakeLists.txt`. The release gets one zip for each variant and a `manifest.json` with the file, size and SHA-256 of each zip. Copy these values into `engine.json` in `brancusi-desktop`.

The upstream `.gitignore` ignores `tests/`. To add a new file in `brancusi/tests/`, use `git add -f`.

## Protocol (API version 1)

One JSON object on each line. stdout carries only protocol messages. The engine sends all other output, including upstream `printf` output, to stderr. The engine exits when stdin closes.

Commands (core to engine):

| `cmd` | Fields | Reply |
|---|---|---|
| `info` | – | `info` |
| `memory` | – | `memory` with `devices`, and `parts` when a model is loaded and no command runs |
| `load` | `components` (part name to path), `options` | `stage`, `log`, then `loaded` or `error` |
| `unload` | – | `unloaded` or `error` |
| `generate` | `job`, `params`, `output_dir`, `preview` | `stage`, `progress`, `preview`, `metrics`, `log`, then `done` or `error` |
| `cancel` | `job`, `mode` (`all` or `new_latents`) | `ok`. The running job then ends with `error` code `cancelled`. |
| `convert` | `input` or `components`, `output`, `type`, `rules` (optional), `convert_name` (optional) | `stage` (`convert`), `progress` (step = tensors written), `log`, then `ok` (`output`, `bytes`, `ms`) or `error` |
| `inspect` | `components` (part name to path), `weight_type` (optional) | `log`, then `inspect` or `error` |
| `set_component` | `component` (`vae` or `taesd`), `path` (a file, or `null` to remove the part) | `stage` (`load`), `log`, then `loaded` or `error` |
| `hardware` | – | `hardware` |
| `schedulers` | – | `schedulers` |
| `log_level` | `level` (`debug`, `verbose`, `info`, `warn`, `error`) | `ok` |
| `shutdown` | – | The engine stops the running job and exits. |

`convert` writes one file with a new weight type (upstream `convert_with_components()`). Give one `input` file, or `components` to merge parts into one file. `components` can have only `model`, `diffusion`, `clip_l`, `clip_g`, `t5xxl` and `vae`. Other parts (for example `llm`) give `unsupported`. `type` is a type name such as `q8_0`. `rules` is optional. It holds `regex=type` pairs for single tensors, as one comma-separated string (`"regex=type,regex=type"`) or as a list of strings. The rules match the tensor names with the part prefix (`model.diffusion_model.`, `text_encoders.clip_l.transformer.`, `vae.`). With `convert_name`, they match the converted names (for example `first_stage_model.` for the VAE). A conversion needs no loaded model: with a model loaded, the engine replies `busy`. A conversion cannot be cancelled: stop the engine process.

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

`set_component` (Layer 2) loads, replaces or removes the VAE or the TAESD on the loaded model, without a model reload. It uses the same upstream path as ControlNet (`apply_model_update`). The text encoder and the diffusion model weights stay in memory. The reply is a `loaded` event with the `info` fields, `ms`, `component`, `path` and `parts`. The command replies `busy` while a command runs, and `not_loaded` without a model. Other parts give `unsupported`: the text encoder and the diffusion model are one upstream runner group, so a change rebuilds both. If the new file does not fit the model, the reply is `bad_model` and the old part stays. Upstream then rebuilds the runners, so the next job loads the weights again. If upstream cannot rebuild the old part, the reply is `internal` and the engine unloads the model. Removing the VAE of a model that has no built-in VAE gives `bad_model`. The `hello` capability `set_component` tells that the engine has the command.

`hardware` returns `system_info`, `features` (the `system_info` flags, for example `{"NEON":true}`), `physical_cores`, `n_threads` and `devices`. `schedulers` returns `samplers`, `schedulers` and `types` (weight type names) from the engine tables. Both commands run at once, also while a job runs.

Part names are the `bsd_component_t` names: `model`, `diffusion`, `vae`, `taesd`, `llm`, `t5xxl`, `clip_l` and others.

`load` options: `backend`, `params_backend`, `max_vram`, `flash_attn`, `model_args`, `weight_type`, `mmap`, `eager_load`, `tae_preview_only`, `skip_memory_check`, `auto_fit`.

`auto_fit` (`true`, `false`, `"on"` or `"off"`):

- Not set: upstream auto-fit. It is on when `params_backend` is not set. On a Mac, it can put a part in CPU memory, which is a second copy in the same memory.
- `true`: the engine makes the plan. If the model does not fit (see [Memory check](#memory-check)), the engine adds `te=disk`, then `diffusion=disk`, then `vae=disk` to `params_backend` until it fits. It does not change a module that `params_backend` sets by name. Upstream auto-fit is off. If the model does not fit with all modules on `disk`, the load stops with `out_of_memory`.
- `false`: upstream auto-fit is off. The engine does not change `params_backend`.

The `loaded` reply has `params_backend`: the value that the load used, as given or as auto-fit set it.

`generate` params: `prompt`, `negative_prompt`, `width`, `height`, `steps`, `seed`, `cfg_scale`, `sample_method`, `scheduler`, `sigmas` (a list, or the preset `qwen-image-2.1-turbo`), `batch_count`, `flow_shift`.

`generate` preview: `mode` (`none`, `proj`, `tae` or `vae`. Default `none`), `interval` (steps, default 2), `max_px` (default 256). The engine sends at most 4 previews each second, as base64 JPEG.

Events during a job:

- `stage`: `load`, `encode_prompt`, `sample` or `decode`. The stages `encode_prompt`, `sample` and `decode` come from hooks in upstream `generate_image()` (Layer 2) and have `start_us`: the start on the engine monotonic clock, in microseconds. Layer 1 finds the `load` stage in the upstream log text (`bsd_detect_stage`). The log text is the fallback when an engine has no hooks.
- `progress`: `step`, `steps`, `sec_per_step`. Upstream also sends tensor loading as progress. Use it as a sampling step only in the `sample` stage.
- `metrics`: device memory (`device`, `used_mib`, `free_mib`) at each stage change and at most once each second, or one stage duration (`timings`). For a hooked stage, the duration event also has `span`: `stage`, `start_us`, `end_us` and `ms` (exact, from the same clock).
- `loaded`: the `info` fields plus `ms`, `default_sample_method`, `default_scheduler`, `image_generation`, `video_generation` and `params_backend`. `capabilities` has `image` and `video` when the model supports them.
- `done`: `outputs`, `timings` (`load_ms`, `encode_prompt_ms`, `sample_ms`, `decode_ms`, `total_ms`), `stages` (the `span` of each hooked stage), `parts`, `peak_bytes` and `peak_mib`.

`parts` (Layer 2) is the weight memory of each part. The `loaded` event, the `memory` reply and the `done` event can have it. A command that runs gives no `parts` in the `memory` reply, because the job changes the part state. Each entry:

```json
{"name":"diffusion","backend":"MTL0","bytes":5019095552,"bytes_gpu":5019095552,"bytes_cpu":0,"on_disk":0}
```

- `name`: `te`, `diffusion`, `vae`, `taesd` (the TAESD as the VAE, or for previews), `control_net` and others.
- `backend`: the params backend of the part, or `disk`.
- `bytes`: all weights of the part. `bytes_gpu` and `bytes_cpu` are the weights in device and host memory now. `bytes_gpu` also counts a staged copy for the compute backend. `on_disk` is the weights that are not in memory now. They load from the file when the part runs. After `load`, all bytes are `on_disk`, because upstream loads weights on first use.

`peak_bytes` is the peak memory in use on the main device (the first GPU, or the CPU) during the job. It includes the weights that stay loaded. A thread reads the device memory (`ggml_backend_dev_memory`) each 50 ms while the job runs. On Metal, the value is the Metal memory of the engine process. The value cannot be more than the Metal working-set limit. `peak_mib` is the same value in MiB.

Sample (Z-Image Turbo, 512x512, 8 steps, M4 Max):

```json
{"event":"loaded","params_backend":"te=disk,diffusion=disk","ms":160,...}
{"event":"done","peak_bytes":7492567040,"peak_mib":7145,"timings":{"load_ms":5180,"encode_prompt_ms":4260,"sample_ms":20800,"decode_ms":2090,"total_ms":27189},...}
```

The engine sends `hello` first. It has `version`, `bsd_api_version`, `sd_commit`, `capabilities` and `devices`. The `info` reply has the same fields.

- `version`: the engine version and the upstream base commit. A release build sends `0.1.1+sd.37bc88b`. Other builds send `0.1.1-dev.<commits>.g<commit>[.dirty]+sd.37bc88b`, so the app can see a local build. `BRANCUSI_VERSION` and `BRANCUSI_SD_BASE` in `brancusi/CMakeLists.txt` set the values. CMake reads git at configure time, so run the configure step again after a commit.
- `capabilities`: the model flags (`image`, `video`, `cancel` and others), the `load` option `auto_fit`, the commands `convert`, `inspect`, `schedulers` and `hardware`, and `peak_bytes` (the `done` field).

Each reply and each event for a command has the same `id`. Job events also have `job`.

Error codes: `out_of_memory`, `missing_file`, `bad_model`, `cancelled`, `unsupported`, `invalid_argument`, `not_loaded`, `busy`, `internal`. An `error` has `message` (safe to show) and can have `detail` (the last engine log lines).

## Memory check

Before a load, the engine adds the file sizes of the kept parts, the size of the largest `disk` part and 1 GiB. A `disk` part loads and releases in turn, so only the largest one counts. The engine compares the sum with the free memory of the largest GPU device. If `max_vram` gives a budget for that device (`<GiB>`, `all=<GiB>` or `<device>=<GiB>`), the engine uses the smaller value. A negative budget is the free memory less that many GiB. If `params_backend` puts a module on `cpu` or `disk`, the engine also adds free CPU memory, but not on unified memory (Apple Silicon). If the sum is larger, the load stops with `out_of_memory`, or `auto_fit` puts modules on `disk`. Set `skip_memory_check` to load anyway.

`bsd_load_bytes()` and `bsd_fit_params_backend()` in `bsd.h` give the same sum and plan to a host or a test.
