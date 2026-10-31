# brancusi/

Brancusi code in the stable-diffusion.cpp fork. Upstream code stays unchanged where possible. All our code is in this folder.

Pinned upstream release: `master-1028-37bc88b`.

## Contents

| Path | Contents |
|---|---|
| `include/bsd.h` | The `bsd_*` C API. Plain C ABI, versioned with `BSD_API_VERSION`. |
| `src/l1_*.cpp` | Layer 1. Uses only `include/stable-diffusion.h` and the public ggml backend API. |
| `src/l2_*.cpp` | Layer 2. Uses upstream internals. None yet. |
| `host/main.cpp` | `brancusi-engine`. JSON lines on stdin and stdout. |
| `tests/test_bsd.cpp` | Unit tests. They need no model. |

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

## Protocol (API version 1)

One JSON object on each line. stdout carries only protocol messages. The engine sends all other output, including upstream `printf` output, to stderr. The engine exits when stdin closes.

Commands (core to engine):

| `cmd` | Fields | Reply |
|---|---|---|
| `info` | – | `info` |
| `memory` | – | `memory` with `devices` |
| `load` | `components` (part name to path), `options` | `stage`, `log`, then `loaded` or `error` |
| `unload` | – | `unloaded` or `error` |
| `generate` | `job`, `params`, `output_dir` | `stage`, `progress`, `log`, then `done` or `error` |
| `cancel` | `job`, `mode` (`all` or `new_latents`) | `ok`. The running job then ends with `error` code `cancelled`. |
| `log_level` | `level` (`debug`, `verbose`, `info`, `warn`, `error`) | `ok` |
| `shutdown` | – | The engine stops the running job and exits. |

Part names are the `bsd_component_t` names: `model`, `diffusion`, `vae`, `taesd`, `llm`, `t5xxl`, `clip_l` and others.

`load` options: `backend`, `params_backend`, `max_vram`, `flash_attn`, `model_args`, `weight_type`, `mmap`, `eager_load`, `tae_preview_only`, `skip_memory_check`.

`generate` params: `prompt`, `negative_prompt`, `width`, `height`, `steps`, `seed`, `cfg_scale`, `sample_method`, `scheduler`, `sigmas` (a list, or the preset `qwen-image-2.1-turbo`), `batch_count`, `flow_shift`.

The engine sends `hello` first. It has `bsd_api_version`, `sd_commit`, `capabilities` and `devices`.

Each reply and each event for a command has the same `id`. Job events also have `job`.

Error codes: `out_of_memory`, `missing_file`, `bad_model`, `cancelled`, `unsupported`, `invalid_argument`, `not_loaded`, `busy`, `internal`. An `error` has `message` (safe to show) and can have `detail` (the last engine log lines).

## Memory check

Before a load, the engine adds the file sizes of all parts and adds 1 GiB. It compares the sum with the free memory of the largest GPU device. If `params_backend` puts a module on `cpu` or `disk`, the engine also adds free CPU memory. If the sum is larger, the load stops with `out_of_memory`. Set `skip_memory_check` to load anyway.
