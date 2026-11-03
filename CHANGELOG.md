# CHANGELOG



---
18:00 Monday 2 November 32026

All six engine fixes are done and tested with real models, but I couldn't report back to `brancusi-desktop-f8`. That session no longer exists. The session list now shows `brancusi-desktop-dc` and `brancusi-desktop-a0`. I did not message either one, so tell me if one of them should get the report. Nothing is committed or pushed.

**Fixed:**
- **E5 (SD 1.5 NaN on Metal):** the Metal GROUP_NORM kernel ignores the batch dimension, so the second CFG image was never normalized. [src/core/ggml_extend.cpp](src/core/ggml_extend.cpp) now folds the batch into the groups. SD 1.5 on Metal renders a correct apple, and the `ggml/` submodule is unchanged.
- **E4 (conversion abort):** `bsd_convert_parts()` catches the exception and returns `BSD_ERR_BAD_MODEL`, and integer tensors stay as they are. The MOSS VAE now converts to f16 without aborting.
- **E3 (Z-Image):** the problem was in detection, not the merge check. [z_image.hpp](src/model/diffusion/z_image.hpp) read only the first qkv part, so it set `num_kv_heads` to 1. It now sums the parts, and Z-Image Turbo loads.
- **E2 (ERNIE and MOSS text encoders):** [name_conversion.cpp](src/name_conversion.cpp) now maps the `llm.language_model.` names. ERNIE and MOSS load. I did not add an explicit ignore for the vision tensors.
- **E1 (FLUX.3 detection):** [model_loader.cpp](src/model_loader.cpp) now detects FLUX.3, and `inspect` reports "Flux.3 Image Klein".
- **L1 (memory check):** [l1_engine.cpp](brancusi/src/l1_engine.cpp) no longer counts unified memory twice on Apple Silicon. A `disk` text encoder is counted as released before sampling. Z-Image Turbo with `te=disk` now passes. Without `params_backend` it still fails with `out_of_memory` (31.6 GiB).

**Verified:** `bsd-tests` passes. Load checks passed on ERNIE, Z-Image, MOSS and FLUX.3 (inspect only). The full render ran only on SD 1.5.

**Still open:** E6, the release, which is your decision. I didn't try FLUX.3 or ERNIE renders.

**Local build:** [build/bin/brancusi-engine](build/bin/brancusi-engine) (Release, Metal).

---
18:32 Monday 2 November 2026

Review of the coding steps for the engine fixes E1 to E5 and L1. Each step lists the file, the cause and the test. All tests used real models and the local build. I did not commit or push.

**Step 1. CodeRabbit finding on `rules`** ([host/main.cpp](brancusi/host/main.cpp), [README.md](brancusi/README.md))
- Cause: `do_convert` read `rules` with `str_or`. A JSON array gave an empty string with no error.
- Change: `rules` can be a string or a list of strings. The engine joins list items with `,`. Other types give `invalid_argument`. The README states both forms.
- Test: syntax check of `host/main.cpp`. STE lint of the README text: 0 violations.

**Step 2. E4, conversion abort** ([l1_engine.cpp](brancusi/src/l1_engine.cpp), [convert.cpp](src/convert.cpp))
- Cause: `convert_with_components()` threw an exception for an I64 tensor (`bn.num_batches_tracked`). Nothing caught it, so the engine stopped.
- Change: `bsd_convert_parts()` catches `std::exception` and any other exception. It returns `BSD_ERR_BAD_MODEL`. `get_export_tensor_type()` keeps I8, I16, I32 and I64 tensors as they are.
- Test: MOSS Image SFT VAE converted to f16 in 0.6 s. The reply was `ok`.

**Step 3. E1, FLUX.3 detection** ([model_loader.cpp](src/model_loader.cpp))
- Cause: diffusers FLUX.3 files have no `emb_in.image` tensor, so the loader chose Flux.
- Change: `get_sd_version()` returns `VERSION_FLUX3` when `mode_blocks.image.0.attn.to_q.weight` or `early_stream_modulations.image.linear.weight` is present.
- Test: `inspect` on FLUX.3 Klein reports "Flux.3 Image Klein".

**Step 4. E2, text encoder names** ([name_conversion.cpp](src/name_conversion.cpp))
- Cause: ERNIE uses `llm.language_model.model.*` and MOSS uses `llm.language_model.*`. The engine mapped only `llm.model.language_model.*`.
- Change: both prefixes map to `llm.model.`. There is no explicit ignore rule for vision tensors.
- Test: ERNIE Image Turbo and MOSS Image SFT load.

**Step 5. E3, Z-Image** ([z_image.hpp](src/model/diffusion/z_image.hpp))
- Cause: the error text came from `validate_tensor()` in `model_manager.cpp`, but the check was correct. `detect_from_weights()` read only the first qkv part (3840). It set `num_kv_heads` to 1, so the model expected a qkv size of 4096 instead of 11520. A temporary log line showed this. I removed the log line.
- Change: the detection adds the sizes of `qkv.weight`, `qkv.weight.1` and `qkv.weight.2`.
- Test: Z-Image Turbo loads.

**Step 6. L1, memory check** ([l1_engine.cpp](brancusi/src/l1_engine.cpp))
- Cause: on Apple Silicon the check added CPU free memory to GPU free memory. These are the same memory. It also counted `disk` parts as resident.
- Change: no addition of CPU memory on unified memory. New helpers `is_te_part()` and `is_disk_part()` read `params_backend` (`disk`, `all=disk`, `te=disk`). The check treats a `disk` text encoder as released before sampling.
- Test: Z-Image Turbo without `params_backend` gives `out_of_memory` (31.6 GiB). With `te=disk` the check passes and the model loads.

**Step 7. E5, white image from SD 1.5 on Metal** ([ggml_extend.cpp](src/core/ggml_extend.cpp))
- Symptom: all pixels were 255. The CPU backend gave a good image. A job with `cfg_scale` 1 (batch 1) also gave a good image. Moving the text encoder and the VAE to the CPU did not help.
- Method: a temporary callback (`sd_set_backend_eval_callback`) read each node after it ran. It found the first node with a NaN, then the first node with a very large value. The first large value (2e10) was in a GROUP_NORM node with a batch of 2. Later nodes converted this value to f16 and gave inf and NaN.
- Cause: the Metal GROUP_NORM dispatch uses `ngrp` thread groups and ignores `ne[3]`. The second image of a CFG batch is never normalized.
- Change: `ggml_ext_group_norm_folded()` folds the batch into the channels and uses `N * groups` groups. Both group-norm call sites use it. I did not change the `ggml/` submodule. I removed the callback and the test code from `host/main.cpp`.
- Test: SD 1.5 on MTL0 gives a good image. The mean pixel value is 82. The CPU gives 83.

**Verification**
- `bsd-tests`: all passed after each build.
- Real-model load checks: ERNIE, Z-Image, MOSS. `inspect`: FLUX.3. Full render: SD 1.5 only.
- CodeRabbit (`--base origin/brancusi`, 13 files): the first run found 2 `rules` findings (fixed in step 1). Later runs found 2 open findings, listed below.

**Open**
- CodeRabbit: the peak-memory formula in `bsd_load` for `disk` parts (`max(resident, te_disk) + largest_disk`). It can under-report or over-report. Proposed: `needed += max(te_disk, largest_disk)`.
- CodeRabbit: `bsd_inspect` runs the model loader on a tokenizer part.
- E6: engine release. The user decides.
- Not tested: renders of FLUX.3, ERNIE, Z-Image and Qwen.
---
18:58 Monday 2 November 2026

Two CodeRabbit fixes. Each step lists the file, the cause, the change and the test.

**Step 8. Peak memory for `disk` parts** ([l1_engine.cpp](brancusi/src/l1_engine.cpp))
- Cause: kept parts stay loaded. A later job reloads a `disk` text encoder while the kept parts are in memory. My formula `max(kept, te_disk) + largest_disk` gave wrong sums in both directions.
- Change: `needed += max(te_disk, largest_disk)`. Kept parts add to the largest `disk` part.
- Test: Z-Image Turbo with `te=disk` now gives `out_of_memory` (31.6 GiB). This is correct, because the kept f32 diffusion model and the text encoder do not fit together in 28.1 GiB. With `all=disk` the model loads. `bsd-tests` passes.

**Step 9. `inspect` with a tokenizer part** ([l2_inspect.cpp](brancusi/src/l2_inspect.cpp))
- Cause: the per-part loop ran `ModelLoader::init_from_file()` on the tokenizer file. A `tokenizer.json` file is not a model file, so `inspect` could return `BSD_ERR_BAD_MODEL` for a part set that loads.
- Change: the loop skips `BSD_COMPONENT_TOKENIZER`.
- Test: `inspect` of Z-Image Turbo with a `tokenizer.json` part returns family "Z-Image".
---
19:20 Monday 2 November 2026

**Step 10. CI action versions** ([brancusi-engine.yml](.github/workflows/brancusi-engine.yml))
- Cause: `actions/checkout@v4` and `actions/upload-artifact@v4` target Node.js 20, which GitHub deprecates. Each CI run showed a warning.
- Change: `checkout` v7, `upload-artifact` v7, `download-artifact` v8. These are the latest major versions. I did not change `Jimver/cuda-toolkit`, because only the CUDA variant uses it and that variant is off.
- Test: see the CI run for the commit that holds this entry.
---
19:56 Monday 2 November 2026

**Version 0.1.1.** The `hello` and `info` replies now tell a release build from a local build.

**Step 11. Engine version in `hello` and `info`** ([main.cpp](brancusi/host/main.cpp), [CMakeLists.txt](brancusi/CMakeLists.txt))
- Cause: the replies had only `sd_version` and `sd_commit`. The release and a local build both sent `9008a3a`, so the app could not tell them apart.
- Change: a new `version` field. `BRANCUSI_VERSION` (0.1.1) and `BRANCUSI_SD_BASE` (the upstream base commit, `37bc88b`) in `brancusi/CMakeLists.txt` set it. A build at the clean tag `brancusi-v0.1.1` sends `0.1.1+sd.37bc88b`. Other builds send a git-describe value, for example `0.1.1-dev.4.g2203686.dirty+sd.37bc88b`.
- Test: `info` on a local build returns the dev value.

**Step 12. Commands in `capabilities`** ([main.cpp](brancusi/host/main.cpp))
- Change: `capabilities` now also has `convert`, `inspect`, `schedulers` and `hardware`.
- Test: `hello` and `info` list the four commands. `bsd-tests` passes.

---
20:44 Monday 2 November 2026

**Version 0.1.2.** Release build.

---
21:40 Monday 2 November 2026

**Step 13. `peak_bytes` in the `done` reply** ([main.cpp](brancusi/host/main.cpp))
- Cause: `peak_mib` used the device memory only at stage changes and steps. It did not see the peak of a lazy load or a VAE decode.
- Change: a thread reads the main device memory (`ggml_backend_dev_memory`, Layer 1) each 50 ms during the job. `done` has the new field `peak_bytes`. `peak_mib` stays.
- Test: Z-Image Turbo, 512x512, 8 steps: `peak_bytes` 12713984000 with all parts kept, 7492567040 with `te=disk,diffusion=disk`.

**Step 14. `auto_fit` load option** ([bsd.h](brancusi/include/bsd.h), [l1_engine.cpp](brancusi/src/l1_engine.cpp), [main.cpp](brancusi/host/main.cpp))
- Cause: when the model did not fit, the load stopped with `out_of_memory`. Upstream auto-fit can put a part in CPU memory, which is a second copy on unified memory.
- Change: `auto_fit` `true` adds `te=disk`, then `diffusion=disk`, then `vae=disk` until the pre-load memory check passes. `true` and `false` set upstream auto-fit off. `loaded` and `info` have `params_backend`. The pre-load check uses a `max_vram` budget. A module entry in `params_backend` now overrides the default in any order, as in upstream. `hello` lists `auto_fit` and `peak_bytes` in `capabilities`.
- Test: `bsd-tests` checks the plan with sparse files. A load with `auto_fit` and `max_vram` 9 chose `te=disk,diffusion=disk` and made the image.

---
22:23 Monday 2 November 2026

**Version 0.1.3.** Release build.

---
21:26 Tuesday 3 November 2026

**Step 15. `set_component` for the VAE and the TAESD** ([l2_parts.cpp](brancusi/src/l2_parts.cpp), [bsd.h](brancusi/include/bsd.h), [main.cpp](brancusi/host/main.cpp))
- Cause: a change of the VAE needed a full `load`. The app loaded all weights again.
- Change: Layer 2. `bsd_set_component()` and the command `set_component` load, replace or remove the VAE or the TAESD on the loaded model. The code uses upstream `apply_model_update()` with the VAE runner group, as ControlNet does. If the new file does not fit, the old part stays. The text encoder and the diffusion model share one runner group, so they stay unsupported. `hello` lists `set_component`.
- Test: Ernie Image Turbo Q4_K_M. A VAE swap takes 11 ms. The next job keeps the text encoder and the diffusion weights in memory (sampling 15.6 s, not 24.1 s). A mismatched VAE gives `bad_model` and the old VAE still makes images. `bsd-tests` passes.

**Step 16. Memory of each part** ([l2_parts.cpp](brancusi/src/l2_parts.cpp), [main.cpp](brancusi/host/main.cpp))
- Change: Layer 2. `bsd_parts()` reads the ModelManager tensor states (a `friend` hook). `loaded`, `memory` and `done` have `parts`: `name`, `backend`, `bytes`, `bytes_gpu`, `bytes_cpu` and `on_disk`. `hello` lists `part_memory`.
- Test: after a job on Ernie Image Turbo, the text encoder, the diffusion model and the VAE report their bytes in `MTL0` memory.

**Step 17. Stage events with exact times** ([image.cpp](src/pipeline/image.cpp), [bsd_hook.h](brancusi/include/bsd_hook.h), [l1_engine.cpp](brancusi/src/l1_engine.cpp), [main.cpp](brancusi/host/main.cpp))
- Cause: `bsd_detect_stage` read the stage from log text. The times had a resolution of 10 ms and depended on the log format.
- Change: Layer 2. `BRANCUSI:` hooks in `generate_image()` mark `encode_prompt`, `sample` and `decode` on the steady clock. `stage` has `start_us`. `metrics` has `span` (`start_us`, `end_us`, `ms`). `done` has `stages`. `bsd_detect_stage` stays for the `load` stage and as the fallback. `hello` lists `stage_timing`.
- Test: `bsd-tests` checks the order, the times and the close on early exit. Each stage of a real job has a span.

---
21:46 Tuesday 3 November 2026

**Version 0.1.4.** Release build.
