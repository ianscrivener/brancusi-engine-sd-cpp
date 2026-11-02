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
