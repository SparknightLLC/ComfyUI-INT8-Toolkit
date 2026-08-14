# Changelog

All notable changes to this project will be documented in this file.

<details open><summary>2.2.2 - 13 August 2026</summary>

### Changed

- Raised the minimum supported ComfyUI version to 0.32.0.

### Fixed

- Adopted ComfyUI's context-managed weight casting for floating-point, W4A4, and W4A8 linear execution so Dynamic VRAM offload cleanup still runs when inference raises an exception.

</details>

<details><summary>2.2.1 - 8 August 2026</summary>

### Fixed

- Made adapter and lazy-compile output caches weakly reference their MODEL outputs so ComfyUI can release the previous quantized model before mapping a replacement checkpoint. Already-quantized `as_needed` inputs now also run architecture-transition cleanup before returning unchanged.

</details>

<details><summary>2.2.0 - 8 August 2026</summary>

### Added

- Added experimental native `asym_w4a8_int8` checkpoint loading, on-the-fly conversion, MODEL adaptation, inference, and native metadata export through ComfyUI/comfy-kitchen's `AsymW4A8Int8Layout`.
- Added W4A8 support to Stochastic LoRA patching through native dequantize, FP32 patch, and one-time requantization.
- Added a temporary compiler-safe custom-op and FakeTensor boundary for W4A8 linear execution, allowing Quantized Lazy Torch Compile to optimize the surrounding transformer graph while the native kernel remains opaque.
- Added preliminary RTX 3090/Krea2 W4A8 benchmarks covering warm throughput, reported model-weight footprint, and early visual observations.

### Changed

- Exposed `w4a8` as a distinct quantization mode. It retains architecture `keep_float` exclusions but does not use the layer-mixing `int4_mixed_ratio` or `int4_sensitive` policy.
- Routed W4A8 targets in Dynamic LoRA mode through ComfyUI's Standard patch path and added an explicit console warning. INT8 and W4A4 targets retain runtime deltas.
- Made W4A8 compile support capability-driven: the Toolkit shim is used while upstream lacks a compiler-safe operator, and registration failures return an eager MODEL with a detailed warning.

### Fixed

- Kept native low-bit tensor replacement format-specific so W4A4 and W4A8 tensors cannot be mistaken for one another during patch reconstruction.
- Made `as_needed` ignore quantized modules temporarily installed on a shared model by another cached patcher, preventing format switches from returning an unowned stale quantization state and subsequently loading the full floating-point model. Toolkit object-patch cleanup now also survives extension hot reloads.

### Removed

- None.

</details>

<details><summary>2.1.0 - 7 August 2026</summary>

### Added

- Added preliminary MiniMax H3 keep-float and INT4-sensitive quantization tiers, including automatic architecture detection.
- Added native ConvRot metadata regression coverage for heterogeneous group sizes and invalid checkpoint metadata.

### Fixed

- Preserved each native INT8 ConvRot layer's checkpoint-defined group size at runtime, including MiniMax H3 AdaLN projections that use 64-channel groups.
- Validated ConvRot group divisibility and Hadamard state while loading and saving so malformed checkpoints fail with a clear error instead of crashing later or exporting inconsistent metadata.

</details>

<details><summary>2.0.0 - 4 August 2026</summary>

### Added

- Added `LoRA Stack Entry (Quantized)` and `Apply LoRA Stack (Quantized)`. Entry nodes produce independently bypassable path/strength descriptors, while the apply node uses ComfyUI v3 autogrow inputs for up to 100 LoRAs and shares the existing Standard, Stochastic, and Dynamic patch implementations.
- Added native ComfyUI/comfy-kitchen `convrot_w4a4` checkpoint loading, on-the-fly quantization, MODEL adaptation, inference, LoRA re-quantization, and native metadata export.
- Added architecture-derived `int4_sensitive` tiers for Krea 2 and Anima residual write-back projections, and kept Anima's final latent projection floating-point.
- Added INT4-aware save summaries and runtime Dynamic LoRA support for native INT4 layers.
- Added a temporary compiler-safe custom-op and FakeTensor boundary around comfy-kitchen native ConvRot INT4 linear execution. Eager inference keeps native layout dispatch, compiled inference uses the shim only while upstream support is absent, and the lazy compile node falls back to an uncompiled model with a detailed warning if neither route is available.
- Added reporting for each distinct TorchDynamo guard failure before a lazy-compile cache miss and graph-cache growth logs with per-dispatch timing, making unexpected graph-family compilation attributable to its exact guard and block.
- Added the model-agnostic `int4_mixed_ratio` input, defaulting to `0.2`, to both quantization nodes. Architecture-specific patterns receive priority while the remaining ConvRot INT8 budget is selected deterministically across W4-compatible linears.

### Changed

- Rebranded the human-facing package as ComfyUI Quantization Toolkit while preserving the repository name, immutable package name, and existing node IDs.
- Routed Toolkit runtime output through ComfyUI's configured logger.
- Made autogrow LoRA collection explicitly follow numeric `lora_1`, `lora_2`, ... slot order, independent of mapping insertion order, and recognize native ComfyUI INT8/W4A4 modules through their public `quant_format` marker.
- Logged the effective relative path and strength of each applied LoRA stack entry so legacy and autogrow configurations can be compared directly.
- Consolidated compatible native Dynamic LoRA factors once during synchronization to reduce first-inference transfers, temporary VRAM, and Torch Compile graph complexity without changing the summed LoRA delta.
- Replaced the `Enable Quantization on MODEL` boolean with `as_needed`, `always`, and `bypass` choices. `as_needed` is the new default: it converts FP8 and floating-point inputs while leaving MODEL inputs containing Toolkit-supported INT8 or W4A4 layers unchanged.
- Renamed the lazy compile node's `dynamic` input to `dynamic_shape_tracing` to distinguish TorchDynamo shape specialization from ComfyUI Dynamic VRAM.
- Clarified that `Quantized Lazy Torch Compile` follows ComfyUI's stock behavior by demoting only its MODEL branch from Dynamic VRAM, without requiring Dynamic VRAM to be disabled globally.
- Unified quantization selection under the `quantization_mode` input with `int8`, `int8_convrot`, `int8_quarot`, `int8_hadanorm`, `int4_mixed`, and `int4_full` modes.
- Reworked architecture presets into explicit `keep_float` and `int4_sensitive` tiers. Existing safety exclusions remain floating-point in every mode; `int4_mixed` reserves the second tier for ConvRot INT8 while `int4_full` may promote it to W4A4.
- Preserved a structural compiled-wrapper cache when a LoRA changes on the same base model, while keeping LoRA-specific MODEL outputs separate and resetting both caches on an architecture change.
- Shared one compiled dispatcher across structurally compatible transformer blocks so a cold compile or LoRA guard change does not independently retrace every block; kept disabled runtime diagnostics and aligned `torch._int_mm` safety branches out of the common compile path.
- Normalized equivalent self-bound `functools.partial` forward patches during compiled dispatch, preventing ComfyUI-PPM's Cosmos/Anima attention patches from specializing one Dynamo graph per transformer block while preserving genuinely different forward functions.
- Kept Anima token-id validation outside the diffusion module so changing a LoRA stack does not invalidate the compiled graph through a replaced `preprocess_text_embeds` method.
- Renamed the lazy compile node's `log_compile` input to `verbose` and gated compile preparation, graph-cache, timing, and guard diagnostics behind it.
- Replaced module-global adapter and lazy-compile output caches with model-local caches and cleared the prior Toolkit cache when the base architecture changes.
- Renamed the default save prefix from `int8_models/INT8_Model` to `quantized_models/Quantized_Model`.
- Renamed the public `enable_int8` and `prepack_int8_weights` inputs to `enable_quantization` and `prepack_weights`; INT8-specific internal state remains explicitly named.
- Made mixed-ratio profiles nested while preserving the original default `0.2` selection, so reducing the ratio removes W8A8 layers without selecting a substantially different profile.
- Re-quantized source-checkpoint full-precision matrix-multiply layers and ran ConvRot INT8 through Comfy-Kitchen's fused runtime, restoring full Krea 2 block coverage and native checkpoint-class inference speed.

### Fixed

- Merged ordinary Stochastic LoRA deltas in stable source order and FP32 before Comfy-native INT8/W4A4 requantization, preventing slot order from changing the sequence of FP16 weight additions. Non-additive adapters retain explicit stack-order semantics.
- Applied Dynamic LoRAs at runtime on baked models loaded through ComfyUI's native INT8/W4A4 operations, including source layers hidden behind Quantized Lazy Torch Compile proxies, and reported how many quantized layers were activated instead of allowing a silent no-op.
- Preserved complete LoRA coverage on mixed-precision models by applying unsupported Dynamic targets through ComfyUI's standard patch path instead of silently dropping non-INT8 layers.
- Kept native and mixed-precision Stochastic LoRA adapters on ComfyUI's standard reconstruction path, including negative strengths, so Dynamic VRAM can prefetch them safely.
- Made Toolkit-specific INT8 LoRA adapters preserve their quantization state when ComfyUI Dynamic VRAM reconstructs adapter tensors for VBAR prefetching.
- Kept Quantized Lazy Torch Compile diagnostics from colliding with ComfyUI's active sampling progress bar.
- Normalized mixed-dtype unquantized modules once during per-MODEL Dynamic VRAM demotion, preventing FP16/FP32 failures in Krea 2 keep-float layers without repeated runtime casts or changes to quantized and object-patched modules.
- Rejected unquantized save inputs before ComfyUI attempts model loading or patching.
- Fully unpatch evicted adapter weights before restoring cached source linears, preventing stale INT8 or INT4 weight backups from contaminating later quantization modes.

### Removed

- Removed bespoke bracketed console prefixes and the legacy INT8 Toolkit product name from runtime log messages.

</details>

<details><summary>1.8.1 - 6 July 2026</summary>

### Fixed

- Fixed post-quantization INT8 LoRA stack application for Krea2 and other adapter-quantized MODEL workflows by resolving pending INT8 object patches before falling back to live modules.
- Fixed stochastic and dynamic LoRA stack cache signatures so changed LoRA sets do not reuse stale adapter outputs.
- Fixed LoRA adapter wrapping detection so INT8 LoRA patches do not fall through to ComfyUI's generic quantized weight patching path.

</details>

<details><summary>1.8.0 - 2 July 2026</summary>

### Added

- Added `convrot` as an outlier method for Toolkit on-the-fly quantization and `Enable INT8 on MODEL`.
- Added Toolkit runtime support for native ComfyUI ConvRot INT8 checkpoints by reading `convrot` and `convrot_groupsize` from `.comfy_quant` metadata.
- Added native `.comfy_quant` export for Toolkit plain INT8 and ConvRot INT8 layers.
- Added `boogu` and `krea2` architecture presets.
- Added native-save diagnostics for full, partial, or missing `.comfy_quant` coverage.

### Changed

- Expanded Wan Animate exclusions with `face_adapter`, `face_encoder`, `motion_encoder`, and `pose_patch_embedding`.
- Broadened the `ltx2` exclusion preset for LTX 2.3-sensitive layers.
- Updated README guidance around ComfyUI core INT8, upstream INT8-Fast, and Toolkit-specific workflows.
- Clarified ConvRot attribution and native ComfyUI/comfy-kitchen compatibility framing.
- Moved release-history notes out of the README and into this changelog.

### Compatibility

- Plain INT8 and ConvRot Toolkit saves can include native ComfyUI metadata.
- QuaRot and HadaNorm remain Toolkit-specific because ComfyUI core does not implement their activation transforms.

</details>

<details><summary>1.7.0 - Development Runtime Update</summary>

### Added

- Added `runtime_backend` with `torch_int_mm`, `triton`, and diagnostic `triton_legacy_unsafe` modes.
- Added `small_batch_fallback` with `only_small_layers`, `always`, and `never`.
- Added CUDA-safe padding for `torch._int_mm` tiny-row and non-8-aligned output cases.
- Added experimental `prepack_int8_weights`.
- Added `INT8 Lazy Torch Compile`.
- Added stable Dynamic LoRA patch UUIDs to avoid unnecessary recomposition.
- Added `hidream o1`, `sdxl`, and opt-in `flux2_fast_unsafe` presets.

### Changed

- Changed the default backend to `torch_int_mm`.
- Reordered node inputs so `bake_loaded_loras` and logging controls are near the bottom.
- Alphabetized model type lists with `auto` first and `none` last where applicable.
- Improved cache-reuse logs and runtime diagnostics.
- Restored prior INT8 object patches before requantizing.

### Fixed

- Fixed Triton edge tiles so tail shapes no longer wrap reads with modulo offsets.
- Fixed module eligibility issues for Flux2.
- Added Torch Compile compatibility fixes.
- Added `Standard` LoRA mode for stock-style A/B comparisons.

</details>

<details><summary>0.1.1 - Early Maintenance</summary>

### Changed

- Added INT8 kernel config tuner and runtime optimization work.
- Published the extension to the Comfy Registry.

</details>

<details><summary>0.1.0 - Initial Release</summary>

### Added

- Initial Toolkit fork with INT8 tensorwise loading, on-the-fly quantization, and LoRA support.

</details>
