# ComfyUI-INT8-Toolkit

INT8 quantization stores model weights in 8-bit integers instead of higher-precision formats, reducing VRAM use and accelerating matrix-heavy inference on GPUs with strong INT8 throughput. This is especially useful on Ampere cards such as the RTX 30 series. The tradeoff is that quantization changes model numerics, so quality-sensitive layers, LoRA order, runtime backends, and Torch Compile behavior all matter.

This project began as a fork of [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast), but it is now maintained as its own INT8 toolkit for ComfyUI.

![workflow](example_workflows/load_workflow.png)

The main differentiator is `Enable INT8 on MODEL`: a `MODEL -> MODEL` adapter that converts a model loaded by ComfyUI's stock diffusion loader into this extension's INT8 runtime. Standard diffusion loaders and stock LoRA loaders can stay in the workflow, then INT8 can be enabled after those patches are in place.

Other notable features include unified INT8 LoRA nodes, stock-loader LoRA baking, selectable INT8 runtime backends, small-batch fallback controls, an experimental prepacked-weight path, a lazy Torch Compile node, and safer Triton edge-tile handling.

## Recommended Workflows

### Stock Loader Compatibility

Use this when your graph is built around ComfyUI's stock loaders.

```text
Load Diffusion Model
-> optional stock Load LoRA nodes
-> Enable INT8 on MODEL
-> optional INT8 Lazy Torch Compile
-> sampler
```

With `bake_loaded_loras` enabled, `Enable INT8 on MODEL` applies stock LoRA weight patches in float space, quantizes the resulting layer weights, and removes consumed patches so they are not applied twice. Bias patches and excluded-layer patches are left for ComfyUI to handle normally.

### Pre-Quantized Or OTF INT8 Loader

Use this when you are already loading an INT8 checkpoint, or when you want the extension to quantize eligible layers during model load.

```text
Load Diffusion Model INT8 (W8A8)
-> optional Load LoRA INT8
-> optional INT8 Lazy Torch Compile
-> sampler
```

`Load Diffusion Model INT8 (W8A8)` supports pre-quantized INT8 checkpoints and optional on-the-fly quantization for float or FP8 source weights.

### Add Or Swap LoRAs After INT8

Use this when the model is already INT8 and you want to add or change LoRAs without re-running the stock-loader bake step.

```text
INT8 model
-> Load LoRA INT8
-> sampler
```

`Load LoRA INT8` has a `mode` selector:

- `Stochastic`: applies the LoRA delta into INT8 weights with stochastic rounding.
- `Dynamic`: keeps compatible plain LoRAs as runtime additions instead of modifying INT8 weights.
- `Standard`: applies the LoRA through ComfyUI's regular MODEL patch path without INT8-specific handling. This is useful for pre-INT8 A/B testing against `Stochastic`.

## Node Summary

### Enable INT8 on MODEL

Converts an already-loaded diffusion `MODEL` to INT8 by object-patching eligible linear layers.

Settings:

- `model_type`: defaults to `auto`, which inspects the loaded `MODEL` and selects a known exclusion preset when possible. Use a specific preset to override detection. Use `none` only for experiments because it disables preset exclusions.
- `outlier_method`: choose `none`, `quarot`, or `hadanorm`. `none` is fastest. `quarot` applies Hadamard rotation for compatible layers. `hadanorm` adds static per-channel scaling, Hadamard mixing, dynamic centering, and a runtime correction term.
- `small_batch_fallback`: defaults to `only_small_layers`. This falls back to fp16/bf16 math for tiny activation row counts only when the layer has `out_features * in_features <= INT8_SMALL_LAYER_MAX_PARAMS` (default `1000000`). `always` may help very small row counts but often slows larger layers because it dequantizes full weights. `never` forces the selected INT8 backend.
- `runtime_backend`: defaults to `torch_int_mm`. `torch_int_mm` uses PyTorch `torch._int_mm` with CUDA padding for tiny row counts and non-8-aligned output columns. `triton` uses this extension's fused Triton kernels and may be faster on some model shapes. `triton_legacy_unsafe` reproduces old upstream edge-tile behavior for diagnostics only and may produce incorrect output on tail shapes.
- `prepack_int8_weights`: experimental. Keeps an extra transposed INT8 weight buffer for the Triton path. This is a simple contiguous transpose, not a cuBLASLt/CUTLASS packed layout, so it can be neutral or slower and costs roughly one extra INT8 copy of each quantized weight.
- `bake_loaded_loras`: applies current stock LoRA weight patches before quantization and removes consumed patches.
- `log_progress`: prints quantization progress and layer counts.

If `auto` cannot identify the architecture, the adapter uses a conservative union of known exclusion patterns and logs a warning. Manual `model_type` selection is faster when you know the architecture.

### Load Diffusion Model INT8 (W8A8)

Loads INT8 diffusion models using `Int8TensorwiseOps` and architecture-specific exclusion presets.

When `on_the_fly_quantization` is enabled, eligible float or FP8 weights are quantized to INT8 with per-row weight scales. The loader exposes the same `outlier_method`, `small_batch_fallback`, `runtime_backend`, and `prepack_int8_weights` controls as `Enable INT8 on MODEL`.

Supported `model_type` presets:

- `anima`
- `chroma`
- `ernie`
- `flux2`
- `ltx2`
- `qwen`
- `sdxl`
- `wan`
- `z-image`

> [!NOTE]
> SDXL can be slower with INT8 enabled because only linear layers are quantized while convolutional UNet blocks, attention kernels, and other non-INT8 work still dominate runtime. Larger transformer-heavy architectures are more likely to benefit.

### INT8 Lazy Torch Compile

Lazily applies `torch.compile` at the first sampling call, after Comfy object patches such as INT8 module replacement are active.

Recommended placement:

```text
Enable INT8 on MODEL
-> INT8 Lazy Torch Compile
-> sampler
```

Useful settings:

- `compile_transformer_blocks_only`: defaults to enabled. This compiles recognized repeated transformer block lists instead of the entire diffusion model, reducing cold-start compilation and avoiding some parent-wrapper guard churn.
- `use_guard_filter`: defaults to enabled. This ignores guards involving `transformer_options`, matching Comfy's stock compile behavior and reducing recompiles caused by per-sampling metadata.
- `dynamic`: defaults to `true`, which is usually safer when image sizes or batch shapes change. `false` can be faster for fixed shapes.
- `mode`: supports `default`, `max-autotune`, `max-autotune-no-cudagraphs`, and `reduce-overhead`. With the `inductor` backend and `use_guard_filter` enabled, the node expands the selected mode into Inductor backend options so the guard filter and mode-style tuning can be used together.
- `dynamo_cache_size_limit`: raises the process cache limit for workflows that compile many repeated modules.

The stock `TorchCompileModel` can still be faster for some architectures when whole-model compilation is stable. This lazy node is meant to be the safer INT8-aware option when object patch order, LoRAs, or architecture-specific block compilation matter.

### Load LoRA INT8

Loads one LoRA with selectable `Stochastic`, `Dynamic`, or `Standard` mode.

`Stochastic` mode is the speed-oriented INT8 path and can be queued before `Enable INT8 on MODEL` without collapsing into the same bake behavior as `Standard`. `Dynamic` mode can preserve more of the original LoRA math for compatible plain LoRAs, but it is slower because extra LoRA matmuls run during inference. `Standard` mode uses ComfyUI's regular MODEL LoRA patching without INT8-specific wrapping.

### Load LoRA Stack INT8

Loads up to 10 LoRAs with the same mode behavior as `Load LoRA INT8`.

In `Stochastic` mode, compatible LoRAs are combined before one stochastic rounding step. This is usually better than repeatedly rounding each LoRA one by one. In `Standard` mode, the stack behaves like chaining stock MODEL-only LoRA patches.

### INT8 Kernel Config

Applies fixed Triton kernel settings at runtime. Optional microbench mode tests candidate configs and prints environment variable values that can be reused later.

Environment variables:

- `INT8_TRITON_AUTOTUNE`
- `INT8_TRITON_BLOCK_M`
- `INT8_TRITON_BLOCK_N`
- `INT8_TRITON_BLOCK_K`
- `INT8_TRITON_GROUP_SIZE_M`
- `INT8_TRITON_NUM_WARPS`
- `INT8_TRITON_NUM_STAGES`
- `INT8_SMALL_BATCH_FALLBACK_MAX_ROWS`
- `INT8_SMALL_BATCH_FALLBACK_MIN_ROWS`
- `INT8_SMALL_LAYER_MAX_PARAMS`
- `INT8_SMALL_BATCH_FALLBACK_ADAPTIVE`
- `INT8_RUNTIME_STATS`
- `INT8_DYNAMIC_LORA_DEBUG`
- `INT8_DYNAMIC_LORA_BATCH`
- `INT8_DYNAMIC_LORA_BATCH_MAX_RANK`
- `INT8_FORCE_DISABLE_TORCH_COMPILE`

Keep `INT8_RUNTIME_STATS=0` for normal use. Runtime stats are useful for backend diagnosis but add console work and should not be used for performance benchmarking.

## LoRA Order And VRAM Behavior

Some LoRA orders can temporarily materialize large float tensors. On lower-VRAM cards this can look like a sudden spike and may OOM even if normal sampling would fit.

| LoRA method | Before `Enable INT8 on MODEL` | After `Enable INT8 on MODEL` | Notes |
| --- | --- | --- | --- |
| Stock `Load LoRA` | Best stock-loader path. The LoRA is baked by `Enable INT8 on MODEL` when `bake_loaded_loras` is enabled. | Avoid for INT8 layers unless testing. ComfyUI's generic patch path may dequantize or build large temporary tensors. | Easiest compatibility path before INT8. Riskier after INT8 because it is not INT8-aware. |
| `Load LoRA INT8` with `Standard` | Equivalent to a stock-style MODEL-only LoRA patch. Useful when you want the same node surface before INT8 and plan to bake later. | Mainly for testing. It intentionally skips INT8-specific handling. | Best fit for fast A/B comparisons against `Stochastic` on an existing pre-INT8 LoRA stack. |
| `Load LoRA INT8` with `Stochastic` | Carries deferred INT8-aware patches so `Enable INT8 on MODEL` can quantize the base layer first, then apply the LoRA with stochastic INT8 rounding. | Preferred for adding LoRAs after INT8. | Speed-oriented INT8 LoRA mode. |
| `Load LoRA INT8` with `Dynamic` | Not the preferred order. Dynamic LoRA state is intended for an INT8 runtime path. | Preferred when the LoRA is compatible and you can accept slower runtime. | Avoids permanently changing INT8 weights, but adds runtime matmuls. Unsupported formats fall back to static-safe patching. |

Practical guidance:

- For stock workflows, put stock `Load LoRA` before `Enable INT8 on MODEL`.
- If you want to A/B the same `Load LoRA INT8` or `Load LoRA Stack INT8` graph before INT8 conversion, switch `mode` between `Standard` and `Stochastic`.
- For already-INT8 workflows, use `Load LoRA INT8` after INT8 is enabled.
- If a graph OOMs during LoRA application but not during sampling, try the other INT8 LoRA mode, reduce the LoRA stack, or bake the LoRA before INT8 conversion.
- Leave `bake_loaded_loras` enabled unless you intentionally want patched layers skipped by the adapter.

## Runtime Backend Guidance

Backend speed is architecture- and shape-dependent, so benchmark on the workflow you actually use.

- `torch_int_mm` is the default because it is simple, robust on Windows, and tested well on Z-Image Turbo in current workflows.
- `triton` can still be faster on some architectures and shape mixes, especially when its fused dynamic-quantization path avoids extra PyTorch overhead.
- `triton_legacy_unsafe` is only for diagnosis. It keeps the old modulo edge-tile behavior and may read wrapped values for non-divisible output shapes.
- `prepack_int8_weights` currently stores only `weight.T.contiguous()`. A real Ampere-optimized packed layout would require a cuBLASLt, CUTLASS, or custom-kernel path.
- `small_batch_fallback=only_small_layers` is the recommended default based on current Anima and Z-Image Turbo testing.

## Torch Compile Guidance

Torch Compile is often the difference between "INT8 works" and "INT8 is actually fast" in ComfyUI.

Recommended defaults:

- Put `INT8 Lazy Torch Compile` after `Enable INT8 on MODEL`.
- Use `compile_transformer_blocks_only=True` unless an architecture benefits from whole-model compilation.
- Use `use_guard_filter=True` for normal ComfyUI workflows.
- Use `dynamic=true` when changing image size or batch shape. Try `dynamic=false` only for fixed workflows.
- After source-code hot reloads or failed compile experiments, restart ComfyUI before drawing conclusions. TorchDynamo guards and generated kernels can outlive local Python edits inside the same process.

If you launch ComfyUI through this project's `run.bat`, TorchInductor and Triton compile artifacts are stored under the local `torch_compile_cache` directory.

## ModelSave Round Trip

If you quantize with `on_the_fly_quantization` and save with ComfyUI `ModelSave`, the saved checkpoint can be loaded back with `Load Diffusion Model INT8 (W8A8)` without re-quantizing as long as the checkpoint includes INT8 `weight` tensors and matching `weight_scale` tensors.

## Checkpoint Notes

Pre-quantized checkpoints are still useful when available. On-the-fly quantization is more flexible, but it requires loading source weights and quantizing them locally. On a Geforce RTX 3090, this process should usually take seconds rather than minutes once the environment is warm.

Vistralis checkpoints:

| Model | Link |
| --- | --- |
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

Additional checkpoints:

| Model | Link |
| --- | --- |
| Chroma1-HD | [Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise) |
| Z-Image-Turbo | [Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise) |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

## Requirements

- Recent ComfyUI
- NVIDIA GPU with useful INT8 throughput
- PyTorch build compatible with your ComfyUI install
- `triton-windows` for the optional fused Triton backend on Windows

Windows note: use the Triton build that matches your PyTorch/CUDA stack. In the tested Comfy Anaconda environment, PyTorch `2.8.0+cu126` imports Triton `3.4.0` from `triton-windows 3.4.0.post21`.

## Recent Development Changes

- Added `runtime_backend` and removed the old visible `use_triton` switch.
- Changed the default backend to `torch_int_mm`.
- Added `small_batch_fallback` with `only_small_layers`, `always`, and `never`.
- Added CUDA-safe padding for `torch._int_mm` tiny-row and non-8-aligned output cases.
- Fixed Triton edge tiles so tail shapes no longer wrap reads with modulo offsets.
- Added diagnostic `triton_legacy_unsafe`.
- Added experimental `prepack_int8_weights`.
- Added `INT8 Lazy Torch Compile`.
- Added stable Dynamic LoRA patch UUIDs to avoid unnecessary recomposition.
- Restored prior INT8 object patches before requantizing.
- Improved cache-reuse logs and runtime diagnostics.
- Reordered node inputs so `bake_loaded_loras` and logging controls are near the bottom.
- Alphabetized model type lists with `auto` first and `none` last where applicable.

## Credits

- dxqb / OneTrainer INT8 work: https://github.com/Nerogar/OneTrainer/pull/1034
- silveroxides / convert_to_quant: https://github.com/silveroxides/convert_to_quant
- silveroxides / ComfyUI-QuantOps: https://github.com/silveroxides/ComfyUI-QuantOps
- newgrit1004 / QuaRot reference code: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
