# Advanced Usage

This guide covers runtime and workflow details that are useful after the basic Quantization Toolkit workflow is running.

## ComfyUI Core Or The Toolkit?

ComfyUI can load supported native W4A4, W4A8, and W8A8 checkpoints through its stock `Load Diffusion Model` node when the checkpoint contains `.comfy_quant` metadata. Prefer the stock node when no local conversion, Toolkit LoRA handling, runtime tuning, or native-format export is needed.

Use the Toolkit when you need to:

- Quantize a stock-loaded `MODEL` or convert a checkpoint locally.
- Apply architecture-aware mixed W4A4/W8A8 policies or experimental W4A8.
- Apply LoRAs without losing the quantized fast path on every patched layer.
- Tune the INT8 runtime or compile after object patches are active.
- Save a compatible locally quantized model.

## Important Runtime Controls

- `small_batch_fallback=only_small_layers` is the recommended default. It limits floating-point fallback to small Toolkit W8A8 layers where tiny activation batches can make integer matrix multiplication inefficient. Native W4A4 and W4A8 execution ignore this option.
- `runtime_backend=torch_int_mm` is the simple, robust default for ordinary INT8 layers.
- `runtime_backend=triton` can be faster for some GPUs and shapes. It requires a Triton build compatible with the installed PyTorch/CUDA stack.
- `triton_legacy_unsafe` is diagnostic only and can be incorrect on tail shapes.
- `prepack_weights` affects only the Triton INT8 path and consumes an additional INT8 copy of affected weights.
- `int8_convrot` always uses comfy-kitchen's native fused runtime, regardless of `runtime_backend`.
- `w4a8` always uses comfy-kitchen's native asymmetric W4A8 runtime; `runtime_backend` and `prepack_weights` do not alter that path.
- Keep runtime diagnostics disabled for normal benchmarks because console activity can distort results.

## LoRA Ordering

Some LoRA orders temporarily materialize large floating-point tensors and can cause a VRAM spike.

| LoRA method | Before `Enable Quantization on MODEL` | After quantization |
| --- | --- | --- |
| Stock `Load LoRA` | Recommended for stock workflows; bake during quantization. | Avoid on quantized layers unless testing. |
| Quantized `Standard` | Useful for A/B testing. | Compatibility/testing path. |
| Quantized `Stochastic` | Carries deferred quantization-aware patches. | Preferred speed-oriented mode; native W4A8 is dequantized, patched, and requantized once. |
| Quantized `Dynamic` | Usually unnecessary. | Preserves runtime deltas for compatible INT8/W4A4 layers; W4A8 warns and uses Standard fallback. |

Dynamic LoRAs retain their factors and add matrix multiplications to each affected layer. Large or numerous LoRAs therefore consume more VRAM and make the first compiled dispatch more expensive. Compatible factors are consolidated per layer and reused while the model and stack remain unchanged.

On mixed-precision models, Dynamic mode applies unsupported targets through ComfyUI's standard patch path so the complete LoRA remains active. This includes W4A8 until a quality-preserving runtime-delta implementation is available.

## Lazy Torch Compile

Place `Quantized Lazy Torch Compile` after quantization and quantized LoRAs:

```text
quantized MODEL
-> Apply LoRA Stack (Quantized)
-> Quantized Lazy Torch Compile
-> sampler
```

Recommended defaults:

- Keep `compile_transformer_blocks_only=True` unless an architecture requires whole-model compilation.
- Keep `use_guard_filter=True` so volatile per-block transformer options do not cause avoidable retracing.
- Leave `disable_dynamic_vram=True`. Like ComfyUI's stock compile behavior, this demotes only the compile node's MODEL branch rather than disabling Dynamic VRAM globally.
- Use `verbose=True` only while diagnosing preparation, graph growth, timing, or guard failures.

`dynamic_shape_tracing` controls TorchDynamo shape specialization; it is unrelated to ComfyUI Dynamic VRAM.

The Toolkit shares compatible compiled dispatchers between repeated transformer blocks and preserves structural graph families when a LoRA changes on the same base model. A genuinely different operation or required guard can still compile another graph family. Switching architectures clears the prior Toolkit compile output.

Native ConvRot INT4 compiler support varies with ComfyUI and comfy-kitchen versions. The Toolkit probes for an upstream compiler-safe operator and otherwise supplies a temporary opaque boundary. If neither path is available, it returns the model uncompiled with a detailed warning instead of failing during sampling.

Upstream W4A8 dispatch currently enters its native kernel through Python backend selection and DLPack, which FakeTensor cannot trace directly. The Toolkit supplies a temporary opaque `torch.library` custom operator with FakeTensor metadata so the surrounding transformer graph can still compile while eager inference continues to use the ordinary native layout dispatch. If the compatibility operator cannot be registered, the lazy compile node returns an uncompiled MODEL with a detailed warning.

## Model Saving

`Save Quantized Model (DynamicVRAM Safe)` exports Toolkit W8A8 and native W4A4/W4A8 MODEL outputs.

- The default prefix is `output/quantized_models/Quantized_Model`.
- Unquantized inputs are rejected before ComfyUI loads or patches model weights.
- Dynamic LoRAs are runtime-only and cannot be serialized. Use Stochastic mode or bake stock LoRAs before quantization.
- Plain W8A8, ConvRot W8A8, ConvRot W4A4, and asymmetric W4A8 layers receive native `.comfy_quant` metadata.
- QuaRot and HadaNorm require the Toolkit loader because ComfyUI core does not implement their activation transforms.

## Platform Notes

Use the PyTorch and CUDA versions supported by the installed ComfyUI release. Optional Triton packages must match that environment exactly. W4A8 requires an SM80-or-newer GPU (Ampere or newer); comfy-kitchen's optimized NVIDIA backend currently requires a cu130 PyTorch build, while other configurations may use a much slower fallback. Native comfy-kitchen CUDA kernels are strongly preferred for low-bit modes.

Torch Compile is frequently necessary for quantized inference to realize its expected speedup. Restart ComfyUI after failed compiler experiments before comparing performance.
