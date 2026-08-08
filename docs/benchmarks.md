# Preliminary Benchmarks

These results are early observations from one machine and one model. They are intended to describe practical tradeoffs, not establish a general ranking. Quantization sensitivity varies by architecture, prompt, sampler, LoRA stack, and hardware.

## Krea2 on RTX 3090

Recorded 8 August 2026 with:

- NVIDIA GeForce RTX 3090 with 24 GB VRAM.
- Krea2 at 704 × 1024 with six sampling steps.
- Quantized Lazy Torch Compile enabled.
- The same two Stochastic LoRAs active at strength 1.0; both modes reported 224 quantized layers patched.
- ComfyUI 0.31.0 and comfy-kitchen 0.2.28.
- Identical workflow settings between quantization modes.
- Warm sampling runs after conversion and initial compilation.

| Observation | `w4a8` | `int8_convrot` | Difference |
| --- | ---: | ---: | ---: |
| Warm sampling throughput | Logged samples 1.37–1.41 it/s | Logged samples 1.64–1.66 it/s | W4A8 throughput was about 15% lower using the retained samples |
| Loaded Krea2 model weights reported by ComfyUI | 7,795.33 MB | 12,866.82 MB | W4A8 used 5,071.49 MB, or 39.4%, less |

The weight figure is not a true peak-VRAM measurement. ComfyUI reports the amount loaded for the model, while total process peak also includes activations, compiler workspaces, CUDA graphs, the text encoder, VAE, LoRAs, and allocator reservations. Post-sampling `usable` readings differed by approximately 5.1 GB in the same direction, which supports the model-footprint result but does not replace instrumented peak-memory measurement.

On this Ampere GPU, W4A8 therefore behaved primarily as a memory optimization rather than a speed optimization. Its smaller weights require additional low-bit decoding and use a different native kernel path; the Toolkit compiler shim allows the surrounding graph to compile but deliberately leaves that kernel opaque.

## Preliminary Visual Observations

Matched-prompt comparisons showed moderate composition changes. For example, a prompt that commonly produced a front-facing portrait under INT8 ConvRot sometimes produced a side view under W4A8. This is not sufficient evidence that either composition is worse: small numerical changes can alter the denoising trajectory and produce a different valid image.

Fine texture and fidelity on Krea2 appeared subjectively 10–15% lower with W4A8. Treat that estimate as model- and observer-specific rather than a measured quality score. Krea2 is a large, texture-rich photographic model and may expose four-bit weight error more readily than a more stylized model such as Anima, but Anima requires a separate controlled comparison before drawing that conclusion.

The active Stochastic LoRA stack is also a possible contributor. In W4A8 mode, Stochastic patching dequantizes the weights, applies the LoRA deltas, and requantizes once. Future comparisons should include both an unpatched baseline and a Standard-versus-Stochastic LoRA comparison.

## Improving The Benchmark

Future results should use multiple fixed seeds and prompts, report the median of at least three warm runs, and separate conversion, compilation, and sampling time. True peak VRAM should be collected around the sampling interval using PyTorch peak allocator statistics, with unrelated GPU applications closed where practical.
