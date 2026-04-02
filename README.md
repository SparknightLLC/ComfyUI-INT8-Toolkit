# ComfyUI-INT8-Fast-Fork

This fork tracks [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) but carries additional fixes and workflow-focused changes that are not all in upstream main.

Tested against various Klein 9b and Z-Image models. Your mileage with other architectures may vary.

## What Is Different In This Fork

### 1) Unified LoRA Nodes With Mode Switch

The separate node pairs for stochastic vs dynamic LoRA have been replaced with:

- `Load LoRA INT8`
- `Load LoRA Stack INT8`

Both nodes now expose a top-level `mode` chooser.

This lets you switch LoRA application mode without rebuilding your node graph.

### 2) LoHA/LoKR Support Uses a Different Strategy Than Upstream Main

Upstream main now includes LoHA/LoKR support in the INT8 stochastic path (commit `55762d1`) by reconstructing LoRA deltas directly for multiple adapter payload formats in `int8_quant.py`.

This fork takes a different approach:

- Keeps a fast INT8 path for plain LoRA adapters.
- Uses guarded adapter routing and safe wrapper fallback for non-plain adapters (including LoHA/LoKR-style payloads) instead of assuming one direct reconstruction path.
- In dynamic mode, splits dynamic-compatible vs non-compatible adapters and routes unsupported entries to static-safe patching.

This is intended to favor resilience across ComfyUI adapter/type changes, especially in mixed or edge-case workflows.

### 3) Merged-QKV Per-Row Scale Fixes

INT8 LoRA patching now handles offset/narrowed tensor slices correctly when per-row scales come from merged QKV weights. This prevents shape-mismatch failures during patch application.

### 4) Device-Safe INT8 Scale Handling

Scale tensors are aligned to the active compute device before dequantization and stochastic rounding. This prevents CPU/GPU mismatch crashes in mixed offload setups.

### 5) Dynamic LoRA Runtime Performance Fix

Dynamic LoRA A/B tensors are cached on the active device during forward passes to avoid repeated CPU->GPU transfers each step.

### 6) Per-Row INT8 Runtime Support

Per-row INT8 weight-scale inference support (upstream PR #24 scope) is integrated with fallback handling in this fork.

### 7) On-The-Fly Quantization Upgrades

`Load Diffusion Model INT8 (W8A8)` exposes `on_the_fly_quantization` with these fork-specific behaviors:

- Converts eligible float weights to INT8 with **per-row weight scales**.
- Accepts FP8 source tensors by converting FP8 -> FP16 compute dtype before INT8 quantization.
- Auto-sets layer runtime to the per-row INT8 forward path when `[rows, 1]` scales are produced.

### 8) Triton Kernel Strategy Controls

This fork includes runtime kernel configuration knobs for fixed vs autotuned Triton launch behavior:

You can set these via environment variables, or use the `INT8 Kernel Config` node to:

- Apply manual fixed-kernel values in-graph.
- Optionally run an on-demand microbench and auto-select the best fixed config.
- Print the recommended env var values to console for future reuse.

- `INT8_TRITON_AUTOTUNE`
- `INT8_TRITON_BLOCK_M`
- `INT8_TRITON_BLOCK_N`
- `INT8_TRITON_BLOCK_K`
- `INT8_TRITON_GROUP_SIZE_M`
- `INT8_TRITON_NUM_WARPS`
- `INT8_TRITON_NUM_STAGES`

Additional runtime toggles:

- `INT8_SMALL_BATCH_FALLBACK_MAX_ROWS`
- `INT8_SMALL_BATCH_FALLBACK_MIN_ROWS`
- `INT8_SMALL_BATCH_FALLBACK_ADAPTIVE`
- `INT8_DYNAMIC_LORA_DEBUG`
- `INT8_DYNAMIC_LORA_BATCH`
- `INT8_DYNAMIC_LORA_BATCH_MAX_RANK`

## Node Summary

### Load Diffusion Model INT8 (W8A8)

Loads INT8 diffusion models using `Int8TensorwiseOps` and model-type-specific exclusion presets.

When `on_the_fly_quantization` is enabled, this fork quantizes eligible layers to INT8 using per-row weight scales.

### Load LoRA INT8

Loads one LoRA with selectable `mode`:

- `Stochastic`: INT8-space patching with stochastic rounding.
- `Dynamic`: runtime dynamic LoRA composition hook path.

### Load LoRA Stack

Loads up to 10 LoRAs with the same selectable `mode` behavior as above.

Node display name: `Load LoRA Stack INT8`.

### INT8 Kernel Config

Utility node that applies a fixed Triton kernel config at runtime.

Optional microbench mode runs on demand, selects the fastest fixed config from built-in candidates plus your manual values, and prints env vars to console so you can persist them.

## ModelSave Round-Trip

If you quantize with `on_the_fly_quantization` and then save with ComfyUI `ModelSave`, you can load it back with `Load Diffusion Model INT8 (W8A8)` without re-quantizing, as long as the saved checkpoint includes INT8 `weight` tensors and their `weight_scale` tensors.

## Workflow Compatibility Note

If your old workflows used the separate dynamic/stochastic node IDs, replace them with:

- `Load LoRA INT8`
- `Load LoRA Stack INT8`

Then choose mode from the new `mode` selector.

## Upstream Context

- Earlier fork divergence context: PR #20  
  https://github.com/BobJohnson24/ComfyUI-INT8-Fast/pull/20
- Per-row quant support context: PR #24  
  https://github.com/BobJohnson24/ComfyUI-INT8-Fast/pull/24
- Upstream LoHA/LoKR support commit:  
  https://github.com/BobJohnson24/ComfyUI-INT8-Fast/commit/55762d16e84c8ebf91787637c39f77c90861c3b9
