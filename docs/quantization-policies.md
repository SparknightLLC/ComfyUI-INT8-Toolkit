# Quantization Policies

## Methods And Native Compatibility

| Mode | Weight/activation behavior | Native export |
| --- | --- | --- |
| `int8` | Direct per-row tensorwise W8A8. | Yes |
| `int8_convrot` | Grouped regular Hadamard rotation with native ConvRot W8A8 execution. | Yes |
| `int8_quarot` | Toolkit legacy Hadamard rotation with 128-channel groups. | Toolkit loader required |
| `int8_hadanorm` | Experimental per-channel scaling, Hadamard mixing, and runtime correction. | Toolkit loader required |
| `int4_mixed` | Architecture-aware mixture of native ConvRot W4A4 and W8A8. | Yes |
| `int4_full` | Native ConvRot W4A4 wherever compatible, with INT8 shape fallback. | Yes |

The Toolkit's ConvRot path follows ComfyUI/comfy-kitchen semantics: compatible weights are rotated with grouped regular Hadamard blocks before quantization, and activations receive the matching runtime transform. This is native compatibility work based on the [ConvRot paper](https://arxiv.org/abs/2512.03673) and the broader [QuaRot](https://arxiv.org/abs/2404.00456) lineage.

## Architecture Tiers

Architecture presets contain two policy tiers:

- `keep_float`: quality-sensitive or unsafe layers that remain floating-point in every mode.
- `int4_sensitive`: W4-compatible layers that receive priority within the `int4_mixed_ratio` W8A8 budget.

Krea 2 prioritizes `attn.wo` and `mlp.down`. Anima prioritizes self-attention and cross-attention `output_proj` layers plus `mlp.layer2`, while its final latent projection remains floating-point. On the stock 28-block architectures, ratios of approximately `0.25` for Krea 2 and `0.30` for Anima retain all of those write-back projections in W8A8.

These tiers follow the [Krea 2 implementation](https://github.com/krea-ai/krea-2/blob/main/mmdit.py), [Anima model lineage](https://huggingface.co/circlestone-labs/Anima), and [ComfyUI Cosmos/Anima implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py). They are structurally motivated rather than calibration-proven.

For architectures without a specific sensitive tier, `int4_mixed` distributes its W8A8 budget deterministically across eligible linears. Profiles are nested around the default `0.2`, so reducing the ratio removes W8A8 layers rather than selecting an unrelated profile.

## Presets

Current `model_type` choices include:

- `anima`
- `boogu`
- `chroma`
- `ernie`
- `flux2`
- `flux2_fast_unsafe`
- `hidream o1`
- `ideogram4`
- `krea2`
- `ltx2`
- `qwen`
- `sdxl`
- `wan`
- `z-image`

`auto` is recommended for `Enable Quantization on MODEL`. `flux2_fast_unsafe` is an opt-in, less conservative preset intended for experiments where speed matters more than defensive targeting.

## Mixed-Ratio Interpretation

`int4_mixed_ratio` is the fraction of W4-compatible eligible linears retained in ConvRot INT8:

- `0.0` approaches `int4_full` layer selection.
- `0.2` is the default mixed profile.
- `1.0` retains every compatible eligible linear in INT8.

Architecture-specific patterns receive priority, and any remaining budget is distributed deterministically. A selected fallback layer should not be interpreted as empirically proven sensitive without calibration data.
