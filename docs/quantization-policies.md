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
| `w4a8` | Experimental asymmetric 4-bit weights with ConvRot INT8 activations; incompatible shapes use W8A8. | Yes |

The Toolkit's ConvRot path follows ComfyUI/comfy-kitchen semantics: compatible weights are rotated with grouped regular Hadamard blocks before quantization, and activations receive the matching runtime transform. This is native compatibility work based on the [ConvRot paper](https://arxiv.org/abs/2512.03673) and the broader [QuaRot](https://arxiv.org/abs/2404.00456) lineage.

`w4a8` is fundamentally different from `int4_mixed`. The mixed mode assigns whole eligible layers to either W4A4 or W8A8, and `int4_mixed_ratio` controls how many remain W8A8. W4A8 instead stores each compatible target layer's weights at 4 bits while quantizing its runtime activations to INT8. The mixed-ratio and `int4_sensitive` controls do not change W4A8; `keep_float` exclusions still apply.

## Architecture Tiers

Architecture presets contain two policy tiers:

- `keep_float`: quality-sensitive or unsafe layers that remain floating-point in every mode.
- `int4_sensitive`: W4-compatible layers that receive priority within the `int4_mixed_ratio` W8A8 budget.

Krea 2 prioritizes `attn.wo` and `mlp.down`. Anima prioritizes self-attention and cross-attention `output_proj` layers plus `mlp.layer2`, while its final latent projection remains floating-point. MiniMax H3 preliminarily prioritizes `attn.out_proj` and `mlp.fc2`, while keeping its projections, embeddings, token refiner, AdaLN projections, and final layer floating-point during local conversion. On the stock 28-block architectures, ratios of approximately `0.25` for Krea 2 and `0.30` for Anima retain all of those write-back projections in W8A8.

These tiers follow the [Krea 2 implementation](https://github.com/krea-ai/krea-2/blob/main/mmdit.py), [Anima model lineage](https://huggingface.co/circlestone-labs/Anima), [ComfyUI Cosmos/Anima implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py), [official MiniMax H3 repack](https://huggingface.co/Comfy-Org/MiniMax-H3), and [ComfyUI MiniMax H3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py). They are structurally motivated rather than calibration-proven.

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
- `minimax_h3`
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
