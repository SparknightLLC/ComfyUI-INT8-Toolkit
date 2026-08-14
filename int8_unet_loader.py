import logging

import torch
import folder_paths

from .int8_quant import (
    Int8TensorwiseOps,
    INT8_BACKEND_CHOICES,
    DEFAULT_INT8_BACKEND,
    INT8_BACKEND_TRITON,
    INT8_BACKEND_TRITON_LEGACY_UNSAFE,
    QUANTIZATION_MODE_CHOICES,
    QUANTIZATION_MODE_INT8,
    QUANTIZATION_MODE_INT4_MIXED,
    QUANTIZATION_MODE_W4A8,
    normalize_quantization_mode,
    quantization_mode_outlier_method,
    quantization_mode_is_int4,
    native_int4_available,
    native_w4a8_available,
    SMALL_BATCH_FALLBACK_CHOICES,
    DEFAULT_SMALL_BATCH_FALLBACK,
)
from .quantization_policy import DEFAULT_INT4_MIXED_RATIO
from .quantization_policy import normalize_int4_mixed_ratio


MODEL_TYPE_FLUX2 = "flux2"
MODEL_TYPE_FLUX2_FAST_UNSAFE = "flux2_fast_unsafe"
MODEL_TYPE_BOOGU = "boogu"
MODEL_TYPE_HIDREAM_O1 = "hidream o1"
MODEL_TYPE_IDEOGRAM4 = "ideogram4"
MODEL_TYPE_KREA2 = "krea2"
MODEL_TYPE_MINIMAX_H3 = "minimax_h3"
MODEL_TYPE_CHOICES = [
    "anima",
    MODEL_TYPE_BOOGU,
    "chroma",
    "ernie",
    MODEL_TYPE_FLUX2,
    MODEL_TYPE_FLUX2_FAST_UNSAFE,
    MODEL_TYPE_HIDREAM_O1,
    MODEL_TYPE_IDEOGRAM4,
    MODEL_TYPE_KREA2,
    MODEL_TYPE_MINIMAX_H3,
    "ltx2",
    "qwen",
    "sdxl",
    "wan",
    "z-image",
]
DEFAULT_QUANTIZATION_MODE = QUANTIZATION_MODE_INT8


def _get_model_type_keep_float(model_type):
    if model_type == MODEL_TYPE_FLUX2:
        return [
            "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
            "double_stream_modulation_img", "double_stream_modulation_txt",
            "single_stream_modulation",
        ]
    if model_type == MODEL_TYPE_FLUX2_FAST_UNSAFE:
        return [
            "img_in", "time_in", "guidance_in", "txt_in",
            "double_stream_modulation_img", "double_stream_modulation_txt",
            "single_stream_modulation",
        ]
    if model_type == "z-image":
        return [
            "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token", "context_refiner",
            "final_layer", "noise_refiner", "adaLN",
            "x_pad_token", "layers.0.",
        ]
    if model_type == "chroma":
        return [
            "distilled_guidance_layer", "final_layer", "img_in", "txt_in", "nerf_image_embedder",
            "nerf_blocks", "nerf_final_layer_conv", "__x0__", "nerf_final_layer_conv",
        ]
    if model_type == "qwen":
        return [
            "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
        ]
    if model_type == "ernie":
        return [
            "time", "x_embedder", "text_proj", "adaLN",
        ]
    if model_type == "anima":
        return [
            "embed", "llm", "adaln", "final_layer",
        ]
    if model_type == MODEL_TYPE_BOOGU:
        return [
            "embed", "refine", "norm_out",
        ]
    if model_type == MODEL_TYPE_HIDREAM_O1:
        return [
            "embed", "language_model.layers.35.mlp",
        ]
    if model_type == MODEL_TYPE_IDEOGRAM4:
        return [
            "embed_image_indicator", "t_embedding",
        ]
    if model_type == MODEL_TYPE_KREA2:
        return [
            "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
        ]
    if model_type == MODEL_TYPE_MINIMAX_H3:
        return [
            "video_patch_proj", "audio_patch_proj", "condition_proj", "time_embedder",
            "token_refiner", "adaln_proj", "final_layer",
        ]
    if model_type == "sdxl":
        return [
            "time_embed", "label_emb", "emb_layers", "proj_in", "proj_out",
        ]
    if model_type == "wan":
        return [
            "patch_embedding", "text_embedding", "time_embedding", "time_projection", "head",
            "img_emb", "face_adapter", "face_encoder", "motion_encoder", "pose_patch_embedding",
        ]
    if model_type == "ltx2":
        return [
            "adaln", "embedding", "patchify", "to_gate_logits", "proj_out",
            "model.audio", "model.video", "model.av", "model.patch", "model.proj", "shift",
            "adaln_single", "audio_adaln_single", "audio_caption_projection", "audio_patchify_proj", "audio_proj_out",
            "audio_scale_shift_table", "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single", "av_ca_v2a_gate_adaln_single",
            "av_ca_video_scale_shift_adaln_single", "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
        ]
    return []


def _get_model_type_int4_sensitive(model_type):
    if model_type == "anima":
        return [
            "self_attn.output_proj", "cross_attn.output_proj", "mlp.layer2",
        ]
    if model_type == MODEL_TYPE_KREA2:
        return [
            "attn.wo", "mlp.down",
        ]
    if model_type == MODEL_TYPE_MINIMAX_H3:
        return [
            "attn.out_proj", "mlp.fc2",
        ]
    return []


def get_model_type_quantization_preset(model_type):
    return {
        "keep_float": tuple(_get_model_type_keep_float(model_type)),
        "int4_sensitive": tuple(_get_model_type_int4_sensitive(model_type)),
    }


def _read_safetensors_metadata(path):
    if not isinstance(path, str) or not path.lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            return dict(metadata) if isinstance(metadata, dict) else None
    except Exception:
        return None


def _stash_safetensors_metadata(model, metadata):
    if not isinstance(metadata, dict):
        return

    metadata = dict(metadata)
    try:
        model._safetensors_metadata = metadata
    except Exception:
        pass

    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return

    try:
        inner_model._int8_source_metadata = metadata
    except Exception:
        pass


class UNetLoaderINTW8A8:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    Inference uses fast torch._int_mm for blazing speed. (insert rocket emoji, fire emoji to taste)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "Diffusion model checkpoint to load from ComfyUI's diffusion_models folder."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp16", "bf16"], {"tooltip": "Requested source weight dtype passed to ComfyUI during model construction. INT8 checkpoints still load as INT8 when weight_scale tensors are present."}),
                "model_type": (MODEL_TYPE_CHOICES, {"tooltip": "Architecture preset. Known quality-sensitive or unsafe layers remain floating-point in every mode. flux2_fast_unsafe is opt-in and less conservative."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize eligible float or FP8 weights using the selected mode during loading. Leave off for checkpoints that already contain native quantization metadata."}),
                "quantization_mode": (QUANTIZATION_MODE_CHOICES, {"default": DEFAULT_QUANTIZATION_MODE, "tooltip": "Quantization mode. int4_mixed and int4_full use W4A4; w4a8 stores 4-bit weights while retaining ConvRot INT8 activations. All low-bit modes preserve keep-float layers."}),
                "int4_mixed_ratio": ("FLOAT", {"default": DEFAULT_INT4_MIXED_RATIO, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of W4-compatible eligible linears kept in ConvRot INT8 when using int4_mixed. Architecture-specific patterns are prioritized; the remaining budget is distributed deterministically across the model. 0 matches int4_full layer selection and 1 keeps all compatible linears in INT8."}),
                "small_batch_fallback": (SMALL_BATCH_FALLBACK_CHOICES, {"default": DEFAULT_SMALL_BATCH_FALLBACK, "tooltip": "Controls the fp16/bf16 fallback for very small activation batches on Toolkit W8A8 layers. It does not alter native W4A4 or W4A8 execution. only_small_layers is the default; always can help tiny row counts but often slows larger layers; never forces the INT8 backend."}),
                "runtime_backend": (INT8_BACKEND_CHOICES, {"default": DEFAULT_INT8_BACKEND, "tooltip": "Backend for non-ConvRot INT8 linear layers. int8_convrot always uses Comfy-Kitchen's native fused runtime. torch_int_mm is the default for other INT8 modes; triton may be faster on some shapes; triton_legacy_unsafe is diagnostic only and may be incorrect on tail shapes."}),
                "prepack_weights": ("BOOLEAN", {"default": False, "tooltip": "Experimental runtime weight prepacking. This currently applies only to Triton INT8 layers, where it keeps an extra transposed weight buffer so output columns are read contiguously. It may improve speed but adds roughly one extra INT8 copy of each affected weight."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load native W8A8, W4A4, or experimental W4A8 checkpoints, or quantize float and FP8 diffusion models on the fly."

    def load_unet(
        self,
        unet_name,
        weight_dtype,
        model_type,
        on_the_fly_quantization,
        quantization_mode=DEFAULT_QUANTIZATION_MODE,
        int4_mixed_ratio=DEFAULT_INT4_MIXED_RATIO,
        small_batch_fallback=DEFAULT_SMALL_BATCH_FALLBACK,
        runtime_backend=DEFAULT_INT8_BACKEND,
        prepack_weights=False,
    ):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Use Int8TensorwiseOps for proper direct int8 loading
        model_options = {"custom_operations": Int8TensorwiseOps}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        
        # We need to peek at the model type to set exclusions for Flux
        # ComfyUI loads metadata before the full model
        from comfy.sd import load_diffusion_model
        
        # Set quantization flags for this load
        if runtime_backend not in INT8_BACKEND_CHOICES:
            runtime_backend = DEFAULT_INT8_BACKEND
        Int8TensorwiseOps.keep_float_names = []
        Int8TensorwiseOps.int4_sensitive_names = []
        Int8TensorwiseOps.dynamic_quantize = on_the_fly_quantization
        quantization_mode = normalize_quantization_mode(quantization_mode)
        if on_the_fly_quantization and quantization_mode == QUANTIZATION_MODE_W4A8 and not native_w4a8_available():
            raise RuntimeError("W4A8 quantization requires ComfyUI 0.32.0 or newer with a compatible comfy-kitchen installation")
        if on_the_fly_quantization and quantization_mode != QUANTIZATION_MODE_W4A8 and quantization_mode_is_int4(quantization_mode) and not native_int4_available():
            raise RuntimeError("INT4 quantization requires a recent ComfyUI and comfy-kitchen with ConvRot W4A4 support")
        Int8TensorwiseOps.quantization_mode = quantization_mode if on_the_fly_quantization else QUANTIZATION_MODE_INT8
        Int8TensorwiseOps.int4_mixed_ratio = normalize_int4_mixed_ratio(int4_mixed_ratio)
        Int8TensorwiseOps.outlier_method = quantization_mode_outlier_method(Int8TensorwiseOps.quantization_mode)
        Int8TensorwiseOps.use_triton = True
        Int8TensorwiseOps.small_batch_fallback_mode = small_batch_fallback
        Int8TensorwiseOps.runtime_backend = runtime_backend
        Int8TensorwiseOps.runtime_uses_triton = runtime_backend in (INT8_BACKEND_TRITON, INT8_BACKEND_TRITON_LEGACY_UNSAFE)
        Int8TensorwiseOps.runtime_uses_legacy_triton = runtime_backend == INT8_BACKEND_TRITON_LEGACY_UNSAFE
        Int8TensorwiseOps.prepack_int8_weights = bool(prepack_weights)
        Int8TensorwiseOps._is_prequantized = False
        Int8TensorwiseOps.reset_otf_progress()
        
        quantization_preset = get_model_type_quantization_preset(model_type)
        Int8TensorwiseOps.keep_float_names = list(quantization_preset["keep_float"])
        Int8TensorwiseOps.int4_sensitive_names = list(quantization_preset["int4_sensitive"])
        if on_the_fly_quantization and model_type == MODEL_TYPE_FLUX2_FAST_UNSAFE:
            logging.info("Quantization Toolkit loader: flux2_fast_unsafe selected; using the less conservative Flux2 keep-float preset.")

        # Load model directly - Int8TensorwiseOps handles int8 weights natively
        model = load_diffusion_model(unet_path, model_options=model_options)
        metadata = _read_safetensors_metadata(unet_path)
        if metadata is not None:
            _stash_safetensors_metadata(model, metadata)

        if on_the_fly_quantization:
            Int8TensorwiseOps.summarize_otf_progress()

        return (model,)

