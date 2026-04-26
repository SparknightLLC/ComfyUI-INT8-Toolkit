import torch
import folder_paths

from .int8_quant import (
    Int8TensorwiseOps,
    INT8_BACKEND_CHOICES,
    DEFAULT_INT8_BACKEND,
    INT8_BACKEND_TRITON,
    INT8_BACKEND_TRITON_LEGACY_UNSAFE,
    OUTLIER_METHOD_CHOICES,
    OUTLIER_METHOD_NONE,
    SMALL_BATCH_FALLBACK_CHOICES,
    DEFAULT_SMALL_BATCH_FALLBACK,
)


MODEL_TYPE_CHOICES = ["anima", "chroma", "ernie", "flux2", "ltx2", "qwen", "sdxl", "wan", "z-image"]
DEFAULT_OUTLIER_METHOD = OUTLIER_METHOD_NONE


def get_model_type_exclusions(model_type):
    if model_type == "flux2":
        return [
            "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
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
            "time", "x_embedder", "adaLN", "final", "text_proj", "norm", "layers.0.", "layers.35",
        ]
    if model_type == "anima":
        return [
            "embed", "llm", "blocks.0.", "blocks.1.", "blocks.2.",
        ]
    if model_type == "sdxl":
        return [
            "time_embed", "label_emb", "emb_layers", "proj_in", "proj_out",
        ]
    if model_type == "wan":
        return [
            "patch_embedding", "text_embedding", "time_embedding", "time_projection", "head",
            "img_emb",
        ]
    if model_type == "ltx2":
        return [
            "adaln_single", "audio_adaln_single", "audio_caption_projection", "audio_patchify_proj", "audio_proj_out",
            "audio_scale_shift_table", "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single", "av_ca_v2a_gate_adaln_single",
            "av_ca_video_scale_shift_adaln_single", "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
        ]
    return []


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
                "model_type": (MODEL_TYPE_CHOICES, {"tooltip": "Architecture preset used to skip layers that are usually quality-sensitive or unsafe to quantize."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize eligible float or FP8 weights to INT8 during loading. Leave off for already-quantized INT8 checkpoints."}),
                "outlier_method": (OUTLIER_METHOD_CHOICES, {"default": DEFAULT_OUTLIER_METHOD, "tooltip": "Outlier mitigation to apply during on-the-fly INT8 quantization. QuaRot uses a Hadamard rotation. HadaNorm adds per-channel scaling, Hadamard mixing, and a runtime correction term for compatible layers."}),
                "small_batch_fallback": (SMALL_BATCH_FALLBACK_CHOICES, {"default": DEFAULT_SMALL_BATCH_FALLBACK, "tooltip": "Controls the fp16/bf16 fallback for very small activation batches. only_small_layers is the default and limits fallback to layers with out_features * in_features <= INT8_SMALL_LAYER_MAX_PARAMS, default 1,000,000; always can help tiny row counts but often slows larger layers by dequantizing full weights; never forces the INT8 backend."}),
                "runtime_backend": (INT8_BACKEND_CHOICES, {"default": DEFAULT_INT8_BACKEND, "tooltip": "Backend for INT8 linear layers. torch_int_mm is the default and uses PyTorch torch._int_mm with tiny-row padding for CUDA compatibility; triton uses this extension's fused Triton kernels and may be faster on some model shapes; triton_legacy_unsafe reproduces the old upstream edge-tile behavior for diagnostics only and may be incorrect on tail shapes."}),
                "prepack_int8_weights": ("BOOLEAN", {"default": False, "tooltip": "Experimental: keep an extra transposed INT8 weight buffer for Triton so output columns are read contiguously. May improve speed but adds roughly one extra INT8 copy of each quantized weight."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load INT8 tensorwise quantized models with fast torch._int_mm inference."

    def load_unet(
        self,
        unet_name,
        weight_dtype,
        model_type,
        on_the_fly_quantization,
        outlier_method=DEFAULT_OUTLIER_METHOD,
        small_batch_fallback=DEFAULT_SMALL_BATCH_FALLBACK,
        runtime_backend=DEFAULT_INT8_BACKEND,
        prepack_int8_weights=False,
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
        Int8TensorwiseOps.excluded_names = []
        Int8TensorwiseOps.dynamic_quantize = on_the_fly_quantization
        Int8TensorwiseOps.outlier_method = outlier_method if on_the_fly_quantization else DEFAULT_OUTLIER_METHOD
        Int8TensorwiseOps.use_triton = True
        Int8TensorwiseOps.small_batch_fallback_mode = small_batch_fallback
        Int8TensorwiseOps.runtime_backend = runtime_backend
        Int8TensorwiseOps.runtime_uses_triton = runtime_backend in (INT8_BACKEND_TRITON, INT8_BACKEND_TRITON_LEGACY_UNSAFE)
        Int8TensorwiseOps.runtime_uses_legacy_triton = runtime_backend == INT8_BACKEND_TRITON_LEGACY_UNSAFE
        Int8TensorwiseOps.prepack_int8_weights = bool(prepack_int8_weights)
        Int8TensorwiseOps._is_prequantized = False
        Int8TensorwiseOps.reset_otf_progress()
        
        # Check explicit model_type for exclusions
        Int8TensorwiseOps.excluded_names = get_model_type_exclusions(model_type)

        # Load model directly - Int8TensorwiseOps handles int8 weights natively
        model = load_diffusion_model(unet_path, model_options=model_options)

        if on_the_fly_quantization:
            Int8TensorwiseOps.summarize_otf_progress()

        return (model,)

