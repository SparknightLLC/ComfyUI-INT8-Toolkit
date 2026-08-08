import folder_paths
import comfy.utils
import comfy.lora
import comfy.patcher_extension
import logging
import uuid

from .int8_lora_patching import (
    LoRAAdapter,
    _LORA_ADAPTER_AVAILABLE,
    _append_lora_signature,
    _get_key_map,
    _get_supported_quantization_format,
    _model_has_w4a8_modules,
    _resolve_target_module_cached,
    _wrap_static_int8_patches,
    W4A8_QUANTIZATION_FORMAT,
)

_DYNAMIC_LORA_WRAPPER_KEY = "int8_dynamic_lora_sync"


def _is_dynamic_compatible_adapter(adapter):
    if not _LORA_ADAPTER_AVAILABLE or not isinstance(adapter, LoRAAdapter):
        return False

    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        return False

    dora_scale = weights[4] if len(weights) > 4 else None
    reshape = weights[5] if len(weights) > 5 else None
    return dora_scale is None and reshape is None


def _partition_dynamic_patches(model_patcher, patch_dict, module_cache):
    dynamic_patch_dict = {}
    static_patch_dict = {}

    for key, adapter in patch_dict.items():
        if not _is_dynamic_compatible_adapter(adapter):
            static_patch_dict[key] = adapter
            continue

        try:
            target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
            quantization_format = _get_supported_quantization_format(target_module)
        except Exception:
            quantization_format = None

        if quantization_format == W4A8_QUANTIZATION_FORMAT:
            static_patch_dict[key] = adapter
        elif quantization_format is not None:
            dynamic_patch_dict[key] = adapter
        else:
            # Dynamic runtime hooks intentionally cover Toolkit-supported INT8
            # and W4A4 modules only. Preserve LoRA coverage on floating-point and
            # other mixed-precision layers through ComfyUI's ordinary patch path.
            static_patch_dict[key] = adapter

    return dynamic_patch_dict, static_patch_dict


def _warn_if_w4a8_dynamic_fallback(model_patcher):
    if _model_has_w4a8_modules(model_patcher):
        logging.warning(
            "Quantization Toolkit Dynamic LoRA: W4A8 runtime patching is not supported; "
            "affected W4A8 layers will use ComfyUI's Standard LoRA patch path."
        )


def _dynamic_lora_sync_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    if transformer_options is None and len(args) > 5:
        transformer_options = args[5]
    if transformer_options is None:
        transformer_options = {}

    base_model = executor.class_obj
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is not None:
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.sync_from_transformer_options(diffusion_model, transformer_options)

    return executor(*args, **kwargs)

def _ensure_dynamic_sync_wrapper(model_patcher):
    model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, _DYNAMIC_LORA_WRAPPER_KEY)
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        _DYNAMIC_LORA_WRAPPER_KEY,
        _dynamic_lora_sync_wrapper
    )

class INT8DynamicLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "INT8 diffusion model to receive a runtime dynamic LoRA."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file from ComfyUI's loras folder."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "LoRA strength for the diffusion model. Negative values invert the LoRA effect."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()
        _warn_if_w4a8_dynamic_fallback(model_patcher)
        
        # 1. Get Patch Map
        key_map = _get_key_map(model_patcher)

        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        del lora

        dynamic_patch_dict, static_patch_dict = _partition_dynamic_patches(
            model_patcher,
            patch_dict,
            module_cache={},
        )
        del patch_dict

        # 2. Register Global Hook (if not exists)
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        # 3. Add to Dynamic LoRA list in transformer_options
        # This ensures ComfyUI's cloning handles everything and it's non-sticky
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        else:
            model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()
        
        opts = model_patcher.model_options["transformer_options"]
        if "dynamic_loras" not in opts:
            opts["dynamic_loras"] = []
        else:
            # Shallow copy the list to avoid modifying the parent patcher's list
            opts["dynamic_loras"] = opts["dynamic_loras"].copy()
            
        if dynamic_patch_dict:
            opts["dynamic_loras"].append({
                "name": lora_name,
                "strength": strength,
                "patches": dynamic_patch_dict,
                "patch_uuid": uuid.uuid4().hex,
            })

        if static_patch_dict:
            wrapped_static = _wrap_static_int8_patches(model_patcher, static_patch_dict, module_cache={})
            model_patcher.add_patches(wrapped_static, strength)

        _append_lora_signature(model_patcher, "Dynamic", lora_name, strength)
        return (model_patcher,)

class INT8DynamicLoraStack:
    """
    Apply multiple LoRAs in one node for efficiency.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {"model": ("MODEL", {"tooltip": "INT8 diffusion model to receive runtime dynamic LoRAs."})},
            "optional": {},
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list, {"tooltip": f"Optional dynamic LoRA slot {i}. Choose None to leave this slot unused."})
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": f"Strength for dynamic LoRA slot {i}. Ignored when the slot is None or strength is 0."})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "loaders"

    def apply_stack(self, model, **kwargs):
        lora_entries = []
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                lora_entries.append((lora_name, strength))

        return self.apply_loras(model, lora_entries)

    def apply_loras(self, model, lora_entries):

        if not lora_entries:
            return (model,)

        model_patcher = model.clone()
        _warn_if_w4a8_dynamic_fallback(model_patcher)

        key_map = _get_key_map(model_patcher)

        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        else:
            model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()

        opts = model_patcher.model_options["transformer_options"]
        existing_loras = opts.get("dynamic_loras", [])
        opts["dynamic_loras"] = existing_loras.copy()
        module_cache = {}
        dynamic_patch_count = 0
        static_patch_count = 0

        for lora_name, strength in lora_entries:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patch_dict = comfy.lora.load_lora(lora_data, key_map, log_missing=True)
            del lora_data

            dynamic_patch_dict, static_patch_dict = _partition_dynamic_patches(
                model_patcher,
                patch_dict,
                module_cache,
            )
            del patch_dict
            dynamic_patch_count += len(dynamic_patch_dict)
            static_patch_count += len(static_patch_dict)

            if dynamic_patch_dict:
                opts["dynamic_loras"].append({
                    "name": lora_name,
                    "strength": strength,
                    "patches": dynamic_patch_dict,
                    "patch_uuid": uuid.uuid4().hex,
                })

            if static_patch_dict:
                wrapped_static = _wrap_static_int8_patches(
                    model_patcher,
                    static_patch_dict,
                    module_cache=module_cache,
                )
                model_patcher.add_patches(wrapped_static, strength)

            _append_lora_signature(model_patcher, "Dynamic", lora_name, strength)

        logging.info(
            f"Quantization Toolkit LoRA stack (Dynamic): loaded {len(lora_entries)} LoRAs in a single pass "
            f"(runtime patch targets={dynamic_patch_count}, standard fallback targets={static_patch_count})."
        )
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8DynamicLoraLoader": INT8DynamicLoraLoader,
    "INT8DynamicLoraStack": INT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8DynamicLoraLoader": "Load LoRA (Quantized, Dynamic)",
    "INT8DynamicLoraStack": "Load LoRA Stack (Quantized, Dynamic)",
}
