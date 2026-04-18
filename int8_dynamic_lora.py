import folder_paths
import comfy.utils
import comfy.lora
import comfy.patcher_extension
import logging
import torch

try:
    from comfy.weight_adapter.lora import LoRAAdapter
    _LORA_ADAPTER_AVAILABLE = True
except Exception:
    LoRAAdapter = None
    _LORA_ADAPTER_AVAILABLE = False

try:
    from comfy.weight_adapter.base import WeightAdapterBase
    _WEIGHT_ADAPTER_BASE_AVAILABLE = True
except Exception:
    WeightAdapterBase = None
    _WEIGHT_ADAPTER_BASE_AVAILABLE = False

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


def _is_weight_adapter(adapter):
    return _WEIGHT_ADAPTER_BASE_AVAILABLE and isinstance(adapter, WeightAdapterBase)


def _resolve_target_module(model_patcher, key, module_cache=None):
    layer_name = key[0] if isinstance(key, tuple) else key
    if isinstance(layer_name, str) and layer_name.endswith(".weight"):
        layer_name = layer_name[:-7]

    if module_cache is not None and layer_name in module_cache:
        return module_cache[layer_name]

    try:
        target_module = model_patcher.get_model_object(layer_name)
        if module_cache is not None:
            module_cache[layer_name] = target_module
        return target_module
    except Exception:
        pass

    parts = layer_name.split(".")
    target_module = model_patcher.model.diffusion_model
    for part in parts[1:] if parts and parts[0] == "diffusion_model" else parts:
        if part.isdigit():
            target_module = target_module[int(part)]
        else:
            target_module = getattr(target_module, part)

    if module_cache is not None:
        module_cache[layer_name] = target_module
    return target_module


def _wrap_static_int8_patches(model_patcher, patch_dict, seed=318008, module_cache=None):
    from .int8_quant import INT8LoRAPatchAdapter, INT8WeightPatchAdapter

    if module_cache is None:
        module_cache = {}

    wrapped_patch_dict = {}
    for key, adapter in patch_dict.items():
        if not _is_weight_adapter(adapter):
            wrapped_patch_dict[key] = adapter
            continue

        try:
            target_module = _resolve_target_module(model_patcher, key, module_cache)
            if not (hasattr(target_module, "_is_quantized") and target_module._is_quantized):
                wrapped_patch_dict[key] = adapter
                continue

            w_scale = target_module.weight_scale
            if isinstance(w_scale, torch.Tensor):
                w_scale = w_scale.item() if w_scale.numel() == 1 else w_scale
            outlier_method = getattr(target_module, "_outlier_method", None)
            hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)

            if _LORA_ADAPTER_AVAILABLE and isinstance(adapter, LoRAAdapter):
                wrapped_patch_dict[key] = INT8LoRAPatchAdapter(
                    adapter.loaded_keys,
                    adapter.weights,
                    w_scale,
                    seed=seed,
                    outlier_method=outlier_method,
                    hadanorm_sigma=hadanorm_sigma,
                )
            else:
                wrapped_patch_dict[key] = INT8WeightPatchAdapter(
                    adapter,
                    w_scale,
                    seed=seed,
                    outlier_method=outlier_method,
                    hadanorm_sigma=hadanorm_sigma,
                )
        except Exception:
            wrapped_patch_dict[key] = adapter

    return wrapped_patch_dict

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
        
        # 1. Get Patch Map
        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        del lora

        dynamic_patch_dict = {}
        static_patch_dict = {}
        for key, adapter in patch_dict.items():
            if _is_dynamic_compatible_adapter(adapter):
                dynamic_patch_dict[key] = adapter
            else:
                static_patch_dict[key] = adapter
        del patch_dict

        # 2. Register Global Hook (if not exists)
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        # 3. Add to Dynamic LoRA list in transformer_options
        # This ensures ComfyUI's cloning handles everything and it's non-sticky
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
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
                "patches": dynamic_patch_dict
            })

        if static_patch_dict:
            wrapped_static = _wrap_static_int8_patches(model_patcher, static_patch_dict, module_cache={})
            model_patcher.add_patches(wrapped_static, strength)

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

        if not lora_entries:
            return (model,)

        model_patcher = model.clone()

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)
        _ensure_dynamic_sync_wrapper(model_patcher)

        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}

        opts = model_patcher.model_options["transformer_options"]
        existing_loras = opts.get("dynamic_loras", [])
        opts["dynamic_loras"] = existing_loras.copy()
        module_cache = {}

        for lora_name, strength in lora_entries:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patch_dict = comfy.lora.load_lora(lora_data, key_map, log_missing=True)
            del lora_data

            dynamic_patch_dict = {}
            static_patch_dict = {}
            for key, adapter in patch_dict.items():
                if _is_dynamic_compatible_adapter(adapter):
                    dynamic_patch_dict[key] = adapter
                else:
                    static_patch_dict[key] = adapter
            del patch_dict

            if dynamic_patch_dict:
                opts["dynamic_loras"].append({
                    "name": lora_name,
                    "strength": strength,
                    "patches": dynamic_patch_dict
                })

            if static_patch_dict:
                wrapped_static = _wrap_static_int8_patches(
                    model_patcher,
                    static_patch_dict,
                    module_cache=module_cache,
                )
                model_patcher.add_patches(wrapped_static, strength)

        logging.info(f"INT8 Dynamic LoRA Stack: Loaded {len(lora_entries)} LoRAs in a single pass.")
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8DynamicLoraLoader": INT8DynamicLoraLoader,
    "INT8DynamicLoraStack": INT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8DynamicLoraLoader": "Load LoRA INT8 (Dynamic)",
    "INT8DynamicLoraStack": "INT8 LoRA Stack (Dynamic)",
}
