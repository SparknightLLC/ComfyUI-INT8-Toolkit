import logging

import comfy.lora
import comfy.utils
import folder_paths
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


def _is_plain_lora_adapter(adapter):
	return _LORA_ADAPTER_AVAILABLE and isinstance(adapter, LoRAAdapter)


def _is_weight_adapter(adapter):
	return _WEIGHT_ADAPTER_BASE_AVAILABLE and isinstance(adapter, WeightAdapterBase)


def _extract_layer_name(key):
	layer_name = key[0] if isinstance(key, tuple) else key
	if isinstance(layer_name, str) and layer_name.endswith(".weight"):
		layer_name = layer_name[:-7]
	return layer_name


def _resolve_target_module_cached(model_patcher, key, module_cache):
	layer_name = _extract_layer_name(key)
	if not isinstance(layer_name, str):
		raise TypeError("Unsupported key type for layer resolution")
	if layer_name in module_cache:
		return module_cache[layer_name]

	try:
		target_module = model_patcher.get_model_object(layer_name)
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

	module_cache[layer_name] = target_module
	return target_module


def _get_key_map(model_patcher):
	key_map = {}
	if model_patcher.model.model_type.name != "ModelType.CLIP":
		key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
	return key_map


def _get_weight_scale_for_module(target_module):
	weight_scale = target_module.weight_scale
	if isinstance(weight_scale, torch.Tensor):
		return weight_scale.item() if weight_scale.numel() == 1 else weight_scale
	return weight_scale


def _upgrade_patch_dict_for_int8(model_patcher, patch_dict, seed, module_cache):
	from .int8_quant import INT8LoRAPatchAdapter, INT8WeightPatchAdapter

	final_patch_dict = {}
	applied_count = 0

	for key, adapter in patch_dict.items():
		try:
			target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
			is_quantized = hasattr(target_module, "_is_quantized") and target_module._is_quantized

			if is_quantized and _is_weight_adapter(adapter):
				weight_scale = _get_weight_scale_for_module(target_module)
				outlier_method = getattr(target_module, "_outlier_method", None)
				hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)
				if _is_plain_lora_adapter(adapter):
					new_adapter = INT8LoRAPatchAdapter(
						adapter.loaded_keys,
						adapter.weights,
						weight_scale,
						seed=seed,
						outlier_method=outlier_method,
						hadanorm_sigma=hadanorm_sigma,
					)
				else:
					new_adapter = INT8WeightPatchAdapter(
						adapter,
						weight_scale,
						seed=seed,
						outlier_method=outlier_method,
						hadanorm_sigma=hadanorm_sigma,
					)

				final_patch_dict[key] = new_adapter
				applied_count += 1
			else:
				final_patch_dict[key] = adapter
		except Exception:
			final_patch_dict[key] = adapter

	return final_patch_dict, applied_count


def _dispatch_dynamic_single(model, lora_name, strength):
	from .int8_dynamic_lora import INT8DynamicLoraLoader
	return INT8DynamicLoraLoader().load_lora(model, lora_name, strength)


def _dispatch_dynamic_stack(model, kwargs):
	from .int8_dynamic_lora import INT8DynamicLoraStack
	return INT8DynamicLoraStack().apply_stack(model, **kwargs)


class INT8LoraLoader:
	"""
	Unified INT8 LoRA loader.

	Use `mode` to switch between stochastic INT8-space patching and dynamic runtime LoRA.
	"""

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"mode": (["Stochastic", "Dynamic"], {"tooltip": "Stochastic merges LoRA deltas into INT8 weights using stochastic rounding. Dynamic keeps compatible LoRAs as runtime additions without modifying INT8 weights."}),
				"model": ("MODEL", {"tooltip": "INT8 or float diffusion model to receive the LoRA patch."}),
				"lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file from ComfyUI's loras folder."}),
				"strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA strength for the diffusion model. Negative values invert the LoRA effect."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "load_lora"
	CATEGORY = "loaders"
	DESCRIPTION = "Load one LoRA for INT8 models. Choose between stochastic patching and dynamic runtime composition."

	def load_lora(self, mode, model, lora_name, strength, seed=318008):
		if strength == 0:
			return (model,)

		if mode == "Dynamic":
			return _dispatch_dynamic_single(model, lora_name, strength)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

		model_patcher = model.clone()
		key_map = _get_key_map(model_patcher)
		patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
		del lora

		module_cache = {}
		final_patch_dict, applied_count = _upgrade_patch_dict_for_int8(
			model_patcher=model_patcher,
			patch_dict=patch_dict,
			seed=seed,
			module_cache=module_cache,
		)

		model_patcher.add_patches(final_patch_dict, strength)

		logging.info(
			f"INT8 LoRA ({mode}): Registered '{lora_name}' with strength {strength:.2f} for {applied_count} quantized layers."
		)
		print(f"[INT8 LoRA:{mode}] Patched {applied_count} layers, skipped {len(patch_dict) - applied_count}.")
		return (model_patcher,)


class INT8LoraLoaderStack:
	"""
	Unified INT8 LoRA stack loader.

	Use `mode` to switch between stochastic stack patching and dynamic runtime stack composition.
	"""

	@classmethod
	def INPUT_TYPES(s):
		inputs = {
			"required": {
				"mode": (["Stochastic", "Dynamic"], {"tooltip": "Stochastic combines stack deltas before one INT8 rounding step. Dynamic keeps compatible LoRAs as runtime additions."}),
				"model": ("MODEL", {"tooltip": "INT8 or float diffusion model to receive the LoRA stack."}),
			},
			"optional": {}
		}
		lora_list = ["None"] + folder_paths.get_filename_list("loras")
		for i in range(1, 11):
			inputs["optional"][f"lora_{i}"] = (lora_list, {"tooltip": f"Optional LoRA slot {i}. Choose None to leave this slot unused."})
			inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": f"Strength for LoRA slot {i}. Ignored when the slot is None or strength is 0."})
		return inputs

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "apply_stack"
	CATEGORY = "loaders"
	DESCRIPTION = "Apply a LoRA stack for INT8 models with a selectable patching mode."

	def apply_stack(self, mode, model, seed=318008, **kwargs):
		if mode == "Dynamic":
			return _dispatch_dynamic_stack(model, kwargs)

		lora_entries = []
		for i in range(1, 11):
			name = kwargs.get(f"lora_{i}")
			strength = kwargs.get(f"strength_{i}", 0)
			if name and name != "None" and strength != 0:
				lora_entries.append((name, strength))

		if not lora_entries:
			return (model,)

		if len(lora_entries) == 1:
			lora_name, strength = lora_entries[0]
			return INT8LoraLoader().load_lora("Stochastic", model, lora_name, strength, seed=seed)

		model_patcher = model.clone()
		key_map = _get_key_map(model_patcher)

		layered_patches = {}
		for name, strength in lora_entries:
			path = folder_paths.get_full_path("loras", name)
			data = comfy.utils.load_torch_file(path, safe_load=True)
			patch_dict = comfy.lora.load_lora(data, key_map, log_missing=True)
			del data
			for key, adapter in patch_dict.items():
				if key not in layered_patches:
					layered_patches[key] = []
				layered_patches[key].append((adapter, strength))

		from .int8_quant import INT8MergedLoRAPatchAdapter
		final_patch_dict = {}
		applied_count = 0
		module_cache = {}

		for key, patches in layered_patches.items():
			try:
				target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
				is_quantized = hasattr(target_module, "_is_quantized") and target_module._is_quantized

				if not is_quantized:
					for adapter, adapter_strength in patches:
						model_patcher.add_patches({key: adapter}, adapter_strength)
					continue

				weight_scale = _get_weight_scale_for_module(target_module)
				outlier_method = getattr(target_module, "_outlier_method", None)
				hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)
				mergeable = all(hasattr(adapter, "calculate_weight") for adapter, _ in patches)
				if mergeable:
					final_patch_dict[key] = INT8MergedLoRAPatchAdapter(
						patches,
						weight_scale,
						seed=seed,
						outlier_method=outlier_method,
						hadanorm_sigma=hadanorm_sigma,
					)
					applied_count += 1
				else:
					for adapter, adapter_strength in patches:
						model_patcher.add_patches({key: adapter}, adapter_strength)
			except Exception:
				for adapter, strength in patches:
					model_patcher.add_patches({key: adapter}, strength)

		model_patcher.add_patches(final_patch_dict, 1.0)

		logging.info(f"INT8 LoRA Stack ({mode}): Merged {len(lora_entries)} LoRAs for {applied_count} quantized layers.")
		print(f"[INT8 LoRA Stack:{mode}] Applied {len(lora_entries)} LoRAs, merged {applied_count} quantized layers.")
		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8LoraLoader": INT8LoraLoader,
	"INT8LoraLoaderStack": INT8LoraLoaderStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8LoraLoader": "Load LoRA INT8",
	"INT8LoraLoaderStack": "Load LoRA Stack INT8",
}
