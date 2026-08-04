import json
import logging
import os
from contextlib import contextmanager

import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy.cli_args import args

try:
	from comfy.quant_ops import QuantizedTensor
except (ImportError, AttributeError):
	QuantizedTensor = None


_SUPPORTED_QUANTIZATION_FORMATS = {"int8_tensorwise", "convrot_w4a4"}


def _install_lazy_casting_param_workaround():
	try:
		import comfy.model_patcher as comfy_model_patcher
	except Exception:
		return None

	lazy_param_classes = [
		getattr(comfy_model_patcher, "LazyCastingParam", None),
		getattr(comfy_model_patcher, "LazyCastingParamPiece", None),
	]
	original_new_methods = []

	def _safe_new(cls, *args, **kwargs):
		tensor = kwargs.get("tensor", None)
		if tensor is None and args:
			tensor = args[-1]
		requires_grad = bool(
			isinstance(tensor, torch.Tensor)
			and (tensor.is_floating_point() or tensor.is_complex())
		)
		return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

	for lazy_param_cls in lazy_param_classes:
		if lazy_param_cls is None:
			continue
		original_new = getattr(lazy_param_cls, "__new__", None)
		if original_new is None:
			continue
		original_new_methods.append((lazy_param_cls, original_new))
		lazy_param_cls.__new__ = staticmethod(_safe_new)

	if not original_new_methods:
		return None

	def _restore():
		for lazy_param_cls, original_new in original_new_methods:
			lazy_param_cls.__new__ = original_new

	return _restore


def _is_int8_quantized_module(module):
	if not getattr(module, "_is_quantized", False):
		return False
	if getattr(module, "_quant_format", None) == "convrot_w4a4":
		return True

	weight = getattr(module, "weight", None)
	if not isinstance(weight, torch.Tensor):
		return False

	return weight.dtype == torch.int8


def _is_supported_quantized_module(module):
	if _is_int8_quantized_module(module):
		return True

	quant_format = getattr(module, "quant_format", None)
	if quant_format not in _SUPPORTED_QUANTIZATION_FORMATS:
		return False
	weight = getattr(module, "weight", None)
	return QuantizedTensor is not None and isinstance(weight, QuantizedTensor)


def _model_contains_supported_quantization(model_patcher):
	base_model = getattr(model_patcher, "model", None)
	if base_model is not None and hasattr(base_model, "modules"):
		if any(_is_supported_quantized_module(module) for module in base_model.modules()):
			return True

	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if not isinstance(object_patch_map, dict):
			continue
		if any(_is_supported_quantized_module(module) for module in object_patch_map.values()):
			return True

	return False


def _validate_quantized_model(model_patcher):
	if not _model_contains_supported_quantization(model_patcher):
		raise ValueError(
			"Save Quantized Model requires a MODEL containing Toolkit INT8 or native ConvRot INT4 weights. "
			"Place Enable Quantization on MODEL before Load LoRA (Quantized) and this save node."
		)

	get_attachment = getattr(model_patcher, "get_attachment", None)
	if not callable(get_attachment):
		return
	lora_signature = get_attachment("int8_lora_signature")
	if not isinstance(lora_signature, tuple):
		return
	if any(isinstance(entry, tuple) and entry and entry[0] == "Dynamic" for entry in lora_signature):
		raise ValueError(
			"Save Quantized Model cannot serialize Dynamic LoRAs because they are runtime-only. "
			"Use Stochastic mode, or bake stock LoRAs before quantization."
		)


def _module_has_non_float_weight(module):
	if getattr(module, "_quant_format", None) == "convrot_w4a4":
		return True
	weight = getattr(module, "weight", None)
	if not isinstance(weight, torch.Tensor):
		return False
	if weight.is_floating_point() or weight.is_complex():
		return False
	return True


def _resolve_source_metadata(model_patcher):
	seen_patcher_ids = set()
	current_patcher = model_patcher

	while current_patcher is not None and id(current_patcher) not in seen_patcher_ids:
		seen_patcher_ids.add(id(current_patcher))

		metadata = getattr(current_patcher, "_safetensors_metadata", None)
		if isinstance(metadata, dict) and metadata:
			return dict(metadata)

		inner_model = getattr(current_patcher, "model", None)
		metadata = getattr(inner_model, "_int8_source_metadata", None)
		if isinstance(metadata, dict) and metadata:
			return dict(metadata)

		current_patcher = getattr(current_patcher, "parent", None)

	return None


def _resolve_patch_target_module(base_model, patch_key):
	try:
		return comfy.utils.get_attr(base_model, patch_key)
	except Exception:
		pass

	diffusion_model = getattr(base_model, "diffusion_model", None)
	if diffusion_model is None:
		return None

	trimmed_key = patch_key
	if trimmed_key.startswith("diffusion_model."):
		trimmed_key = trimmed_key[len("diffusion_model."):]

	try:
		return comfy.utils.get_attr(diffusion_model, trimmed_key)
	except Exception:
		return None


def _collect_modules_for_save_workaround(model_patcher):
	base_model = getattr(model_patcher, "model", None)
	if base_model is None:
		return []

	modules = []
	seen_module_ids = set()

	if hasattr(base_model, "named_modules"):
		for _, module in base_model.named_modules():
			if not _is_int8_quantized_module(module):
				continue
			module_id = id(module)
			if module_id in seen_module_ids:
				continue
			seen_module_ids.add(module_id)
			modules.append(module)

	object_patches = getattr(model_patcher, "object_patches", None)
	if isinstance(object_patches, dict):
		for patch_key, patch_obj in object_patches.items():
			if not isinstance(patch_key, str):
				continue
			if not patch_key.startswith("diffusion_model."):
				continue
			if not _is_int8_quantized_module(patch_obj):
				continue

			target_module = _resolve_patch_target_module(base_model, patch_key)
			if target_module is None:
				continue

			module_id = id(target_module)
			if module_id in seen_module_ids:
				continue
			seen_module_ids.add(module_id)
			modules.append(target_module)

	diffusion_model = getattr(base_model, "diffusion_model", None)
	if diffusion_model is not None and hasattr(diffusion_model, "named_modules"):
		for _, module in diffusion_model.named_modules():
			if not getattr(module, "comfy_cast_weights", False):
				continue
			if not _module_has_non_float_weight(module):
				continue

			module_id = id(module)
			if module_id in seen_module_ids:
				continue
			seen_module_ids.add(module_id)
			modules.append(module)

	return modules


def _set_comfy_patched_weights_flag(modules):
	flag_states = []
	for module in modules:
		had_flag = hasattr(module, "comfy_patched_weights")
		old_flag = getattr(module, "comfy_patched_weights", False) if had_flag else False
		flag_states.append((module, had_flag, old_flag))
		module.comfy_patched_weights = True
	return flag_states


def _restore_comfy_patched_weights_flag(flag_states):
	for module, had_flag, old_flag in flag_states:
		if had_flag:
			module.comfy_patched_weights = old_flag
		else:
			delattr(module, "comfy_patched_weights")


@contextmanager
def _temporary_save_compatibility_shim(modules):
	flag_states = _set_comfy_patched_weights_flag(modules)
	restore_lazy_param = _install_lazy_casting_param_workaround()
	if restore_lazy_param is not None:
		logging.info("Quantization Toolkit save: enabled the temporary LazyCastingParam requires_grad workaround.")

	try:
		yield
	finally:
		if restore_lazy_param is not None:
			restore_lazy_param()
		_restore_comfy_patched_weights_flag(flag_states)


def _apply_object_patches_for_save(model_patcher):
	object_patches = getattr(model_patcher, "object_patches", None)
	if not isinstance(object_patches, dict) or not object_patches:
		return False

	patch_model = getattr(model_patcher, "patch_model", None)
	if not callable(patch_model):
		return False

	try:
		patch_model(load_weights=False)
		return True
	except Exception as e:
		logging.warning(f"Quantization Toolkit Save: failed to apply object patches before save ({e}).")
		return False


def _summarize_saved_quantized_checkpoint(path):
	try:
		from safetensors import safe_open
	except Exception:
		return

	try:
		dtype_counts = {}
		int8_weights = 0
		total_weights = 0
		weight_scales = 0
		comfy_quant_layers = 0
		int4_layers = 0
		int8_layers = 0
		with safe_open(path, framework="pt", device="cpu") as handle:
			metadata = handle.metadata()
			has_legacy_quant_metadata = isinstance(metadata, dict) and "_quantization_metadata" in metadata
			for key in handle.keys():
				tensor_slice = handle.get_slice(key)
				dtype = tensor_slice.get_dtype()
				dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
				if key.endswith(".weight"):
					total_weights += 1
					if dtype == "I8":
						int8_weights += 1
				elif key.endswith(".weight_scale"):
					weight_scales += 1
				elif key.endswith(".comfy_quant"):
					comfy_quant_layers += 1
					try:
						quant_config = json.loads(handle.get_tensor(key).numpy().tobytes())
						if quant_config.get("format") == "convrot_w4a4":
							int4_layers += 1
						elif quant_config.get("format") == "int8_tensorwise":
							int8_layers += 1
					except Exception:
						pass

		logging.info(
			"Quantization Toolkit save: checkpoint summary "
			f"(int8_weights={int8_weights}, weight_scales={weight_scales}, "
			f"int4_layers={int4_layers}, int8_layers={int8_layers}, "
			f"comfy_quant_layers={comfy_quant_layers}, total_weights={total_weights}, "
			f"dtypes={dtype_counts})."
		)
		if int8_weights == 0 or weight_scales == 0:
			logging.warning("Quantization Toolkit Save: saved checkpoint does not appear to contain packed INT4 or INT8 weights.")
		elif comfy_quant_layers == 0 and not has_legacy_quant_metadata:
			logging.warning(
				"Quantization Toolkit Save: saved checkpoint does not include native ComfyUI .comfy_quant metadata."
			)
		elif comfy_quant_layers < int8_weights and not has_legacy_quant_metadata:
			logging.warning(
				"Quantization Toolkit Save: only some packed weights include native ComfyUI .comfy_quant metadata. "
				"Plain INT8 and ConvRot layers can export natively; Toolkit QuaRot/HadaNorm layers require the Toolkit loader."
			)
	except Exception as e:
		logging.warning(f"Quantization Toolkit Save: failed to inspect saved checkpoint ({e}).")


class INT8ModelSave:
	def __init__(self):
		self.output_dir = folder_paths.get_output_directory()

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model": ("MODEL",),
				"filename_prefix": ("STRING", {"default": "quantized_models/Quantized_Model"}),
			},
			"hidden": {
				"prompt": "PROMPT",
				"extra_pnginfo": "EXTRA_PNGINFO",
			},
		}

	RETURN_TYPES = ()
	FUNCTION = "save"
	OUTPUT_NODE = True
	CATEGORY = "loaders"
	DESCRIPTION = "Save MODEL outputs that include Toolkit INT8 or native ConvRot INT4 layers with a DynamicVRAM-safe save path."

	def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
		_validate_quantized_model(model)

		full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
			filename_prefix,
			self.output_dir,
		)

		prompt_info = ""
		if prompt is not None:
			prompt_info = json.dumps(prompt)

		metadata = {}
		source_metadata = _resolve_source_metadata(model)
		if isinstance(source_metadata, dict):
			metadata.update(source_metadata)
		if not args.disable_metadata:
			metadata["prompt"] = prompt_info
			if extra_pnginfo is not None:
				for key, value in extra_pnginfo.items():
					metadata[key] = json.dumps(value)

		output_checkpoint = f"{filename}_{counter:05}_.safetensors"
		output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

		finalize_pending_int8 = getattr(model, "finalize_pending_int8", None)
		if callable(finalize_pending_int8):
			finalize_pending_int8()

		applied_object_patches = _apply_object_patches_for_save(model)
		if applied_object_patches:
			logging.info("Quantization Toolkit save: applied object patches before saving quantized modules.")

		modules_to_patch = _collect_modules_for_save_workaround(model)
		if not modules_to_patch:
			logging.warning("Quantization Toolkit Save: no target modules were found for the DynamicVRAM save workaround.")
		else:
			logging.info(f"Quantization Toolkit save: applying the Dynamic VRAM workaround to {len(modules_to_patch)} module(s).")
		with _temporary_save_compatibility_shim(modules_to_patch):
			comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata)

		_summarize_saved_quantized_checkpoint(output_checkpoint)

		return {}
