import logging

import comfy.lora
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .int8_lora_patching import (
	_append_lora_signature,
	_can_merge_stochastic_stack,
	_create_stochastic_stack_adapter,
	_get_key_map,
	_get_supported_quantization_format,
	_get_weight_scale_for_module,
	_is_additive_stochastic_patch,
	_model_has_supported_quantized_modules,
	_model_has_int4_modules,
	_normalize_lora_source_key,
	_resolve_target_module_cached,
	_set_lora_source_key,
	_uses_toolkit_quantized_runtime,
	_upgrade_patch_dict_for_int8,
	_wrap_adapter_for_stochastic,
)


LORA_MODE_STOCHASTIC = "Stochastic"
LORA_MODE_DYNAMIC = "Dynamic"
LORA_MODE_STANDARD = "Standard"
LORA_MODE_CHOICES = [LORA_MODE_STOCHASTIC, LORA_MODE_DYNAMIC, LORA_MODE_STANDARD]
QUANTIZED_LORA_TYPE = "QUANTIZATION_TOOLKIT_LORA"
QUANTIZED_LORA_IO = io.Custom(QUANTIZED_LORA_TYPE)
MAX_AUTOGROW_LORAS = 100


class QuantizedLoraSpec:
	def __init__(self, path, strength):
		self.path = path
		self.strength = strength


def _collect_autogrow_lora_entries(loras):
	def lora_index(item):
		input_name, _lora = item
		try:
			return int(input_name.rsplit("_", 1)[1])
		except (IndexError, ValueError):
			return MAX_AUTOGROW_LORAS + 1

	return [
		(lora.path, lora.strength)
		for _input_name, lora in sorted((loras or {}).items(), key=lora_index)
		if lora is not None and lora.strength != 0
	]


def _summarize_lora_entries(lora_entries, limit=10):
	visible_entries = lora_entries[:limit]
	summary = ", ".join(
		f"{name}@{float(strength):g}"
		for name, strength in visible_entries
	)
	remaining_count = len(lora_entries) - len(visible_entries)
	if remaining_count > 0:
		summary += f", ... (+{remaining_count} more)"
	return summary


def _dispatch_dynamic_single(model, lora_name, strength):
	from .int8_dynamic_lora import INT8DynamicLoraLoader
	return INT8DynamicLoraLoader().load_lora(model, lora_name, strength)


def _dispatch_dynamic_stack(model, lora_entries):
	from .int8_dynamic_lora import INT8DynamicLoraStack
	return INT8DynamicLoraStack().apply_loras(model, lora_entries)


def _dispatch_standard_single(model, lora_name, strength, seed=318008):
	if _model_has_supported_quantized_modules(model):
		lora_path = folder_paths.get_full_path("loras", lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_patcher = model.clone()
		key_map = _get_key_map(model_patcher)
		patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
		del lora
		for adapter in patch_dict.values():
			_set_lora_source_key(adapter, lora_name)

		final_patch_dict, applied_count = _upgrade_patch_dict_for_int8(
			model_patcher=model_patcher,
			patch_dict=patch_dict,
			seed=seed,
			module_cache={},
			defer_unquantized=False,
		)
		model_patcher.add_patches(final_patch_dict, strength)
		_append_lora_signature(model_patcher, LORA_MODE_STANDARD, lora_name, strength, seed)
		logging.info(f"Quantization Toolkit LoRA ({LORA_MODE_STANDARD}): patched {applied_count} INT8-aware layers.")
		return (model_patcher,)

	lora_path = folder_paths.get_full_path("loras", lora_name)
	lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
	model_patcher, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
	del lora
	return (model_patcher,)


def _collect_lora_entries(kwargs):
	lora_entries = []
	for i in range(1, 11):
		name = kwargs.get(f"lora_{i}")
		strength = kwargs.get(f"strength_{i}", 0)
		if name and name != "None" and strength != 0:
			lora_entries.append((name, strength))
	return lora_entries


def _dispatch_standard_stack(model, lora_entries, seed=318008):
	model_patcher = model
	for lora_name, strength in lora_entries:
		model_patcher = _dispatch_standard_single(model_patcher, lora_name, strength, seed=seed)[0]
	return (model_patcher,)


class INT8LoraLoader:
	"""
	Unified INT8 LoRA loader.

	Use `mode` to switch between standard patching, stochastic INT8-space patching,
	and dynamic runtime LoRA.
	"""

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"mode": (LORA_MODE_CHOICES, {"tooltip": "Standard uses ComfyUI's regular MODEL patch path. Stochastic requantizes patched weights and can lose small INT4 deltas. Dynamic applies LoRA deltas at runtime and preserves them for both INT8 and INT4."}),
				"model": ("MODEL", {"tooltip": "Quantized or float diffusion model to receive the LoRA patch."}),
				"lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file from ComfyUI's loras folder."}),
				"strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA strength for the diffusion model. Negative values invert the LoRA effect."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "load_lora"
	CATEGORY = "loaders"
	DESCRIPTION = "Load one LoRA with format-aware patching for Toolkit INT8 and native ConvRot INT4 models."

	def load_lora(self, mode, model, lora_name, strength, seed=318008):
		if strength == 0:
			return (model,)

		if mode == LORA_MODE_DYNAMIC:
			return _dispatch_dynamic_single(model, lora_name, strength)

		if mode == LORA_MODE_STOCHASTIC and _model_has_int4_modules(model):
			logging.warning(
				"Quantization Toolkit: Stochastic LoRA requantizes INT4 weights and may lose small deltas; "
				"use Dynamic mode to preserve the LoRA delta at runtime."
			)

		if mode == LORA_MODE_STANDARD:
			return _dispatch_standard_single(model, lora_name, strength)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

		model_patcher = model.clone()
		key_map = _get_key_map(model_patcher)
		patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
		del lora
		for adapter in patch_dict.values():
			_set_lora_source_key(adapter, lora_name)

		module_cache = {}
		final_patch_dict, applied_count = _upgrade_patch_dict_for_int8(
			model_patcher=model_patcher,
			patch_dict=patch_dict,
			seed=seed,
			module_cache=module_cache,
		)

		model_patcher.add_patches(final_patch_dict, strength)
		_append_lora_signature(model_patcher, mode, lora_name, strength, seed)

		logging.info(
			f"Quantization Toolkit LoRA ({mode}): registered '{lora_name}' with strength {strength:.2f}; "
			f"patched {applied_count} quantized layers and skipped {len(patch_dict) - applied_count}."
		)
		return (model_patcher,)


class INT8LoraLoaderStack:
	"""
	Unified INT8 LoRA stack loader.

	Use `mode` to switch between standard stack patching, stochastic INT8 stack
	patching, and dynamic runtime stack composition.
	"""

	@classmethod
	def INPUT_TYPES(s):
		inputs = {
			"required": {
				"mode": (LORA_MODE_CHOICES, {"tooltip": "Standard uses ComfyUI patching. Stochastic combines and requantizes patched weights, which can lose small INT4 deltas. Dynamic preserves LoRA deltas at runtime for both INT8 and INT4."}),
				"model": ("MODEL", {"tooltip": "Quantized or float diffusion model to receive the LoRA stack."}),
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
	DESCRIPTION = "Apply a LoRA stack to Toolkit INT8 or native ConvRot INT4 models."

	def apply_stack(self, mode, model, seed=318008, **kwargs):
		lora_entries = _collect_lora_entries(kwargs)
		return self.apply_loras(mode, model, lora_entries, seed=seed)

	def apply_loras(self, mode, model, lora_entries, seed=318008):
		if not lora_entries:
			return (model,)

		logging.info(
			"Quantization Toolkit LoRA stack (%s): effective entries: %s.",
			mode,
			_summarize_lora_entries(lora_entries),
		)

		if mode == LORA_MODE_DYNAMIC:
			return _dispatch_dynamic_stack(model, lora_entries)

		if mode == LORA_MODE_STOCHASTIC and _model_has_int4_modules(model):
			logging.warning(
				"Quantization Toolkit: Stochastic LoRA stacks requantize INT4 weights and may lose small deltas; "
				"use Dynamic mode to preserve LoRA deltas at runtime."
			)

		if mode == LORA_MODE_STANDARD:
			return _dispatch_standard_stack(model, lora_entries, seed=seed)

		if len(lora_entries) == 1:
			lora_name, strength = lora_entries[0]
			return INT8LoraLoader().load_lora(LORA_MODE_STOCHASTIC, model, lora_name, strength, seed=seed)

		model_patcher = model.clone()
		key_map = _get_key_map(model_patcher)

		layered_patches = {}
		for name, strength in lora_entries:
			path = folder_paths.get_full_path("loras", name)
			data = comfy.utils.load_torch_file(path, safe_load=True)
			patch_dict = comfy.lora.load_lora(data, key_map, log_missing=True)
			del data
			source_key = _normalize_lora_source_key(name)
			for key, adapter in patch_dict.items():
				_set_lora_source_key(adapter, name)
				if key not in layered_patches:
					layered_patches[key] = []
				layered_patches[key].append((adapter, strength, source_key))

		final_patch_dict = {}
		applied_count = 0
		module_cache = {}
		model_is_quantized = _model_has_supported_quantized_modules(model_patcher)

		for key, sourced_patches in layered_patches.items():
			if all(_is_additive_stochastic_patch(adapter) for adapter, _strength, _source_key in sourced_patches):
				sourced_patches = sorted(sourced_patches, key=lambda patch: patch[2])
			patches = [
				(adapter, strength)
				for adapter, strength, _source_key in sourced_patches
			]
			try:
				target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
				quantization_format = _get_supported_quantization_format(target_module)

				if quantization_format is None:
					if model_is_quantized:
						for adapter, adapter_strength in patches:
							model_patcher.add_patches({key: adapter}, adapter_strength)
						continue
					if _can_merge_stochastic_stack(patches):
						final_patch_dict[key] = _create_stochastic_stack_adapter(
							patches,
							1.0,
							seed,
							defer_until_quantized=True,
						)
					else:
						for adapter, adapter_strength in patches:
							wrapped_adapter = _wrap_adapter_for_stochastic(
								adapter,
								1.0,
								seed,
								defer_until_quantized=True,
							)
							model_patcher.add_patches({key: wrapped_adapter}, adapter_strength)
					continue

				applied_count += 1
				if (
					not _uses_toolkit_quantized_runtime(target_module)
					or quantization_format == "convrot_w4a4"
				):
					# Native ComfyUI quantized modules aggregate the full patch list
					# before one requantization. Keep their adapters native so VBAR can
					# reconstruct and prefetch them using ComfyUI's standard protocol.
					for adapter, adapter_strength in patches:
						model_patcher.add_patches({key: adapter}, adapter_strength)
					continue

				weight_scale = _get_weight_scale_for_module(target_module)
				outlier_method = getattr(target_module, "_outlier_method", None)
				hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)
				mergeable = all(hasattr(adapter, "calculate_weight") for adapter, _ in patches)
				if mergeable:
					final_patch_dict[key] = _create_stochastic_stack_adapter(
						patches,
						weight_scale,
						seed=seed,
						outlier_method=outlier_method,
						hadanorm_sigma=hadanorm_sigma,
					)
				else:
					for adapter, adapter_strength in patches:
						model_patcher.add_patches({key: adapter}, adapter_strength)
			except Exception:
				for adapter, strength in patches:
					model_patcher.add_patches({key: adapter}, strength)

		model_patcher.add_patches(final_patch_dict, 1.0)
		for lora_name, strength in lora_entries:
			_append_lora_signature(model_patcher, mode, lora_name, strength, seed)

		logging.info(
			f"Quantization Toolkit LoRA stack ({mode}): applied {len(lora_entries)} LoRAs and patched "
			f"{applied_count} quantized layers."
		)
		return (model_patcher,)


class QuantizedLoraConfig(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="QuantizedLoraConfig",
			display_name="LoRA Stack Entry (Quantized)",
			category="loaders",
			description="Configure one LoRA for Apply LoRA Stack (Quantized) without loading or modifying a MODEL.",
			inputs=[
				io.Combo.Input(
					"path",
					options=folder_paths.get_filename_list("loras"),
					tooltip="LoRA path relative to ComfyUI's loras folder.",
				),
				io.Float.Input(
					"strength",
					default=1.0,
					min=-10.0,
					max=10.0,
					step=0.01,
					tooltip="LoRA strength. Set to 0 or bypass this node to disable the LoRA.",
				),
			],
			outputs=[QUANTIZED_LORA_IO.Output(display_name="lora")],
		)

	@classmethod
	def execute(cls, path, strength):
		return io.NodeOutput(QuantizedLoraSpec(path, strength))


class QuantizedLoraPatcher(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		lora_inputs = io.Autogrow.TemplateNames(
			input=QUANTIZED_LORA_IO.Input("lora"),
			names=[f"lora_{index}" for index in range(1, MAX_AUTOGROW_LORAS + 1)],
			min=0,
		)
		# Keep mode first in the backend schema for logical consistency. ComfyUI's
		# frontend renders widgets after non-widget autogrow sockets regardless of
		# schema order, so the dropdown currently appears below the LoRA inputs.
		return io.Schema(
			node_id="QuantizedLoraPatcher",
			display_name="Apply LoRA Stack (Quantized)",
			category="loaders",
			description="Patch any number of configured LoRAs into a quantized or floating-point diffusion MODEL.",
			inputs=[
				io.Combo.Input(
					"mode",
					options=LORA_MODE_CHOICES,
					default=LORA_MODE_STOCHASTIC,
					tooltip="Standard uses ComfyUI patching. Stochastic requantizes patched weights. Dynamic preserves LoRA deltas at runtime.",
				),
				io.Model.Input("model", tooltip="Quantized or floating-point diffusion model to receive the LoRAs."),
				io.Autogrow.Input(
					"loras",
					template=lora_inputs,
					optional=True,
					tooltip="Connect LoRA Stack Entry (Quantized) outputs; another input appears as each one is connected.",
				),
			],
			outputs=[io.Model.Output()],
		)

	@classmethod
	def execute(cls, model, mode, loras=None):
		lora_entries = _collect_autogrow_lora_entries(loras)
		return INT8LoraLoaderStack().apply_loras(mode, model, lora_entries)


NODE_CLASS_MAPPINGS = {
	"INT8LoraLoader": INT8LoraLoader,
	"INT8LoraLoaderStack": INT8LoraLoaderStack,
	"QuantizedLoraConfig": QuantizedLoraConfig,
	"QuantizedLoraPatcher": QuantizedLoraPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8LoraLoader": "Load LoRA (Quantized)",
	"INT8LoraLoaderStack": "Load LoRA Stack (Quantized)",
	"QuantizedLoraConfig": "LoRA Stack Entry (Quantized)",
	"QuantizedLoraPatcher": "Apply LoRA Stack (Quantized)",
}
