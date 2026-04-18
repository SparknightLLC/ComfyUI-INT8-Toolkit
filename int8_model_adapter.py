import logging
import uuid

import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import torch
from torch import nn

from .int8_quant import (
	Int8TensorwiseOps,
	_QUAROT_AVAILABLE,
	_QUAROT_GROUP_SIZE,
	_get_int8_compute_device,
	_is_float8_dtype,
	_quarot_build_hadamard,
	_quarot_rotate_weight,
	quantize_int8_rowwise,
)
from .int8_unet_loader import MODEL_TYPE_CHOICES as LOADER_MODEL_TYPE_CHOICES
from .int8_unet_loader import get_model_type_exclusions


AUTO_MODEL_TYPE = "auto"
NONE_MODEL_TYPE = "none"
MODEL_TYPE_CHOICES = [AUTO_MODEL_TYPE] + LOADER_MODEL_TYPE_CHOICES + [NONE_MODEL_TYPE]
_INT8_MODEL_ADAPTER_WRAPPER_KEY = "int8_model_adapter_cache_notice"

MODEL_TYPE_FINGERPRINTS = {
	"flux2": (
		"guidance_in",
		"img_in",
		"txt_in",
		"final_layer",
		"double_blocks",
		"single_blocks",
		"double_stream_modulation_img",
		"double_stream_modulation_txt",
		"single_stream_modulation",
	),
	"z-image": (
		"cap_embedder",
		"context_refiner",
		"noise_refiner",
		"cap_pad_token",
		"x_pad_token",
	),
	"chroma": (
		"distilled_guidance_layer",
		"nerf_image_embedder",
		"nerf_blocks",
		"nerf_final_layer_conv",
	),
	"wan": (
		"patch_embedding",
		"text_embedding",
		"time_embedding",
		"time_projection",
		"img_emb",
	),
	"ltx2": (
		"audio_adaln_single",
		"audio_caption_projection",
		"audio_patchify_proj",
		"av_ca_a2v_gate_adaln_single",
		"av_ca_v2a_gate_adaln_single",
	),
	"qwen": (
		"time_text_embed",
		"norm_out",
		"proj_out",
	),
	"ernie": (
		"text_proj",
		"layers.35",
		"x_embedder",
		"adaLN",
	),
	"anima": (
		"llm",
		"blocks.0.",
		"blocks.1.",
		"blocks.2.",
	),
}

MODEL_TYPE_REQUIRED_MARKERS = {
	"flux2": ("guidance_in", "double_stream_modulation_img", "double_stream_modulation_txt"),
	"z-image": ("cap_embedder", "context_refiner", "noise_refiner"),
	"wan": ("patch_embedding", "time_projection"),
	"ltx2": ("audio_adaln_single", "audio_caption_projection", "audio_patchify_proj"),
	"ernie": ("text_proj", "layers.35"),
	"anima": ("llm",),
	"qwen": ("time_text_embed", "norm_out", "proj_out"),
}

def _module_weight_key(module_name):
	return f"diffusion_model.{module_name}.weight"


def _module_patch_key(module_name):
	return f"diffusion_model.{module_name}"


def _patch_base_key(patch_key):
	return patch_key[0] if isinstance(patch_key, tuple) else patch_key


def _is_excluded(module_name, excluded_names):
	return any(excluded_name in module_name for excluded_name in excluded_names)


def _marker_in_module_names(module_names, marker):
	return any(marker in module_name for module_name in module_names)


def _infer_model_type_from_modules(diffusion_model):
	module_names = [
		module_name
		for module_name, _module in diffusion_model.named_modules()
		if module_name
	]
	if not module_names:
		return None

	scores = []
	for candidate_model_type, markers in MODEL_TYPE_FINGERPRINTS.items():
		required_markers = MODEL_TYPE_REQUIRED_MARKERS.get(candidate_model_type, ())
		if required_markers and not any(_marker_in_module_names(module_names, marker) for marker in required_markers):
			continue

		score = sum(1 for marker in markers if _marker_in_module_names(module_names, marker))
		if score >= 2:
			scores.append((score, candidate_model_type))

	if not scores:
		return None

	scores.sort(reverse=True)
	best_score, best_model_type = scores[0]
	if len(scores) > 1 and scores[1][0] == best_score:
		return None
	return best_model_type


def _get_conservative_auto_exclusions():
	excluded_names = []
	seen_names = set()
	for candidate_model_type in LOADER_MODEL_TYPE_CHOICES:
		for excluded_name in get_model_type_exclusions(candidate_model_type):
			if excluded_name in seen_names:
				continue
			seen_names.add(excluded_name)
			excluded_names.append(excluded_name)
	return excluded_names


def _resolve_model_type_and_exclusions(model_type, diffusion_model, log_progress):
	if model_type == AUTO_MODEL_TYPE:
		detected_model_type = _infer_model_type_from_modules(diffusion_model)
		if detected_model_type is None:
			logging.warning(
				"INT8 Model Adapter: auto model_type could not identify this model; "
				"using conservative union exclusions. Select a model_type manually for better speed."
			)
			return "auto-conservative", _get_conservative_auto_exclusions()

		if log_progress:
			print(f"[INT8 Model Adapter] auto model_type resolved to {detected_model_type}")
		return detected_model_type, get_model_type_exclusions(detected_model_type)

	if model_type == NONE_MODEL_TYPE:
		return NONE_MODEL_TYPE, []

	return model_type, get_model_type_exclusions(model_type)


def _is_supported_linear(module):
	if isinstance(module, Int8TensorwiseOps.Linear):
		return False
	if getattr(module, "_is_quantized", False):
		return False
	if not isinstance(module, nn.Linear):
		return False
	weight = getattr(module, "weight", None)
	if not isinstance(weight, torch.Tensor):
		return False
	if weight.ndim != 2:
		return False
	if weight.shape[0] <= 1 or weight.shape[1] <= 1:
		return False
	return weight.dtype in (torch.float16, torch.bfloat16, torch.float32) or _is_float8_dtype(weight.dtype)


def _collect_layer_patch_keys(model_patcher, module_name):
	weight_key = _module_weight_key(module_name)
	return [
		patch_key
		for patch_key in model_patcher.patches
		if _patch_base_key(patch_key) == weight_key
	]


def _get_source_weight(model_patcher, module_name, module, bake_loaded_loras):
	weight = module.weight.detach()
	weight_key = _module_weight_key(module_name)
	layer_patch_keys = _collect_layer_patch_keys(model_patcher, module_name)

	if not bake_loaded_loras or not layer_patch_keys:
		return weight, []

	compute_device = _get_int8_compute_device(weight.device)
	try:
		intermediate_dtype = comfy.model_management.lora_compute_dtype(compute_device)
	except Exception:
		intermediate_dtype = torch.float32
	if intermediate_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
		intermediate_dtype = torch.float16

	work_weight = weight.to(compute_device, dtype=intermediate_dtype, copy=True)
	layer_patches = []
	for patch_key in layer_patch_keys:
		layer_patches.extend(model_patcher.patches.get(patch_key, []))
	patched_weight = comfy.lora.calculate_weight(
		layer_patches,
		work_weight,
		weight_key,
		intermediate_dtype=intermediate_dtype,
	)
	return patched_weight.detach(), layer_patch_keys


def _quantize_linear_module(module_name, module, source_weight, enable_quarot):
	compute_device = _get_int8_compute_device(source_weight.device)
	weight_work = source_weight.to(compute_device, non_blocking=True)

	if _is_float8_dtype(weight_work.dtype):
		weight_work = weight_work.to(torch.float16 if compute_device.type == "cuda" else torch.float32)
	elif enable_quarot and weight_work.dtype in (torch.float16, torch.bfloat16):
		weight_work = weight_work.float()

	use_quarot = False
	quarot_hadamard = None
	if (
		enable_quarot
		and _QUAROT_AVAILABLE
		and weight_work.ndim == 2
		and weight_work.shape[1] % _QUAROT_GROUP_SIZE == 0
	):
		try:
			h_matrix = _quarot_build_hadamard(_QUAROT_GROUP_SIZE, device=compute_device, dtype=weight_work.dtype)
			weight_work = _quarot_rotate_weight(weight_work, h_matrix, group_size=_QUAROT_GROUP_SIZE)
			use_quarot = True
			quarot_hadamard = h_matrix.detach().cpu()
		except Exception as e:
			logging.warning(f"INT8 Model Adapter: QuaRot skipped for {module_name} ({e}).")

	q_weight, q_scale = quantize_int8_rowwise(weight_work)
	q_module = Int8TensorwiseOps.Linear(
		module.in_features,
		module.out_features,
		bias=module.bias is not None,
		device=torch.device("meta"),
	)
	q_module.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
	q_module.weight_scale = (
		q_scale.cpu()
		if isinstance(q_scale, torch.Tensor)
		else torch.tensor([float(q_scale)], dtype=torch.float32)
	)
	q_module._is_quantized = True
	q_module._is_per_row = q_module.weight_scale.dim() == 2 and q_module.weight_scale.shape[1] == 1
	q_module._use_quarot = use_quarot
	q_module.quarot_hadamard = quarot_hadamard
	q_module.compute_dtype = getattr(module, "compute_dtype", torch.bfloat16)
	q_module.dynamic_lora_entries = None
	q_module.lora_A = None
	q_module.lora_B = None
	q_module.lora_alpha = None

	if module.bias is not None:
		q_module.bias = nn.Parameter(module.bias.detach().cpu(), requires_grad=False)
	else:
		q_module.bias = None

	q_module.train(module.training)
	return q_module, use_quarot


def _cleanup_torch_memory():
	if not torch.cuda.is_available():
		return
	try:
		torch.cuda.empty_cache()
	except Exception:
		pass


def _extract_transformer_options(args, kwargs):
	transformer_options = kwargs.get("transformer_options", None)
	if transformer_options is None and len(args) > 5:
		transformer_options = args[5]
	if transformer_options is None:
		transformer_options = {}
	return transformer_options


def _is_first_sampling_step(transformer_options):
	sample_sigmas = transformer_options.get("sample_sigmas", None)
	current_sigmas = transformer_options.get("sigmas", None)
	if not isinstance(sample_sigmas, torch.Tensor) or sample_sigmas.numel() == 0:
		return False
	if not isinstance(current_sigmas, torch.Tensor) or current_sigmas.numel() == 0:
		return False

	try:
		start_sigma = float(sample_sigmas.reshape(-1)[0].item())
		current_sigma = float(current_sigmas.reshape(-1)[0].item())
	except Exception:
		return False

	return abs(current_sigma - start_sigma) <= max(1e-6, abs(start_sigma) * 1e-6)


def _int8_model_adapter_notice_wrapper(executor, *args, **kwargs):
	transformer_options = _extract_transformer_options(args, kwargs)
	adapter_state = transformer_options.get("int8_model_adapter", None)
	base_model = executor.class_obj
	diffusion_model = getattr(base_model, "diffusion_model", None)

	if isinstance(adapter_state, dict) and adapter_state.get("log_progress") and diffusion_model is not None:
		if getattr(diffusion_model, "_int8_model_adapter_skip_cache_notice_once", False):
			diffusion_model._int8_model_adapter_skip_cache_notice_once = False
			diffusion_model._int8_model_adapter_notice_in_generation = True
		elif _is_first_sampling_step(transformer_options):
			if not getattr(diffusion_model, "_int8_model_adapter_notice_in_generation", False):
				print(
					"[INT8 Model Adapter] Reusing cached INT8 MODEL output "
					f"(quantized_layers={adapter_state.get('quantized_layers', '?')}, "
					f"model_type={adapter_state.get('model_type', '?')})."
				)
				diffusion_model._int8_model_adapter_notice_in_generation = True
		else:
			diffusion_model._int8_model_adapter_notice_in_generation = False

	return executor(*args, **kwargs)


def _ensure_int8_model_adapter_notice_wrapper(model_patcher):
	model_patcher.remove_wrappers_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		_INT8_MODEL_ADAPTER_WRAPPER_KEY,
	)
	model_patcher.add_wrapper_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		_INT8_MODEL_ADAPTER_WRAPPER_KEY,
		_int8_model_adapter_notice_wrapper,
	)


def _collect_int8_candidates(diffusion_model, excluded_names):
	return [
		(module_name, module)
		for module_name, module in diffusion_model.named_modules()
		if module_name
		and not _is_excluded(module_name, excluded_names)
		and _is_supported_linear(module)
	]


def _clear_prior_int8_object_patches(model_patcher):
	for patch_key, patch_obj in list(model_patcher.object_patches.items()):
		if patch_key.startswith("diffusion_model.") and isinstance(patch_obj, Int8TensorwiseOps.Linear):
			model_patcher.object_patches.pop(patch_key, None)


class INT8ModelAdapter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL", {"tooltip": "The stock-loaded diffusion model to convert to this extension's INT8 linear runtime."}),
				"enable_int8": ("BOOLEAN", {"default": True, "tooltip": "Disable this to pass the input model through unchanged without removing the node from a workflow."}),
				"model_type": (MODEL_TYPE_CHOICES, {"default": AUTO_MODEL_TYPE, "tooltip": "Architecture preset used to skip layers that are usually quality-sensitive or unsafe to quantize. Auto inspects the loaded MODEL. Use none only for experiments."}),
				"bake_loaded_loras": ("BOOLEAN", {"default": True, "tooltip": "Apply existing stock LoRA weight patches, including sliced patches, before quantization, then remove the consumed patches to avoid applying them twice. If disabled, layers with pending patches are left unquantized."}),
				"enable_quarot": ("BOOLEAN", {"default": False, "tooltip": "Apply Hadamard rotation before quantizing compatible layers. This can improve some outlier cases but may reduce quality on some models."}),
				"use_triton": ("BOOLEAN", {"default": True, "tooltip": "Use this extension's Triton INT8 matmul kernels when available; disable for troubleshooting or fallback benchmarking."}),
				"log_progress": ("BOOLEAN", {"default": True, "tooltip": "Print quantization progress and layer counts to the ComfyUI console."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "apply_int8"
	CATEGORY = "loaders"
	DESCRIPTION = "Convert a stock-loaded diffusion MODEL to INT8 W8A8. Put this after stock Load LoRA to bake loaded LoRAs before quantization."

	def apply_int8(
		self,
		model,
		enable_int8,
		model_type,
		bake_loaded_loras,
		enable_quarot,
		use_triton,
		log_progress,
	):
		if not enable_int8:
			return (model,)

		model_patcher = model.clone()
		_clear_prior_int8_object_patches(model_patcher)
		diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
		if diffusion_model is None:
			logging.warning("INT8 Model Adapter: model has no diffusion_model; returning unchanged model.")
			return (model_patcher,)

		resolved_model_type, excluded_names = _resolve_model_type_and_exclusions(
			model_type,
			diffusion_model,
			bool(log_progress),
		)
		Int8TensorwiseOps.use_triton = bool(use_triton)

		candidates = _collect_int8_candidates(diffusion_model, excluded_names)
		if not candidates:
			try:
				if log_progress:
					print("[INT8 Model Adapter] No eligible layers found on first scan; forcing model load and rescanning.")
				comfy.model_management.load_models_gpu([model_patcher], force_patch_weights=True, force_full_load=True)
				diffusion_model = getattr(model_patcher.model, "diffusion_model", diffusion_model)
				candidates = _collect_int8_candidates(diffusion_model, excluded_names)
			except Exception as e:
				logging.warning(f"INT8 Model Adapter: forced model load failed during candidate scan ({e}).")

		total = len(candidates)
		quantized = 0
		quarot_count = 0
		baked_lora_count = 0
		skipped_patched_count = 0
		last_bucket = -1

		if log_progress:
			print(f"[INT8 Model Adapter] Starting MODEL quantization (eligible linear layers: {total})")

		for index, (module_name, module) in enumerate(candidates, start=1):
			try:
				pending_patch_keys = _collect_layer_patch_keys(model_patcher, module_name)
				if pending_patch_keys and not bake_loaded_loras:
					skipped_patched_count += 1
					continue

				source_weight, baked_patch_keys = _get_source_weight(
					model_patcher,
					module_name,
					module,
					bool(bake_loaded_loras),
				)
				q_module, used_quarot = _quantize_linear_module(
					module_name,
					module,
					source_weight,
					bool(enable_quarot),
				)
				model_patcher.add_object_patch(_module_patch_key(module_name), q_module)
				quantized += 1
				if used_quarot:
					quarot_count += 1
				if baked_patch_keys:
					for patch_key in baked_patch_keys:
						model_patcher.patches.pop(patch_key, None)
					baked_lora_count += len(baked_patch_keys)
				del source_weight
			except Exception as e:
				logging.warning(f"INT8 Model Adapter: skipped {module_name} ({e}).")

			if (index % 8) == 0:
				_cleanup_torch_memory()

			if log_progress and total > 0:
				percent = min(100, int((index * 100) / total))
				bucket = percent // 5
				if bucket != last_bucket:
					last_bucket = bucket
					print(
						f"[INT8 Model Adapter] {percent:3d}% "
						f"({index}/{total}) quantized={quantized} "
						f"baked_patches={baked_lora_count} "
						f"skipped_patched={skipped_patched_count} "
						f"quarot={quarot_count}"
					)

		if "transformer_options" not in model_patcher.model_options:
			model_patcher.model_options["transformer_options"] = {}
		else:
			model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()

		model_patcher.model_options["transformer_options"]["int8_model_adapter"] = {
			"model_type": resolved_model_type,
			"requested_model_type": model_type,
			"bake_loaded_loras": bool(bake_loaded_loras),
			"enable_quarot": bool(enable_quarot),
			"use_triton": bool(use_triton),
			"log_progress": bool(log_progress),
			"quantized_layers": quantized,
			"baked_lora_layers": baked_lora_count,
			"quarot_layers": quarot_count,
			"skipped_patched_layers": skipped_patched_count,
		}
		setattr(diffusion_model, "_int8_model_adapter_skip_cache_notice_once", True)
		_ensure_int8_model_adapter_notice_wrapper(model_patcher)
		model_patcher.patches_uuid = uuid.uuid4()
		_cleanup_torch_memory()

		if log_progress:
			print(
				"[INT8 Model Adapter] Complete "
				f"(quantized={quantized}, baked_patches={baked_lora_count}, "
				f"skipped_patched_layers={skipped_patched_count}, quarot={quarot_count})"
			)

		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8ModelAdapter": INT8ModelAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8ModelAdapter": "Enable INT8 on MODEL",
}
