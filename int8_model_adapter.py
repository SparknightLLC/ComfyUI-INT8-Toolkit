import logging
import os
import uuid
import weakref

import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import torch
from torch import nn

from .int8_quant import (
	Int8TensorwiseOps,
	OUTLIER_METHOD_CONVROT,
	OUTLIER_METHOD_HADANORM,
	OUTLIER_METHOD_NONE,
	OUTLIER_METHOD_QUAROT,
	QUANTIZATION_MODE_CHOICES,
	QUANTIZATION_MODE_INT4_MIXED,
	normalize_quantization_mode,
	quantization_mode_is_int4,
	quantization_mode_outlier_method,
	quantize_native_int4,
	native_int4_available,
	INT8_BACKEND_CHOICES,
	DEFAULT_INT8_BACKEND,
	INT8_BACKEND_TRITON,
	INT8_BACKEND_TRITON_LEGACY_UNSAFE,
	SMALL_BATCH_FALLBACK_CHOICES,
	DEFAULT_SMALL_BATCH_FALLBACK,
	_build_outlier_hadamard,
	_compute_hadanorm_sigma,
	_get_outlier_group_size,
	_prepack_torch_int_mm_weight,
	_get_int8_compute_device,
	_is_float8_dtype,
	_outlier_method_uses_hadamard,
	_rotate_weight_for_outlier_method,
	configure_int8_module_runtime,
	quantize_int8_rowwise,
	tensor_to_int8_compute_device,
)
from .int8_unet_loader import MODEL_TYPE_CHOICES as LOADER_MODEL_TYPE_CHOICES
from .int8_unet_loader import MODEL_TYPE_BOOGU
from .int8_unet_loader import MODEL_TYPE_FLUX2_FAST_UNSAFE
from .int8_unet_loader import MODEL_TYPE_HIDREAM_O1
from .int8_unet_loader import MODEL_TYPE_IDEOGRAM4
from .int8_unet_loader import MODEL_TYPE_KREA2
from .int8_unet_loader import DEFAULT_QUANTIZATION_MODE
from .int8_unet_loader import get_model_type_quantization_preset
from .quantization_policy import DEFAULT_INT4_MIXED_RATIO
from .quantization_policy import is_int4_compatible_linear
from .quantization_policy import normalize_int4_mixed_ratio
from .quantization_policy import select_int4_mixed_layers

try:
	import comfy_api.torch_helpers.torch_compile as comfy_torch_compile
	_TORCH_COMPILE_HELPER_AVAILABLE = True
except Exception:
	comfy_torch_compile = None
	_TORCH_COMPILE_HELPER_AVAILABLE = False

try:
	from comfy.weight_adapter.lora import LoRAAdapter
	from comfy.weight_adapter.base import WeightAdapterBase
	_WEIGHT_ADAPTER_AVAILABLE = True
except Exception:
	LoRAAdapter = None
	WeightAdapterBase = None
	_WEIGHT_ADAPTER_AVAILABLE = False


AUTO_MODEL_TYPE = "auto"
NONE_MODEL_TYPE = "none"
MODEL_TYPE_CHOICES = [AUTO_MODEL_TYPE] + LOADER_MODEL_TYPE_CHOICES + [NONE_MODEL_TYPE]
QUANTIZATION_CONTROL_AS_NEEDED = "as_needed"
QUANTIZATION_CONTROL_ALWAYS = "always"
QUANTIZATION_CONTROL_BYPASS = "bypass"
QUANTIZATION_CONTROL_CHOICES = [
	QUANTIZATION_CONTROL_AS_NEEDED,
	QUANTIZATION_CONTROL_ALWAYS,
	QUANTIZATION_CONTROL_BYPASS,
]
_INT8_MODEL_ADAPTER_WRAPPER_KEY = "int8_model_adapter_cache_notice"
_INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY = "int8_model_adapter_original_modules"
_INT8_MODEL_ADAPTER_OUTPUT_CACHE_KEY = "int8_model_adapter_output_cache"
_INT8_LORA_SIGNATURE_ATTACHMENT_KEY = "int8_lora_signature"
_INT8_MODEL_ADAPTER_LAST_MODEL_REF = None
_INT8_MODEL_ADAPTER_LAST_MODEL_ID = None
try:
	_INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT = max(0, int(os.environ.get("INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT", "1")))
except ValueError:
	_INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT = 1

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
	MODEL_TYPE_HIDREAM_O1: (
		"embed",
		"language_model",
		"language_model.layers",
		"language_model.layers.35.mlp",
	),
	MODEL_TYPE_IDEOGRAM4: (
		"embed_image_indicator",
		"t_embedding",
	),
	MODEL_TYPE_KREA2: (
		"first",
		"last",
		"tmlp",
		"tproj",
		"txtfusion",
		"txtmlp",
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
	MODEL_TYPE_BOOGU: (
		"double_stream_layers",
		"img_instruct_attn",
		"ref_image_patch_embedder",
		"noise_refiner",
		"ref_image_refiner",
		"context_refiner",
		"norm_out",
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
		"time",
		"x_embedder",
		"adaLN",
	),
	"anima": (
		"embed",
		"llm",
		"adaln",
	),
}

MODEL_TYPE_REQUIRED_MARKERS = {
	"flux2": ("guidance_in", "double_stream_modulation_img", "double_stream_modulation_txt"),
	MODEL_TYPE_BOOGU: ("double_stream_layers", "img_instruct_attn", "ref_image_patch_embedder"),
	MODEL_TYPE_HIDREAM_O1: ("language_model.layers",),
	MODEL_TYPE_IDEOGRAM4: ("embed_image_indicator", "t_embedding"),
	MODEL_TYPE_KREA2: ("txtfusion", "tmlp", "tproj"),
	"z-image": ("cap_embedder", "context_refiner", "noise_refiner"),
	"wan": ("patch_embedding", "time_projection"),
	"ltx2": ("audio_adaln_single", "audio_caption_projection", "audio_patchify_proj"),
	"ernie": ("text_proj", "x_embedder"),
	"anima": ("llm",),
	"qwen": ("time_text_embed", "norm_out", "proj_out"),
}

def _module_weight_key(module_name):
	return f"diffusion_model.{module_name}.weight"


def _module_patch_key(module_name):
	return f"diffusion_model.{module_name}"


def _patch_base_key(patch_key):
	return patch_key[0] if isinstance(patch_key, tuple) else patch_key


def _matches_layer_patterns(module_name, layer_patterns):
	return any(layer_pattern in module_name for layer_pattern in layer_patterns)


def _is_comfy_quantized_tensor(value):
	return (
		isinstance(value, torch.Tensor)
		and callable(getattr(value, "dequantize", None))
		and (
			bool(getattr(value, "is_quantized", False))
			or hasattr(value, "_qdata")
			or hasattr(value, "_layout_cls")
			or hasattr(value, "params")
		)
	)


def _is_linear_like(module):
	if isinstance(module, nn.Linear):
		return True
	if module.__class__.__name__ != "Linear":
		return False
	return (
		hasattr(module, "in_features")
		and hasattr(module, "out_features")
		and hasattr(module, "weight")
		and callable(getattr(module, "forward", None))
	)


def _is_supported_weight_tensor(weight):
	if not isinstance(weight, torch.Tensor):
		return False
	if weight.ndim != 2:
		return False
	if weight.shape[0] <= 1 or weight.shape[1] <= 1:
		return False
	if _is_comfy_quantized_tensor(weight):
		return True
	return weight.dtype in (torch.float16, torch.bfloat16, torch.float32) or _is_float8_dtype(weight.dtype)


def _materialize_source_weight(weight, device=None, dtype=None):
	if _is_comfy_quantized_tensor(weight):
		weight = weight.dequantize()
	if device is not None or dtype is not None:
		to_kwargs = {"copy": True}
		if device is not None:
			to_kwargs["device"] = device
		if dtype is not None:
			to_kwargs["dtype"] = dtype
		weight = weight.to(**to_kwargs)
	return weight.detach()


def _marker_in_module_names(module_names, marker):
	return any(marker in module_name for module_name in module_names)


def _is_sdxl_diffusion_model(diffusion_model):
	for module_name, module in diffusion_model.named_modules():
		if not module_name.startswith("label_emb"):
			continue
		in_features = getattr(module, "in_features", None)
		if in_features in (2560, 2816):
			return True
	return False


def _infer_model_type_from_modules(diffusion_model):
	if _is_sdxl_diffusion_model(diffusion_model):
		return "sdxl"

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


def _empty_quantization_preset():
	return {"keep_float": (), "int4_sensitive": ()}


def _get_conservative_auto_preset():
	merged_preset = {"keep_float": [], "int4_sensitive": []}
	seen_names = {key: set() for key in merged_preset}
	for candidate_model_type in LOADER_MODEL_TYPE_CHOICES:
		candidate_preset = get_model_type_quantization_preset(candidate_model_type)
		for preset_key in merged_preset:
			for layer_name in candidate_preset[preset_key]:
				if layer_name in seen_names[preset_key]:
					continue
				seen_names[preset_key].add(layer_name)
				merged_preset[preset_key].append(layer_name)
	return {key: tuple(names) for key, names in merged_preset.items()}


def _resolve_model_type_and_preset(model_type, diffusion_model, log_progress):
	if model_type == AUTO_MODEL_TYPE:
		detected_model_type = _infer_model_type_from_modules(diffusion_model)
		if detected_model_type is None:
			logging.warning(
				"Quantization Model Adapter: auto model_type could not identify this model; "
				"using the conservative union quantization preset. Select a model_type manually for better speed."
			)
			return "auto-conservative", _get_conservative_auto_preset()

		if log_progress:
			logging.info(f"Quantization Model Adapter: auto model_type resolved to {detected_model_type}.")
		return detected_model_type, get_model_type_quantization_preset(detected_model_type)

	if model_type == NONE_MODEL_TYPE:
		return NONE_MODEL_TYPE, _empty_quantization_preset()

	return model_type, get_model_type_quantization_preset(model_type)


def _is_supported_linear(module):
	if isinstance(module, Int8TensorwiseOps.Linear):
		return False
	if getattr(module, "_is_quantized", False):
		return False
	if not _is_linear_like(module):
		return False
	weight = getattr(module, "weight", None)
	return _is_supported_weight_tensor(weight)


def _is_supported_linear_fast_unsafe(module):
	if isinstance(module, Int8TensorwiseOps.Linear):
		return False
	if getattr(module, "_is_quantized", False):
		return False
	if not _is_linear_like(module):
		return False
	weight = getattr(module, "weight", None)
	if not isinstance(weight, torch.Tensor):
		return False
	if weight.ndim != 2:
		return False
	if weight.shape[0] <= 1 or weight.shape[1] <= 1:
		return False
	return weight.dtype in (torch.float16, torch.bfloat16, torch.float32) or _is_float8_dtype(weight.dtype)


def _module_has_target_quantization(module):
	if getattr(module, "_is_quantized", False):
		return True
	return (
		getattr(module, "_quant_format", None) in ("int8_tensorwise", "convrot_w4a4")
		or getattr(module, "quant_format", None) in ("int8_tensorwise", "convrot_w4a4")
	)


def _model_has_quantized_modules(model_patcher):
	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if isinstance(object_patch_map, dict) and any(
			_module_has_target_quantization(module)
			for module in object_patch_map.values()
		):
			return True

	diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
	if diffusion_model is None:
		return False
	return any(_module_has_target_quantization(module) for module in diffusion_model.modules())


def _collect_layer_patch_keys(model_patcher, module_name):
	weight_key = _module_weight_key(module_name)
	return [
		patch_key
		for patch_key in model_patcher.patches
		if _patch_base_key(patch_key) == weight_key
	]


def _is_deferred_int8_stochastic_patch(patch_entry):
	if not isinstance(patch_entry, tuple) or len(patch_entry) < 2:
		return False
	return bool(getattr(patch_entry[1], "_int8_defer_until_quantized", False))


def _build_layer_patch_bake_plan(model_patcher, layer_patch_keys, bake_deferred=False):
	bake_patches = []
	consumed_patch_keys = []
	deferred_patch_keys = []
	remaining_patch_entries = {}

	for patch_key in layer_patch_keys:
		patch_entries = model_patcher.patches.get(patch_key, [])
		bake_entries = []
		deferred_entries = []

		for patch_entry in patch_entries:
			if _is_deferred_int8_stochastic_patch(patch_entry) and not bake_deferred:
				deferred_entries.append(patch_entry)
			else:
				bake_entries.append(patch_entry)

		if bake_entries:
			bake_patches.extend(bake_entries)
			if deferred_entries:
				deferred_patch_keys.append(patch_key)
				remaining_patch_entries[patch_key] = deferred_entries
			else:
				consumed_patch_keys.append(patch_key)
		elif deferred_entries:
			deferred_patch_keys.append(patch_key)

	return bake_patches, consumed_patch_keys, deferred_patch_keys, remaining_patch_entries


def _configure_deferred_int8_patches(model_patcher, deferred_patch_keys, q_module):
	if getattr(q_module, "_quant_format", "int8_tensorwise") == "convrot_w4a4":
		return
	from .int8_quant import INT8LoRAPatchAdapter, INT8MergedLoRAPatchAdapter, INT8WeightPatchAdapter

	weight_scale = getattr(q_module, "weight_scale", None)
	if isinstance(weight_scale, torch.Tensor):
		weight_scale = weight_scale.item() if weight_scale.numel() == 1 else weight_scale

	outlier_method = getattr(q_module, "_outlier_method", None)
	hadanorm_sigma = getattr(q_module, "hadanorm_sigma", None)

	for patch_key in deferred_patch_keys:
		patch_entries = model_patcher.patches.get(patch_key, [])
		if not patch_entries:
			continue

		updated_entries = []
		for patch_entry in patch_entries:
			if not _is_deferred_int8_stochastic_patch(patch_entry):
				updated_entries.append(patch_entry)
				continue

			strength_patch, patch_obj, strength_model, offset, function = patch_entry
			if isinstance(patch_obj, INT8MergedLoRAPatchAdapter):
				configured_patch_obj = INT8MergedLoRAPatchAdapter(
					patch_obj.patches,
					weight_scale,
					seed=patch_obj.seed,
					outlier_method=outlier_method,
					hadanorm_sigma=hadanorm_sigma,
				)
			elif isinstance(patch_obj, INT8WeightPatchAdapter):
				configured_patch_obj = INT8WeightPatchAdapter(
					patch_obj.base_adapter,
					weight_scale,
					seed=patch_obj.seed,
					outlier_method=outlier_method,
					hadanorm_sigma=hadanorm_sigma,
				)
			elif isinstance(patch_obj, INT8LoRAPatchAdapter):
				configured_patch_obj = INT8LoRAPatchAdapter(
					patch_obj.loaded_keys,
					patch_obj.lora_weights,
					weight_scale,
					seed=patch_obj.seed,
					outlier_method=outlier_method,
					hadanorm_sigma=hadanorm_sigma,
				)
			else:
				updated_entries.append(patch_entry)
				continue

			updated_entries.append((strength_patch, configured_patch_obj, strength_model, offset, function))

		model_patcher.patches[patch_key] = updated_entries


def _get_source_weight(model_patcher, module_name, module, bake_loaded_loras, bake_deferred=False):
	weight = _materialize_source_weight(module.weight)
	weight_key = _module_weight_key(module_name)
	layer_patch_keys = _collect_layer_patch_keys(model_patcher, module_name)

	if not bake_loaded_loras or not layer_patch_keys:
		return weight, [], [], {}

	bake_patches, consumed_patch_keys, deferred_patch_keys, remaining_patch_entries = _build_layer_patch_bake_plan(
		model_patcher,
		layer_patch_keys,
		bake_deferred=bake_deferred,
	)

	if not bake_patches:
		return weight, [], deferred_patch_keys, remaining_patch_entries

	compute_device = _get_int8_compute_device(weight.device)
	try:
		intermediate_dtype = comfy.model_management.lora_compute_dtype(compute_device)
	except Exception:
		intermediate_dtype = torch.float32
	if intermediate_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
		intermediate_dtype = torch.float16

	work_weight = _materialize_source_weight(weight, device=compute_device, dtype=intermediate_dtype)
	patched_weight = comfy.lora.calculate_weight(
		bake_patches,
		work_weight,
		weight_key,
		intermediate_dtype=intermediate_dtype,
	)
	return patched_weight.detach(), consumed_patch_keys, deferred_patch_keys, remaining_patch_entries


def _quantize_linear_module(module_name, module, source_weight, quantization_mode, int4_sensitive=False):
	compute_device = _get_int8_compute_device(source_weight.device)
	weight_work = tensor_to_int8_compute_device(source_weight, compute_device, non_blocking=True)
	quantization_mode = normalize_quantization_mode(quantization_mode)
	int4_mode = quantization_mode_is_int4(quantization_mode)
	outlier_method = quantization_mode_outlier_method(quantization_mode)

	if _is_float8_dtype(weight_work.dtype):
		weight_work = weight_work.to(torch.float16 if compute_device.type == "cuda" else torch.float32)

	use_int4 = (
		int4_mode
		and not (quantization_mode == QUANTIZATION_MODE_INT4_MIXED and int4_sensitive)
		and weight_work.ndim == 2
		and weight_work.shape[1] % 256 == 0
	)
	if use_int4:
		q_weight = quantize_native_int4(weight_work, linear_dtype="int4").to("cpu")
		q_module = Int8TensorwiseOps.Linear(
			module.in_features,
			module.out_features,
			bias=module.bias is not None,
			device=torch.device("meta"),
		)
		q_module.weight = nn.Parameter(q_weight, requires_grad=False)
		q_module.weight_scale = q_weight._params.scale
		q_module.weight_packed = None
		q_module.weight_int_mm = None
		q_module._is_quantized = True
		q_module._quant_format = "convrot_w4a4"
		q_module._is_per_row = True
		q_module._use_quarot = False
		q_module._outlier_method = OUTLIER_METHOD_NONE
		q_module._convrot_groupsize = q_weight._params.convrot_groupsize
		q_module._quant_group_size = q_weight._params.quant_group_size
		q_module._linear_dtype = q_weight._params.linear_dtype
		q_module.quarot_hadamard = None
		q_module.hadanorm_sigma = None
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
		return q_module, True

	if int4_mode:
		outlier_method = OUTLIER_METHOD_CONVROT if weight_work.shape[1] % 256 == 0 else OUTLIER_METHOD_NONE
	use_quarot = outlier_method == OUTLIER_METHOD_QUAROT
	group_size = _get_outlier_group_size(outlier_method)

	if outlier_method != OUTLIER_METHOD_NONE and weight_work.dtype in (torch.float16, torch.bfloat16):
		weight_work = weight_work.float()

	quarot_hadamard = None
	hadanorm_sigma = None
	if (
		_outlier_method_uses_hadamard(outlier_method)
		and weight_work.ndim == 2
		and weight_work.shape[1] % group_size == 0
	):
		try:
			h_matrix = _build_outlier_hadamard(outlier_method, group_size, device=compute_device, dtype=weight_work.dtype)
			if h_matrix is None:
				raise RuntimeError(f"{outlier_method} Hadamard helper is unavailable")
			if outlier_method == OUTLIER_METHOD_HADANORM:
				hadanorm_sigma = _compute_hadanorm_sigma(weight_work).to(device=compute_device, dtype=weight_work.dtype)
				weight_work = weight_work * hadanorm_sigma.view(1, -1)
			weight_work = _rotate_weight_for_outlier_method(weight_work, h_matrix, group_size=group_size, method=outlier_method)
			quarot_hadamard = h_matrix.detach().cpu()
		except Exception as e:
			logging.warning(f"Quantization Model Adapter: {outlier_method} skipped for {module_name} ({e}).")
			use_quarot = False
			outlier_method = OUTLIER_METHOD_NONE

	if quarot_hadamard is None and not isinstance(hadanorm_sigma, torch.Tensor) and not use_quarot:
		outlier_method = OUTLIER_METHOD_NONE

	q_weight, q_scale = quantize_int8_rowwise(weight_work)
	q_module = Int8TensorwiseOps.Linear(
		module.in_features,
		module.out_features,
		bias=module.bias is not None,
		device=torch.device("meta"),
	)
	q_module.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
	q_module.weight_packed = q_module.weight.detach().T.contiguous() if Int8TensorwiseOps.prepack_int8_weights else None
	q_module.weight_int_mm = _prepack_torch_int_mm_weight(q_module.weight)
	q_module.weight_scale = (
		q_scale.cpu()
		if isinstance(q_scale, torch.Tensor)
		else torch.tensor([float(q_scale)], dtype=torch.float32)
	)
	q_module._is_quantized = True
	q_module._quant_format = "int8_tensorwise"
	q_module._is_per_row = q_module.weight_scale.dim() == 2 and q_module.weight_scale.shape[1] == 1
	q_module._use_quarot = use_quarot
	q_module.quarot_hadamard = quarot_hadamard
	q_module.hadanorm_sigma = hadanorm_sigma.detach().cpu() if isinstance(hadanorm_sigma, torch.Tensor) else None
	q_module._outlier_method = outlier_method
	q_module.compute_dtype = getattr(module, "compute_dtype", torch.bfloat16)
	q_module.dynamic_lora_entries = None
	q_module.lora_A = None
	q_module.lora_B = None
	q_module.lora_alpha = None
	configure_int8_module_runtime(
		q_module,
		small_batch_fallback=Int8TensorwiseOps.small_batch_fallback_mode,
		runtime_backend=Int8TensorwiseOps.runtime_backend,
		prepack_int8_weights=Int8TensorwiseOps.prepack_int8_weights,
	)

	if module.bias is not None:
		q_module.bias = nn.Parameter(module.bias.detach().cpu(), requires_grad=False)
	else:
		q_module.bias = None

	q_module.train(module.training)
	return q_module, outlier_method != OUTLIER_METHOD_NONE


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


def _validate_anima_text_id_dtype(kwargs, adapter_state):
	if not isinstance(adapter_state, dict) or adapter_state.get("model_type") != "anima":
		return

	text_ids = kwargs.get("t5xxl_ids", None)
	if isinstance(text_ids, torch.Tensor) and text_ids.dtype not in (torch.int, torch.long):
		raise RuntimeError(
			"Quantization Model Adapter: Anima text ids arrived with dtype "
			f"{text_ids.dtype}, but Anima requires integer token ids for its embedding layer. "
			"This is usually caused by a conditioning node applying math to the entire conditioning "
			"payload, including t5xxl_ids. RES4LYF ConditioningMultiply is known to do this. Remove "
			"that node for Anima or use conditioning math that leaves token-id tensors unchanged."
		)


def _int8_model_adapter_notice_wrapper(executor, *args, **kwargs):
	transformer_options = _extract_transformer_options(args, kwargs)
	adapter_state = transformer_options.get("int8_model_adapter", None)
	base_model = executor.class_obj
	diffusion_model = getattr(base_model, "diffusion_model", None)

	if isinstance(adapter_state, dict):
		runtime_backend = adapter_state.get("runtime_backend", DEFAULT_INT8_BACKEND)
		if runtime_backend not in INT8_BACKEND_CHOICES:
			runtime_backend = DEFAULT_INT8_BACKEND
		Int8TensorwiseOps.runtime_backend = runtime_backend
		Int8TensorwiseOps.runtime_uses_triton = runtime_backend in (INT8_BACKEND_TRITON, INT8_BACKEND_TRITON_LEGACY_UNSAFE)
		Int8TensorwiseOps.runtime_uses_legacy_triton = runtime_backend == INT8_BACKEND_TRITON_LEGACY_UNSAFE
		Int8TensorwiseOps.small_batch_fallback_mode = adapter_state.get("small_batch_fallback", DEFAULT_SMALL_BATCH_FALLBACK)
		Int8TensorwiseOps.prepack_int8_weights = bool(adapter_state.get("prepack_int8_weights", False))

	if isinstance(adapter_state, dict) and adapter_state.get("log_progress") and diffusion_model is not None:
		if getattr(diffusion_model, "_int8_model_adapter_skip_cache_notice_once", False):
			diffusion_model._int8_model_adapter_skip_cache_notice_once = False
			diffusion_model._int8_model_adapter_notice_in_generation = True
		elif _is_first_sampling_step(transformer_options):
			if not getattr(diffusion_model, "_int8_model_adapter_notice_in_generation", False):
				logging.info(
					"Quantization Model Adapter: reusing cached INT8 MODEL output "
					f"(quantized_layers={adapter_state.get('quantized_layers', '?')}, "
					f"model_type={adapter_state.get('model_type', '?')}, "
					f"backend={adapter_state.get('runtime_backend', '?')}, "
					f"small_batch_fallback={adapter_state.get('small_batch_fallback', '?')}, "
					f"prepack_int8_weights={adapter_state.get('prepack_int8_weights', '?')})."
				)
				Int8TensorwiseOps.reset_runtime_stats()
				diffusion_model._int8_model_adapter_notice_in_generation = True
		else:
			diffusion_model._int8_model_adapter_notice_in_generation = False

	_validate_anima_text_id_dtype(kwargs, adapter_state)
	result = executor(*args, **kwargs)
	if isinstance(adapter_state, dict):
		Int8TensorwiseOps.log_runtime_stats()
	return result


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


def _drop_torch_compile_wrapper(model_patcher):
	if not _TORCH_COMPILE_HELPER_AVAILABLE:
		return False

	compile_kwargs = model_patcher.model_options.pop(comfy_torch_compile.TORCH_COMPILE_KWARGS, None)
	model_patcher.remove_wrappers_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		comfy_torch_compile.COMPILE_KEY,
	)
	return isinstance(compile_kwargs, dict)


def _get_lora_signature(model_patcher):
	if not hasattr(model_patcher, "get_attachment"):
		return None
	signature = model_patcher.get_attachment(_INT8_LORA_SIGNATURE_ATTACHMENT_KEY)
	return signature if isinstance(signature, tuple) else None


def _can_cache_adapter_output(model_patcher, lora_signature):
	if _INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT <= 0:
		return False
	if lora_signature is not None:
		return True
	return len(getattr(model_patcher, "patches", {})) == 0


def _build_adapter_cache_key(
	model_patcher,
	resolved_model_type,
	model_type,
	quantization_mode,
	int4_mixed_ratio,
	small_batch_fallback,
	runtime_backend,
	prepack_int8_weights,
	bake_loaded_loras,
	fast_unsafe_targeting,
	lora_signature,
):
	if lora_signature is None:
		lora_signature = ("no_lora_patches",)

	return (
		"v5",
		id(getattr(model_patcher, "model", None)),
		tuple(lora_signature),
		str(resolved_model_type),
		str(model_type),
		str(quantization_mode),
		float(int4_mixed_ratio),
		str(small_batch_fallback),
		str(runtime_backend),
		bool(prepack_int8_weights),
		bool(bake_loaded_loras),
		bool(fast_unsafe_targeting),
	)


def _get_output_cache(shared_model):
	cache = getattr(shared_model, _INT8_MODEL_ADAPTER_OUTPUT_CACHE_KEY, None)
	if not isinstance(cache, dict):
		cache = {}
		setattr(shared_model, _INT8_MODEL_ADAPTER_OUTPUT_CACHE_KEY, cache)
	return cache


def _make_model_ref(shared_model):
	try:
		return weakref.ref(shared_model)
	except TypeError:
		return lambda: None


def _cleanup_adapter_cache_memory():
	try:
		import gc
		gc.collect()
	except Exception:
		pass
	_cleanup_torch_memory()


def _dispose_cached_adapter_output(model_patcher):
	unpatch_model = getattr(model_patcher, "unpatch_model", None)
	if not callable(unpatch_model):
		return
	try:
		unpatch_model(unpatch_weights=True)
	except Exception as e:
		logging.warning(f"Quantization Model Adapter: failed to unpatch an evicted cached output ({e}).")


def _prepare_model_cache(shared_model):
	global _INT8_MODEL_ADAPTER_LAST_MODEL_ID, _INT8_MODEL_ADAPTER_LAST_MODEL_REF

	shared_model_id = id(shared_model)
	prior_model = _INT8_MODEL_ADAPTER_LAST_MODEL_REF() if callable(_INT8_MODEL_ADAPTER_LAST_MODEL_REF) else None
	model_changed = _INT8_MODEL_ADAPTER_LAST_MODEL_ID is not None and _INT8_MODEL_ADAPTER_LAST_MODEL_ID != shared_model_id
	if model_changed:
		if prior_model is not None:
			prior_cache = getattr(prior_model, _INT8_MODEL_ADAPTER_OUTPUT_CACHE_KEY, None)
			if isinstance(prior_cache, dict):
				for cached_model_patcher in prior_cache.values():
					_dispose_cached_adapter_output(cached_model_patcher)
				prior_cache.clear()
		_cleanup_adapter_cache_memory()

	_INT8_MODEL_ADAPTER_LAST_MODEL_REF = _make_model_ref(shared_model)
	_INT8_MODEL_ADAPTER_LAST_MODEL_ID = shared_model_id
	return _get_output_cache(shared_model)


def _normalize_runtime_backend(runtime_backend):
	return runtime_backend if runtime_backend in INT8_BACKEND_CHOICES else DEFAULT_INT8_BACKEND


def _normalize_small_batch_fallback(small_batch_fallback):
	return small_batch_fallback if small_batch_fallback in SMALL_BATCH_FALLBACK_CHOICES else DEFAULT_SMALL_BATCH_FALLBACK


def _get_current_int8_runtime_settings(existing_modules=None):
	if existing_modules:
		for _module_name, module in existing_modules:
			if not isinstance(module, Int8TensorwiseOps.Linear):
				continue
			return (
				_normalize_small_batch_fallback(
					getattr(module, "_small_batch_fallback_mode", DEFAULT_SMALL_BATCH_FALLBACK)
				),
				_normalize_runtime_backend(
					getattr(module, "_runtime_backend", DEFAULT_INT8_BACKEND)
				),
				bool(getattr(module, "_prepack_int8_weights", False)),
			)
	return (
		_normalize_small_batch_fallback(
			getattr(Int8TensorwiseOps, "small_batch_fallback_mode", DEFAULT_SMALL_BATCH_FALLBACK)
		),
		_normalize_runtime_backend(
			getattr(Int8TensorwiseOps, "runtime_backend", DEFAULT_INT8_BACKEND)
		),
		bool(getattr(Int8TensorwiseOps, "prepack_int8_weights", False)),
	)


def _apply_int8_runtime_settings(small_batch_fallback, runtime_backend, prepack_int8_weights):
	runtime_backend = _normalize_runtime_backend(runtime_backend)
	small_batch_fallback = _normalize_small_batch_fallback(small_batch_fallback)
	Int8TensorwiseOps.small_batch_fallback_mode = small_batch_fallback
	Int8TensorwiseOps.runtime_backend = runtime_backend
	Int8TensorwiseOps.runtime_uses_triton = runtime_backend in (INT8_BACKEND_TRITON, INT8_BACKEND_TRITON_LEGACY_UNSAFE)
	Int8TensorwiseOps.runtime_uses_legacy_triton = runtime_backend == INT8_BACKEND_TRITON_LEGACY_UNSAFE
	Int8TensorwiseOps.prepack_int8_weights = bool(prepack_int8_weights)


def _remember_cached_output(shared_model, cache_key, model_patcher):
	cache = _get_output_cache(shared_model)
	cache[cache_key] = model_patcher
	while len(cache) > _INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT:
		old_key = next(iter(cache))
		if old_key == cache_key and len(cache) > 1:
			old_key = next(key for key in cache if key != cache_key)
		_dispose_cached_adapter_output(cache.pop(old_key))
	_cleanup_adapter_cache_memory()


def _make_output_cache_room(cache):
	if _INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT <= 0:
		return

	evicted = False
	while len(cache) >= _INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT:
		_dispose_cached_adapter_output(cache.pop(next(iter(cache))))
		evicted = True
	if evicted:
		_cleanup_adapter_cache_memory()


def _collect_int8_candidates(diffusion_model, keep_float_names, fast_unsafe=False):
	is_supported = _is_supported_linear_fast_unsafe if fast_unsafe else _is_supported_linear
	return [
		(module_name, module)
		for module_name, module in diffusion_model.named_modules()
		if module_name
		and not _matches_layer_patterns(module_name, keep_float_names)
		and is_supported(module)
	]


def _collect_existing_int8_modules(diffusion_model, keep_float_names):
	return [
		(module_name, module)
		for module_name, module in diffusion_model.named_modules()
		if module_name
		and not _matches_layer_patterns(module_name, keep_float_names)
		and isinstance(module, Int8TensorwiseOps.Linear)
		and getattr(module, "_is_quantized", False)
	]


def _get_int8_patch_weight_scale(q_module):
	weight_scale = getattr(q_module, "weight_scale", None)
	if isinstance(weight_scale, torch.Tensor):
		return weight_scale.item() if weight_scale.numel() == 1 else weight_scale
	return weight_scale


def _wrap_existing_int8_patch(q_module, patch_obj, seed):
	if not _WEIGHT_ADAPTER_AVAILABLE or not isinstance(patch_obj, WeightAdapterBase):
		return patch_obj, False
	if getattr(q_module, "_quant_format", "int8_tensorwise") == "convrot_w4a4":
		return patch_obj, False

	from .int8_quant import INT8LoRAPatchAdapter, INT8MergedLoRAPatchAdapter, INT8WeightPatchAdapter

	weight_scale = _get_int8_patch_weight_scale(q_module)
	outlier_method = getattr(q_module, "_outlier_method", None)
	hadanorm_sigma = getattr(q_module, "hadanorm_sigma", None)

	if isinstance(patch_obj, INT8MergedLoRAPatchAdapter):
		return (
			INT8MergedLoRAPatchAdapter(
				patch_obj.patches,
				weight_scale,
				seed=patch_obj.seed,
				outlier_method=outlier_method,
				hadanorm_sigma=hadanorm_sigma,
			),
			True,
		)

	if isinstance(patch_obj, INT8WeightPatchAdapter):
		return (
			INT8WeightPatchAdapter(
				patch_obj.base_adapter,
				weight_scale,
				seed=patch_obj.seed,
				outlier_method=outlier_method,
				hadanorm_sigma=hadanorm_sigma,
			),
			True,
		)

	if isinstance(patch_obj, INT8LoRAPatchAdapter):
		return (
			INT8LoRAPatchAdapter(
				patch_obj.loaded_keys,
				patch_obj.lora_weights,
				weight_scale,
				seed=patch_obj.seed,
				outlier_method=outlier_method,
				hadanorm_sigma=hadanorm_sigma,
			),
			True,
		)

	if isinstance(patch_obj, LoRAAdapter):
		return (
			INT8LoRAPatchAdapter(
				patch_obj.loaded_keys,
				patch_obj.weights,
				weight_scale,
				seed=seed,
				outlier_method=outlier_method,
				hadanorm_sigma=hadanorm_sigma,
			),
			True,
		)

	return (
		INT8WeightPatchAdapter(
			patch_obj,
			weight_scale,
			seed=seed,
			outlier_method=outlier_method,
			hadanorm_sigma=hadanorm_sigma,
		),
		True,
	)


def _set_existing_int8_weight(q_module, new_weight):
	if getattr(q_module, "_quant_format", "int8_tensorwise") == "convrot_w4a4":
		q_module.set_weight(new_weight, seed=comfy.utils.string_to_seed("int4_existing_patch"))
		return
	if hasattr(q_module, "_replace_weight"):
		q_module._replace_weight(new_weight)
	else:
		q_module.weight = nn.Parameter(new_weight, requires_grad=False)
		q_module.weight_packed = (
			q_module.weight.detach().T.contiguous()
			if getattr(q_module, "_prepack_int8_weights", Int8TensorwiseOps.prepack_int8_weights)
			else None
		)
		q_module.weight_int_mm = _prepack_torch_int_mm_weight(q_module.weight)


def _clone_existing_int8_module(q_module):
	cloned_module = Int8TensorwiseOps.Linear(
		q_module.in_features,
		q_module.out_features,
		bias=q_module.bias is not None,
		device=torch.device("meta"),
	)
	cloned_module.weight = nn.Parameter(q_module.weight.detach().clone(), requires_grad=False)
	cloned_module._quant_format = getattr(q_module, "_quant_format", "int8_tensorwise")
	cloned_module.weight_scale = (
		q_module.weight_scale.detach().clone()
		if isinstance(q_module.weight_scale, torch.Tensor)
		else q_module.weight_scale
	)
	cloned_module.weight_packed = (
		cloned_module.weight.detach().T.contiguous()
		if cloned_module._quant_format == "int8_tensorwise"
		and getattr(q_module, "_prepack_int8_weights", Int8TensorwiseOps.prepack_int8_weights)
		else None
	)
	cloned_module.weight_int_mm = (
		_prepack_torch_int_mm_weight(cloned_module.weight)
		if cloned_module._quant_format == "int8_tensorwise"
		else None
	)
	cloned_module._convrot_groupsize = getattr(q_module, "_convrot_groupsize", 256)
	cloned_module._quant_group_size = getattr(q_module, "_quant_group_size", 64)
	cloned_module._linear_dtype = getattr(q_module, "_linear_dtype", "int4")
	cloned_module.quarot_hadamard = (
		q_module.quarot_hadamard.detach().clone()
		if isinstance(q_module.quarot_hadamard, torch.Tensor)
		else None
	)
	cloned_module.hadanorm_sigma = (
		q_module.hadanorm_sigma.detach().clone()
		if isinstance(q_module.hadanorm_sigma, torch.Tensor)
		else None
	)
	cloned_module._is_quantized = bool(getattr(q_module, "_is_quantized", False))
	cloned_module._is_per_row = bool(getattr(q_module, "_is_per_row", False))
	cloned_module._use_quarot = bool(getattr(q_module, "_use_quarot", False))
	cloned_module._outlier_method = getattr(q_module, "_outlier_method", OUTLIER_METHOD_NONE)
	cloned_module.compute_dtype = getattr(q_module, "compute_dtype", torch.bfloat16)
	cloned_module.dynamic_lora_entries = None
	cloned_module.lora_A = None
	cloned_module.lora_B = None
	cloned_module.lora_alpha = None
	configure_int8_module_runtime(
		cloned_module,
		small_batch_fallback=getattr(q_module, "_small_batch_fallback_mode", Int8TensorwiseOps.small_batch_fallback_mode),
		runtime_backend=getattr(q_module, "_runtime_backend", Int8TensorwiseOps.runtime_backend),
		prepack_int8_weights=getattr(q_module, "_prepack_int8_weights", Int8TensorwiseOps.prepack_int8_weights),
	)

	if q_module.bias is not None:
		cloned_module.bias = nn.Parameter(q_module.bias.detach().clone(), requires_grad=False)
	else:
		cloned_module.bias = None

	cloned_module.train(q_module.training)
	return cloned_module


def _configure_existing_int8_patches(model_patcher, existing_modules, bake_loaded_loras):
	configured_count = 0
	baked_count = 0

	for module_name, q_module in existing_modules:
		layer_patch_keys = _collect_layer_patch_keys(model_patcher, module_name)
		if not layer_patch_keys:
			continue

		target_module = q_module
		bake_module = None
		_configure_deferred_int8_patches(model_patcher, layer_patch_keys, q_module)
		for patch_key in layer_patch_keys:
			patch_entries = model_patcher.patches.get(patch_key, [])
			if not patch_entries:
				continue

			updated_entries = []
			patch_seed = comfy.utils.string_to_seed(str(_patch_base_key(patch_key)))
			for patch_entry in patch_entries:
				if not isinstance(patch_entry, tuple) or len(patch_entry) < 5:
					updated_entries.append(patch_entry)
					continue

				strength_patch, patch_obj, strength_model, offset, function = patch_entry
				if _is_deferred_int8_stochastic_patch(patch_entry):
					updated_entries.append(patch_entry)
					continue

				wrapped_patch_obj, was_wrapped = _wrap_existing_int8_patch(target_module, patch_obj, patch_seed)
				if was_wrapped:
					configured_count += 1
				updated_entries.append((strength_patch, wrapped_patch_obj, strength_model, offset, function))

			if bake_loaded_loras:
				if bake_module is None:
					bake_module = _clone_existing_int8_module(q_module)
					target_module = bake_module
					model_patcher.add_object_patch(_module_patch_key(module_name), bake_module)

				weight = getattr(target_module, "weight", None)
				if not isinstance(weight, torch.Tensor):
					model_patcher.patches[patch_key] = updated_entries
					continue

				compute_device = _get_int8_compute_device(weight.device)
				try:
					intermediate_dtype = comfy.model_management.lora_compute_dtype(compute_device)
				except Exception:
					intermediate_dtype = torch.float32
				if _is_float8_dtype(intermediate_dtype):
					intermediate_dtype = torch.float16

				try:
					work_weight = (
						target_module.convert_weight(weight.detach().clone())
						if getattr(target_module, "_quant_format", "int8_tensorwise") == "convrot_w4a4"
						else weight.detach().clone()
					)
					baked_weight = comfy.lora.calculate_weight(
						updated_entries,
						work_weight,
						_module_weight_key(module_name),
						intermediate_dtype=intermediate_dtype,
					)
					_set_existing_int8_weight(target_module, baked_weight.detach())
					model_patcher.patches.pop(patch_key, None)
					baked_count += len(updated_entries)
				except Exception as e:
					logging.warning(f"Quantization Model Adapter: failed to bake existing INT8 patches for {module_name} ({e}).")
					model_patcher.patches[patch_key] = updated_entries
			else:
				model_patcher.patches[patch_key] = updated_entries

	return configured_count, baked_count


def _get_original_module_cache(model_patcher):
	shared_model = getattr(model_patcher, "model", None)
	original_module_cache = getattr(shared_model, _INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY, None)
	if isinstance(original_module_cache, dict):
		model_patcher.set_attachments(_INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY, original_module_cache)
		return original_module_cache

	original_module_cache = model_patcher.get_attachment(_INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY)
	if isinstance(original_module_cache, dict):
		if shared_model is not None:
			setattr(shared_model, _INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY, original_module_cache)
		return original_module_cache

	original_module_cache = {}
	if shared_model is not None:
		setattr(shared_model, _INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY, original_module_cache)
	model_patcher.set_attachments(_INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY, original_module_cache)
	return original_module_cache


def _remember_original_linear_modules(model_patcher, candidates):
	original_module_cache = _get_original_module_cache(model_patcher)

	for module_name, module in candidates:
		if isinstance(module, Int8TensorwiseOps.Linear):
			continue

		patch_key = _module_patch_key(module_name)
		original_module_cache.setdefault(patch_key, module)


def _clear_prior_int8_object_patches(model_patcher):
	for patch_key, patch_obj in list(model_patcher.object_patches.items()):
		if patch_key.startswith("diffusion_model.") and isinstance(patch_obj, Int8TensorwiseOps.Linear):
			model_patcher.object_patches.pop(patch_key, None)


def _reset_prior_int8_object_patches(model_patcher):
	int8_patch_keys = set()
	original_module_cache = _get_original_module_cache(model_patcher)

	for patch_key, patch_obj in list(model_patcher.object_patches_backup.items()):
		if not patch_key.startswith("diffusion_model."):
			continue
		if isinstance(patch_obj, Int8TensorwiseOps.Linear):
			continue
		original_module_cache.setdefault(patch_key, patch_obj)

	for patch_key, patch_obj in list(model_patcher.object_patches.items()):
		if patch_key.startswith("diffusion_model.") and isinstance(patch_obj, Int8TensorwiseOps.Linear):
			int8_patch_keys.add(patch_key)

	for patch_key in list(model_patcher.object_patches_backup.keys()):
		if patch_key.startswith("diffusion_model."):
			int8_patch_keys.add(patch_key)

	if int8_patch_keys:
		try:
			model_patcher.unpatch_model(unpatch_weights=True)
		except Exception as e:
			logging.warning(f"Quantization Model Adapter: failed to fully reset prior INT8 object patches ({e}).")

	for patch_key, original_module in list(original_module_cache.items()):
		try:
			current_module = comfy.utils.get_attr(model_patcher.model, patch_key)
		except Exception:
			continue

		if current_module is not original_module:
			comfy.utils.set_attr(model_patcher.model, patch_key, original_module)
			int8_patch_keys.add(patch_key)

	for patch_key in list(model_patcher.object_patches.keys()):
		if patch_key in int8_patch_keys:
			model_patcher.object_patches.pop(patch_key, None)

	for patch_key in list(model_patcher.object_patches_backup.keys()):
		if patch_key in int8_patch_keys:
			model_patcher.object_patches_backup.pop(patch_key, None)

	return len(int8_patch_keys)


class INT8ModelAdapter:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL", {"tooltip": "The stock-loaded diffusion model to convert to the selected quantization mode."}),
				"enable_quantization": (QUANTIZATION_CONTROL_CHOICES, {"default": QUANTIZATION_CONTROL_AS_NEEDED, "tooltip": "as_needed converts FP8 and floating-point inputs, but leaves MODEL inputs containing Toolkit-supported INT8 or W4A4 layers unchanged. always converts remaining eligible layers; bypass returns the MODEL unchanged."}),
				"model_type": (MODEL_TYPE_CHOICES, {"default": AUTO_MODEL_TYPE, "tooltip": "Architecture preset. Known quality-sensitive or unsafe layers remain floating-point in every mode. Auto inspects the loaded MODEL. flux2_fast_unsafe is opt-in and uses less defensive targeting. Use none only for experiments."}),
				"quantization_mode": (QUANTIZATION_MODE_CHOICES, {"default": DEFAULT_QUANTIZATION_MODE, "tooltip": "Quantization mode. Both INT4 modes preserve keep-float layers. int4_mixed keeps a configurable fraction of compatible linears in ConvRot INT8; int4_full uses W4A4 whenever shapes permit."}),
				"int4_mixed_ratio": ("FLOAT", {"default": DEFAULT_INT4_MIXED_RATIO, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of W4-compatible eligible linears kept in ConvRot INT8 when using int4_mixed. Architecture-specific patterns are prioritized; the remaining budget is distributed deterministically across the model. 0 matches int4_full layer selection and 1 keeps all compatible linears in INT8."}),
				"small_batch_fallback": (SMALL_BATCH_FALLBACK_CHOICES, {"default": DEFAULT_SMALL_BATCH_FALLBACK, "tooltip": "Controls the fp16/bf16 fallback for very small activation batches. only_small_layers is the default and limits fallback to layers with out_features * in_features <= INT8_SMALL_LAYER_MAX_PARAMS, default 1,000,000; always can help tiny row counts but often slows larger layers by dequantizing full weights; never forces the INT8 backend."}),
				"runtime_backend": (INT8_BACKEND_CHOICES, {"default": DEFAULT_INT8_BACKEND, "tooltip": "Backend for non-ConvRot INT8 linear layers. int8_convrot always uses Comfy-Kitchen's native fused runtime. torch_int_mm is the default for other INT8 modes; triton may be faster on some shapes; triton_legacy_unsafe is diagnostic only and may be incorrect on tail shapes."}),
				"prepack_weights": ("BOOLEAN", {"default": False, "tooltip": "Experimental runtime weight prepacking. This currently applies only to Triton INT8 layers, where it keeps an extra transposed weight buffer so output columns are read contiguously. It may improve speed but adds roughly one extra INT8 copy of each affected weight."}),
				"bake_loaded_loras": ("BOOLEAN", {"default": True, "tooltip": "Apply existing stock LoRA weight patches, including sliced patches, before quantization, then remove the consumed patches to avoid applying them twice. If disabled, layers with pending patches are left unquantized."}),
				"log_progress": ("BOOLEAN", {"default": True, "tooltip": "Log quantization progress and layer counts to the ComfyUI console."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "apply_quantization"
	CATEGORY = "loaders"
	DESCRIPTION = "Convert a stock-loaded diffusion MODEL to INT8 or native ConvRot INT4. Put this after stock Load LoRA to bake loaded LoRAs before quantization."

	def apply_quantization(
		self,
		model,
		enable_quantization,
		model_type,
		quantization_mode,
		int4_mixed_ratio=DEFAULT_INT4_MIXED_RATIO,
		small_batch_fallback=DEFAULT_SMALL_BATCH_FALLBACK,
		runtime_backend=DEFAULT_INT8_BACKEND,
		prepack_weights=False,
		bake_loaded_loras=True,
		log_progress=True,
		use_triton=None,
		):
		if enable_quantization not in QUANTIZATION_CONTROL_CHOICES:
			raise ValueError(f"Invalid enable_quantization value {enable_quantization}")
		if enable_quantization == QUANTIZATION_CONTROL_BYPASS:
			return (model,)
		if (
			enable_quantization == QUANTIZATION_CONTROL_AS_NEEDED
			and _model_has_quantized_modules(model)
		):
			if log_progress:
				logging.info("Quantization Model Adapter: input MODEL is already quantized; returning it unchanged.")
			return (model,)

		quantization_mode = normalize_quantization_mode(quantization_mode)
		int4_mixed_ratio = normalize_int4_mixed_ratio(int4_mixed_ratio)
		effective_int4_mixed_ratio = (
			int4_mixed_ratio
			if quantization_mode == QUANTIZATION_MODE_INT4_MIXED
			else 0.0
		)
		if quantization_mode_is_int4(quantization_mode) and not native_int4_available():
			raise RuntimeError("INT4 quantization requires a recent ComfyUI and comfy-kitchen with ConvRot W4A4 support")

		source_model_patcher = model
		output_cache = _prepare_model_cache(source_model_patcher.model)
		source_diffusion_model = getattr(source_model_patcher.model, "diffusion_model", None)
		if source_diffusion_model is None:
			logging.warning("Quantization Model Adapter: model has no diffusion_model; returning unchanged model.")
			return (source_model_patcher,)

		resolved_model_type, quantization_preset = _resolve_model_type_and_preset(
			model_type,
			source_diffusion_model,
			bool(log_progress),
		)
		keep_float_names = quantization_preset["keep_float"]
		int4_sensitive_names = quantization_preset["int4_sensitive"]
		fast_unsafe_targeting = resolved_model_type == MODEL_TYPE_FLUX2_FAST_UNSAFE
		small_batch_fallback = _normalize_small_batch_fallback(small_batch_fallback)
		runtime_backend = _normalize_runtime_backend(runtime_backend)

		source_existing_int8_modules = _collect_existing_int8_modules(source_diffusion_model, keep_float_names)
		prior_small_batch_fallback, prior_runtime_backend, prior_prepack_int8_weights = _get_current_int8_runtime_settings(source_existing_int8_modules)
		source_has_quantizable_candidates = bool(_collect_int8_candidates(
			source_diffusion_model,
			keep_float_names,
			fast_unsafe=fast_unsafe_targeting,
		))
		preserve_existing_runtime = bool(source_existing_int8_modules) and not source_has_quantizable_candidates
		effective_small_batch_fallback = prior_small_batch_fallback if preserve_existing_runtime else small_batch_fallback
		effective_runtime_backend = prior_runtime_backend if preserve_existing_runtime else runtime_backend
		effective_prepack_int8_weights = prior_prepack_int8_weights if preserve_existing_runtime else bool(prepack_weights)

		lora_signature = _get_lora_signature(source_model_patcher)
		use_output_cache = _can_cache_adapter_output(source_model_patcher, lora_signature)
		adapter_cache_key = _build_adapter_cache_key(
			source_model_patcher,
			resolved_model_type,
			model_type,
			quantization_mode,
			effective_int4_mixed_ratio,
			effective_small_batch_fallback,
			effective_runtime_backend,
			effective_prepack_int8_weights,
			bake_loaded_loras,
			fast_unsafe_targeting,
			lora_signature,
		)
		if use_output_cache:
			cached_model_patcher = output_cache.get(adapter_cache_key)
			if cached_model_patcher is not None:
				if log_progress:
					logging.info("Quantization Model Adapter: reusing cached MODEL output.")
				return (cached_model_patcher,)
			_make_output_cache_room(output_cache)

		model_patcher = source_model_patcher.clone()
		restored_prior_patch_count = _reset_prior_int8_object_patches(model_patcher)
		_clear_prior_int8_object_patches(model_patcher)
		diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
		if diffusion_model is None:
			logging.warning("Quantization Model Adapter: model has no diffusion_model; returning unchanged model.")
			return (model_patcher,)

		_apply_int8_runtime_settings(
			effective_small_batch_fallback,
			effective_runtime_backend,
			effective_prepack_int8_weights,
		)

		if log_progress and fast_unsafe_targeting:
			logging.info(
				"Quantization Model Adapter: flux2_fast_unsafe selected; using upstream-style plain nn.Linear "
				"targeting and the less conservative Flux2 keep-float preset."
			)
		if log_progress and preserve_existing_runtime and (
			effective_runtime_backend != runtime_backend
			or effective_small_batch_fallback != small_batch_fallback
			or effective_prepack_int8_weights != bool(prepack_weights)
		):
			logging.info(
				"Quantization Model Adapter: preserving existing INT8 runtime settings from loader "
				f"(backend={effective_runtime_backend}, "
				f"small_batch_fallback={effective_small_batch_fallback}, "
				f"prepack_int8_weights={effective_prepack_int8_weights})."
			)

		candidates = _collect_int8_candidates(
			diffusion_model,
			keep_float_names,
			fast_unsafe=fast_unsafe_targeting,
		)
		existing_int8_modules = _collect_existing_int8_modules(diffusion_model, keep_float_names)
		if not candidates and not existing_int8_modules:
			try:
				if log_progress:
					logging.info("Quantization Model Adapter: no eligible layers found on the first scan; forcing model load and rescanning.")
				comfy.model_management.load_models_gpu([model_patcher], force_patch_weights=True, force_full_load=True)
				diffusion_model = getattr(model_patcher.model, "diffusion_model", diffusion_model)
				candidates = _collect_int8_candidates(
					diffusion_model,
					keep_float_names,
					fast_unsafe=fast_unsafe_targeting,
				)
				existing_int8_modules = _collect_existing_int8_modules(diffusion_model, keep_float_names)
				if existing_int8_modules and not candidates and not preserve_existing_runtime:
					preserve_existing_runtime = True
					effective_small_batch_fallback = prior_small_batch_fallback
					effective_runtime_backend = prior_runtime_backend
					effective_prepack_int8_weights = prior_prepack_int8_weights
					_apply_int8_runtime_settings(
						effective_small_batch_fallback,
						effective_runtime_backend,
						effective_prepack_int8_weights,
					)
			except Exception as e:
				logging.warning(f"Quantization Model Adapter: forced model load failed during candidate scan ({e}).")

		if fast_unsafe_targeting and not candidates:
			candidates = _collect_int8_candidates(
				diffusion_model,
				keep_float_names,
				fast_unsafe=False,
			)
			if log_progress and candidates:
				logging.info(
					"Quantization Model Adapter: flux2_fast_unsafe found no raw fast candidates; "
					"falling back to Comfy linear-like targeting for quantization."
				)

		int4_compatible_names = [
			module_name
			for module_name, module in candidates
			if is_int4_compatible_linear(getattr(module, "weight", None))
		]
		int4_mixed_selected_names = set()
		if quantization_mode == QUANTIZATION_MODE_INT4_MIXED:
			int4_mixed_selected_names = set(select_int4_mixed_layers(
				int4_compatible_names,
				effective_int4_mixed_ratio,
				int4_sensitive_names,
			))
			if log_progress:
				logging.info(
					"Quantization Model Adapter: INT4 mixed profile "
					f"(compatible={len(int4_compatible_names)}, "
					f"kept_int8={len(int4_mixed_selected_names)}, "
					f"ratio={effective_int4_mixed_ratio:.2f}, "
					f"preset_patterns={len(int4_sensitive_names)})"
				)

		if candidates:
			_remember_original_linear_modules(model_patcher, candidates)

		total = len(candidates)
		quantized = 0
		existing_quantized = len(existing_int8_modules)
		quarot_count = 0
		baked_lora_count = 0
		configured_int8_patch_count, baked_existing_int8_patch_count = _configure_existing_int8_patches(
			model_patcher,
			existing_int8_modules,
			bool(bake_loaded_loras),
		)
		baked_lora_count += baked_existing_int8_patch_count
		skipped_patched_count = 0
		last_bucket = -1

		if log_progress:
			if restored_prior_patch_count:
				logging.info(f"Quantization Model Adapter: restored {restored_prior_patch_count} prior INT8 object patches before requantizing.")
			if existing_quantized:
				logging.info(
					f"Quantization Model Adapter: found {existing_quantized} existing INT8 layer(s); "
					f"baked {baked_existing_int8_patch_count} and configured "
					f"{max(0, configured_int8_patch_count - baked_existing_int8_patch_count)} pending patch(es)."
				)
			logging.info(f"Quantization Model Adapter: starting {quantization_mode} conversion (eligible linear layers: {total}).")

		for index, (module_name, module) in enumerate(candidates, start=1):
			try:
				pending_patch_keys = _collect_layer_patch_keys(model_patcher, module_name)
				if pending_patch_keys and not bake_loaded_loras:
					skipped_patched_count += 1
					continue

				source_weight, baked_patch_keys, deferred_patch_keys, remaining_patch_entries = _get_source_weight(
					model_patcher,
					module_name,
					module,
					bool(bake_loaded_loras),
					bake_deferred=quantization_mode_is_int4(quantization_mode),
				)
				q_module, used_quarot = _quantize_linear_module(
					module_name,
					module,
					source_weight,
					quantization_mode,
					int4_sensitive=module_name in int4_mixed_selected_names,
				)
				model_patcher.add_object_patch(_module_patch_key(module_name), q_module)
				for patch_key, patch_entries in remaining_patch_entries.items():
					model_patcher.patches[patch_key] = patch_entries
				_configure_deferred_int8_patches(model_patcher, deferred_patch_keys, q_module)
				quantized += 1
				if used_quarot:
					quarot_count += 1
				if baked_patch_keys:
					for patch_key in baked_patch_keys:
						model_patcher.patches.pop(patch_key, None)
					baked_lora_count += len(baked_patch_keys)
				if remaining_patch_entries:
					baked_lora_count += len(remaining_patch_entries)
				del source_weight
			except Exception as e:
				logging.warning(f"Quantization Model Adapter: skipped {module_name} ({e}).")

			if (index % 8) == 0:
				_cleanup_torch_memory()

			if log_progress and total > 0:
				percent = min(100, int((index * 100) / total))
				bucket = percent // 5
				if bucket != last_bucket:
					last_bucket = bucket
					logging.info(
						f"Quantization Model Adapter: {percent:3d}% "
						f"({index}/{total}) quantized={quantized} "
						f"baked_patches={baked_lora_count} "
						f"skipped_patched={skipped_patched_count} "
						f"outlier_adjusted={quarot_count}"
					)

		if "transformer_options" not in model_patcher.model_options:
			model_patcher.model_options["transformer_options"] = {}
		else:
			model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()

		model_patcher.model_options["transformer_options"]["int8_model_adapter"] = {
			"model_type": resolved_model_type,
			"requested_model_type": model_type,
			"bake_loaded_loras": bool(bake_loaded_loras),
			"quantization_mode": quantization_mode,
			"keep_float_patterns": tuple(keep_float_names),
			"int4_sensitive_patterns": tuple(int4_sensitive_names),
			"int4_mixed_ratio": effective_int4_mixed_ratio,
			"requested_int4_mixed_ratio": int4_mixed_ratio,
			"int4_mixed_selected_layers": tuple(sorted(int4_mixed_selected_names)),
			"small_batch_fallback": effective_small_batch_fallback,
			"runtime_backend": effective_runtime_backend,
			"prepack_int8_weights": bool(effective_prepack_int8_weights),
			"requested_small_batch_fallback": small_batch_fallback,
			"requested_runtime_backend": runtime_backend,
			"requested_prepack_weights": bool(prepack_weights),
			"preserved_existing_runtime": bool(preserve_existing_runtime),
			"fast_unsafe_targeting": bool(fast_unsafe_targeting),
			"log_progress": bool(log_progress),
			"quantized_layers": quantized + existing_quantized,
			"new_quantized_layers": quantized,
			"existing_quantized_layers": existing_quantized,
			"baked_lora_layers": baked_lora_count,
			"baked_existing_int8_patches": baked_existing_int8_patch_count,
			"configured_int8_patch_layers": configured_int8_patch_count,
			"outlier_adjusted_layers": quarot_count,
			"skipped_patched_layers": skipped_patched_count,
		}
		setattr(diffusion_model, "_int8_model_adapter_skip_cache_notice_once", True)
		compile_wrapper_removed = _drop_torch_compile_wrapper(model_patcher)
		_ensure_int8_model_adapter_notice_wrapper(model_patcher)
		if use_output_cache:
			adapter_cache_key = _build_adapter_cache_key(
				source_model_patcher,
				resolved_model_type,
				model_type,
				quantization_mode,
				effective_int4_mixed_ratio,
				effective_small_batch_fallback,
				effective_runtime_backend,
				effective_prepack_int8_weights,
				bake_loaded_loras,
				fast_unsafe_targeting,
				lora_signature,
			)
			model_patcher.patches_uuid = uuid.uuid5(uuid.NAMESPACE_URL, repr(adapter_cache_key))
			_remember_cached_output(source_model_patcher.model, adapter_cache_key, model_patcher)
		else:
			model_patcher.patches_uuid = uuid.uuid4()
		_cleanup_torch_memory()

		if quantized == 0 and existing_quantized == 0 and skipped_patched_count > 0:
			if total > 0 and skipped_patched_count == total:
				logging.warning(
					"Quantization Model Adapter: all eligible layers had pending patches and bake_loaded_loras is disabled; "
					"no layers were quantized. Enable bake_loaded_loras or remove upstream merge/LoRA patches."
				)
			else:
				logging.warning(
					f"Quantization Model Adapter: {skipped_patched_count} eligible layer(s) had pending patches and "
					"bake_loaded_loras is disabled; no layers were quantized. Enable bake_loaded_loras or remove "
					"upstream merge/LoRA patches."
				)
			logging.warning(
				"Quantization Model Adapter: no layers were converted; later dtype/runtime errors are likely "
				"outside the quantized forward path."
			)

		if log_progress:
			logging.info(
				"Quantization Model Adapter: complete "
				f"(mode={quantization_mode}, quantized={quantized}, existing_quantized={existing_quantized}, "
				f"baked_patches={baked_lora_count}, configured_int8_patches={configured_int8_patch_count}, "
				f"skipped_patched_layers={skipped_patched_count}, outlier_adjusted={quarot_count}, "
				f"backend={effective_runtime_backend}, small_batch_fallback={effective_small_batch_fallback}, "
				f"prepack_int8_weights={bool(effective_prepack_int8_weights)}, "
				f"bake_loaded_loras={bool(bake_loaded_loras)})"
			)
			if compile_wrapper_removed:
				logging.info("Quantization Model Adapter: removed the torch.compile wrapper after requantization.")

		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8ModelAdapter": INT8ModelAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8ModelAdapter": "Enable Quantization on MODEL",
}
