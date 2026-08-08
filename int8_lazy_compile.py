import functools
import gc
import importlib.metadata
import logging
import os
import sys
import time
import weakref

import comfy.patcher_extension
import comfy.utils
import torch
from torch import nn
from tqdm.auto import tqdm

from . import int4_compile_compat
from . import w4a8_compile_compat


_LAZY_COMPILE_WRAPPER_KEY = "int8_lazy_torch_compile"
_TORCH_COMPILE_KWARGS = "torch_compile_kwargs"
_WHOLE_MODEL_COMPILE_KEY_LIST = ["diffusion_model"]
_LAZY_COMPILE_OUTPUT_CACHE_KEY = "int8_lazy_compile_output_cache"
_LAZY_COMPILE_STRUCTURE_CACHE_KEY = "int8_lazy_compile_structure_cache"
_LAZY_COMPILE_LAST_MODEL_REF = None
_LAZY_COMPILE_LAST_MODEL_ID = None
_STABLE_PARTIAL_FORWARD_CLASS_CACHE = {}
_COMFY_KITCHEN_PROJECT_URL = "https://github.com/Comfy-Org/comfy-kitchen"
try:
	_LAZY_COMPILE_OUTPUT_CACHE_LIMIT = max(0, int(os.environ.get("INT8_LAZY_COMPILE_OUTPUT_CACHE_LIMIT", "1")))
except ValueError:
	_LAZY_COMPILE_OUTPUT_CACHE_LIMIT = 1
_LAZY_COMPILE_RESET_ON_MODEL_CHANGE = os.environ.get("INT8_LAZY_COMPILE_RESET_ON_MODEL_CHANGE", "1") == "1"


def _log_info_outside_progress_bar(message):
	"""Let tqdm clear and redraw ComfyUI's active sampling progress bar."""
	try:
		with tqdm.external_write_mode(file=sys.stderr):
			logging.info(message)
	except Exception:
		logging.info(message)

def _skip_transformer_options_guards(guard_entries):
	return [("transformer_options" not in entry.name) for entry in guard_entries]


def _dispatch_compiled_module(module, *args, **kwargs):
	return module(*args, **kwargs)


def _get_stable_partial_forward_class(module_class, forward_func):
	cache_key = (module_class, forward_func)
	stable_class = _STABLE_PARTIAL_FORWARD_CLASS_CACHE.get(cache_key)
	if stable_class is not None:
		return stable_class

	stable_class = type(
		f"{module_class.__name__}StablePartialForward",
		(module_class,),
		{
			"__module__": module_class.__module__,
			"forward": forward_func,
		},
	)
	_STABLE_PARTIAL_FORWARD_CLASS_CACHE[cache_key] = stable_class
	return stable_class


def _stabilize_partial_forwards(module):
	restored_forwards = []
	for child_module in module.modules():
		forward = child_module.__dict__.get("forward")
		if not isinstance(forward, functools.partial):
			continue
		if forward.keywords:
			continue
		if len(forward.args) != 1 or forward.args[0] is not child_module:
			continue

		original_class = child_module.__class__
		stable_class = _get_stable_partial_forward_class(original_class, forward.func)
		try:
			child_module.__class__ = stable_class
			del child_module.__dict__["forward"]
		except Exception:
			child_module.__class__ = original_class
			child_module.__dict__["forward"] = forward
			continue
		restored_forwards.append((child_module, original_class, forward))
	return restored_forwards


def _restore_partial_forwards(restored_forwards):
	for child_module, original_class, forward in reversed(restored_forwards):
		child_module.__class__ = original_class
		child_module.__dict__["forward"] = forward


class _CompiledModuleProxy(nn.Module):
	def __init__(
		self,
		module_key,
		source_module,
		compiled_dispatch,
		guard_failure_reporter=None,
		dispatch_reporter=None,
	):
		super().__init__()
		object.__setattr__(self, "_module_key", module_key)
		object.__setattr__(self, "_source_module", source_module)
		object.__setattr__(self, "_compiled_dispatch", compiled_dispatch)
		object.__setattr__(self, "_guard_failure_reporter", guard_failure_reporter)
		object.__setattr__(self, "_dispatch_reporter", dispatch_reporter)

	def forward(self, *args, **kwargs):
		restored_forwards = _stabilize_partial_forwards(self._source_module)
		try:
			if self._guard_failure_reporter is not None:
				self._guard_failure_reporter(self._source_module, args, kwargs)
			if self._dispatch_reporter is None:
				return self._compiled_dispatch(self._source_module, *args, **kwargs)
			start_time = time.perf_counter()
			result = self._compiled_dispatch(self._source_module, *args, **kwargs)
			self._dispatch_reporter(self._module_key, time.perf_counter() - start_time)
			return result
		finally:
			_restore_partial_forwards(restored_forwards)


def _get_dynamic_value(dynamic_shape_tracing):
	dynamic_values = {
		"auto": None,
		"true": True,
		"false": False,
	}
	if dynamic_shape_tracing not in dynamic_values:
		raise ValueError(f"Invalid dynamic_shape_tracing value {dynamic_shape_tracing}")
	return dynamic_values[dynamic_shape_tracing]


def _get_model_inference_dtype(model_patcher):
	base_model = getattr(model_patcher, "model", None)
	get_dtype_inference = getattr(base_model, "get_dtype_inference", None)
	if callable(get_dtype_inference):
		try:
			dtype = get_dtype_inference()
		except Exception:
			dtype = None
		if isinstance(dtype, torch.dtype):
			return dtype

	model_dtype = getattr(model_patcher, "model_dtype", None)
	if callable(model_dtype):
		try:
			dtype = model_dtype()
		except Exception:
			dtype = None
		if isinstance(dtype, torch.dtype):
			return dtype
	return None


def _get_object_patch_module_names(model_patcher):
	module_names = set()
	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if not isinstance(object_patch_map, dict):
			continue
		for patch_key in object_patch_map:
			if patch_key == "diffusion_model":
				module_names.add("")
			elif patch_key.startswith("diffusion_model."):
				module_names.add(patch_key[len("diffusion_model."):])
	return module_names


def _is_replaced_module(module_name, object_patch_module_names):
	return any(
		patch_name == ""
		or module_name == patch_name
		or module_name.startswith(f"{patch_name}.")
		for patch_name in object_patch_module_names
	)


def _normalize_static_clone_dtypes(model_patcher):
	inference_dtype = _get_model_inference_dtype(model_patcher)
	if inference_dtype is None:
		return 0
	try:
		if not torch.empty((), dtype=inference_dtype).is_floating_point():
			return 0
	except Exception:
		return 0

	try:
		diffusion_model = model_patcher.get_model_object("diffusion_model")
	except Exception:
		diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
	if diffusion_model is None:
		return 0

	object_patch_module_names = _get_object_patch_module_names(model_patcher)
	normalized_count = 0
	for module_name, module in diffusion_model.named_modules():
		if _is_replaced_module(module_name, object_patch_module_names):
			continue
		if getattr(module, "_is_quantized", False) or not hasattr(module, "comfy_cast_weights"):
			continue
		parameters = {
			parameter_name: getattr(module, parameter_name, None)
			for parameter_name in ("weight", "bias")
		}
		has_dtype_mismatch = any(
			isinstance(parameter, torch.Tensor)
			and parameter.is_floating_point()
			and parameter.dtype != inference_dtype
			for parameter in parameters.values()
		)
		if not has_dtype_mismatch:
			continue

		for parameter_name, parameter in parameters.items():
			if (
				not isinstance(parameter, torch.Tensor)
				or not parameter.is_floating_point()
				or parameter.dtype == inference_dtype
			):
				continue
			cast_parameter = parameter.to(dtype=inference_dtype)
			if isinstance(parameter, nn.Parameter):
				cast_parameter = nn.Parameter(cast_parameter, requires_grad=parameter.requires_grad)
			setattr(module, parameter_name, cast_parameter)
			setattr(module, f"{parameter_name}_comfy_model_dtype", inference_dtype)

		if not getattr(module, "weight_function", []) and not getattr(module, "bias_function", []):
			module.comfy_cast_weights = False
		normalized_count += 1

	if normalized_count > 0 and hasattr(model_patcher, "size"):
		model_patcher.size = 0
	return normalized_count


def _clone_for_lazy_compile(model, disable_dynamic_vram, verbose):
	is_dynamic = callable(getattr(model, "is_dynamic", None)) and model.is_dynamic()
	try:
		model_patcher = model.clone(disable_dynamic=bool(disable_dynamic_vram))
	except TypeError:
		logging.warning("Quantized Lazy Torch Compile: this ComfyUI version does not support disable_dynamic clone.")
		return model.clone()

	if is_dynamic and disable_dynamic_vram:
		normalized_count = _normalize_static_clone_dtypes(model_patcher)
		if normalized_count > 0 and verbose:
			logging.info(
				"Quantized Lazy Torch Compile: normalized "
				f"{normalized_count} mixed-dtype unquantized module(s) in the static MODEL clone."
			)
	return model_patcher


def _get_mode_options(mode):
	if mode == "default":
		return {}

	try:
		return torch._inductor.list_mode_options(mode)
	except Exception as e:
		logging.warning(f"Quantized Lazy Torch Compile: could not load mode options for {mode}; using default mode ({e}).")
		return {}


def _set_dynamo_cache_limits(cache_limit):
	cache_limit = int(cache_limit)
	torch._dynamo.config.cache_size_limit = cache_limit
	for setting_name in ("recompile_limit", "accumulated_recompile_limit"):
		if hasattr(torch._dynamo.config, setting_name):
			setattr(torch._dynamo.config, setting_name, cache_limit)


def _get_compile_key_list(diffusion_model, compile_transformer_blocks_only):
	if not compile_transformer_blocks_only:
		return _WHOLE_MODEL_COMPILE_KEY_LIST

	layer_types = [
		"double_blocks",
		"single_blocks",
		"layers",
		"transformer_blocks",
		"blocks",
		"visual_transformer_blocks",
		"text_transformer_blocks",
	]
	compile_key_list = []
	for layer_name in layer_types:
		if not hasattr(diffusion_model, layer_name):
			continue
		blocks = getattr(diffusion_model, layer_name)
		try:
			block_count = len(blocks)
		except TypeError:
			continue
		for index in range(block_count):
			compile_key_list.append(f"diffusion_model.{layer_name}.{index}")

	if compile_key_list:
		return compile_key_list

	logging.warning("Quantized Lazy Torch Compile: no known transformer blocks found; compiling the entire diffusion model.")
	return _WHOLE_MODEL_COMPILE_KEY_LIST


def _get_int8_adapter_model_type(model_patcher):
	try:
		transformer_options = model_patcher.model_options.get("transformer_options", {})
		adapter_state = transformer_options.get("int8_model_adapter", {})
		return adapter_state.get("model_type")
	except Exception:
		return None


def _uses_flux_global_modulation(diffusion_model):
	params = getattr(diffusion_model, "params", None)
	return bool(getattr(params, "global_modulation", False))


def _should_force_whole_model_compile(model_patcher, diffusion_model):
	model_type = _get_int8_adapter_model_type(model_patcher)
	if model_type in ("flux2", "flux2_fast_unsafe"):
		return True
	return _uses_flux_global_modulation(diffusion_model)


def _has_native_int4_modules(model_patcher):
	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if not isinstance(object_patch_map, dict):
			continue
		if any(getattr(module, "_quant_format", None) == "convrot_w4a4" for module in object_patch_map.values()):
			return True

	diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
	if diffusion_model is None:
		return False
	return any(getattr(module, "_quant_format", None) == "convrot_w4a4" for module in diffusion_model.modules())


def _has_w4a8_modules(model_patcher):
	def is_w4a8(module):
		return (
			getattr(module, "_quant_format", None) == "asym_w4a8_int8"
			or getattr(module, "quant_format", None) == "asym_w4a8_int8"
		)

	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if isinstance(object_patch_map, dict) and any(is_w4a8(module) for module in object_patch_map.values()):
			return True

	diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
	if diffusion_model is None:
		return False
	return any(is_w4a8(module) for module in diffusion_model.modules())


def _get_comfy_kitchen_version():
	try:
		return importlib.metadata.version("comfy-kitchen")
	except importlib.metadata.PackageNotFoundError:
		return "unknown"


def _build_native_int4_compile_warning():
	return (
		"Quantized Lazy Torch Compile: Native ConvRot INT4 detected; torch.compile was not applied.\n"
		f"  Installed comfy-kitchen version: {_get_comfy_kitchen_version()}\n"
		"  Upstream limitation: TensorCoreConvRotW4A4Layout does not expose a compiler-safe "
		"convrot_w4a4_linear torch.library custom operator with a FakeTensor implementation.\n"
		f"  Toolkit compatibility shim: unavailable ({int4_compile_compat.get_compile_support_error()})\n"
		"  Result: this MODEL is returned uncompiled and will use eager ConvRot INT4 inference. "
		"On systems where torch.compile materially accelerates INT8, Toolkit INT8 plus compile may "
		"be faster despite using more memory.\n"
		f"  Upstream project: {_COMFY_KITCHEN_PROJECT_URL}"
	)


def _build_native_int4_compile_info():
	support_source = int4_compile_compat.get_compile_support_source()
	if support_source == int4_compile_compat.SUPPORT_SOURCE_UPSTREAM:
		support_label = "upstream comfy-kitchen custom operator"
	else:
		support_label = "temporary Toolkit custom-op shim"
	return (
		"Quantized Lazy Torch Compile: native ConvRot INT4 compile support enabled "
		f"via {support_label} (comfy-kitchen {_get_comfy_kitchen_version()})."
	)


def _build_w4a8_compile_warning():
	return (
		"Quantized Lazy Torch Compile: W4A8 detected; torch.compile was not applied.\n"
		f"  Installed comfy-kitchen version: {_get_comfy_kitchen_version()}\n"
		"  Upstream limitation: AsymW4A8Int8Layout does not expose a compiler-safe "
		"w4a8_int8_linear torch.library custom operator with a FakeTensor implementation.\n"
		f"  Toolkit compatibility shim: unavailable ({w4a8_compile_compat.get_compile_support_error()})\n"
		"  Result: this MODEL is returned uncompiled and will use the native eager W4A8 runtime."
	)


def _build_w4a8_compile_info():
	support_source = w4a8_compile_compat.get_compile_support_source()
	if support_source == w4a8_compile_compat.SUPPORT_SOURCE_UPSTREAM:
		support_label = "upstream comfy-kitchen custom operator"
	else:
		support_label = "temporary Toolkit custom-op shim"
	return (
		"Quantized Lazy Torch Compile: W4A8 compile support enabled "
		f"via {support_label} (comfy-kitchen {_get_comfy_kitchen_version()})."
	)


def _remove_compile_wrappers(model_patcher):
	model_patcher.remove_wrappers_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		_LAZY_COMPILE_WRAPPER_KEY,
	)
	model_patcher.model_options.pop(_TORCH_COMPILE_KWARGS, None)
	try:
		from comfy_api.torch_helpers import torch_compile as comfy_torch_compile
		model_patcher.remove_wrappers_with_key(
			comfy.patcher_extension.WrappersMP.APPLY_MODEL,
			comfy_torch_compile.COMPILE_KEY,
		)
		model_patcher.model_options.pop(comfy_torch_compile.TORCH_COMPILE_KWARGS, None)
	except Exception:
		pass


def _get_output_cache(shared_model):
	cache = getattr(shared_model, _LAZY_COMPILE_OUTPUT_CACHE_KEY, None)
	if not isinstance(cache, dict):
		cache = {}
		setattr(shared_model, _LAZY_COMPILE_OUTPUT_CACHE_KEY, cache)
	return cache


def _get_structure_cache(shared_model):
	cache = getattr(shared_model, _LAZY_COMPILE_STRUCTURE_CACHE_KEY, None)
	if not isinstance(cache, dict):
		cache = {}
		setattr(shared_model, _LAZY_COMPILE_STRUCTURE_CACHE_KEY, cache)
	return cache


def _build_cache_key(
	model_patcher,
	backend,
	fullgraph,
	mode,
	dynamic_shape_tracing,
	compile_transformer_blocks_only,
	dynamo_cache_size_limit,
	use_guard_filter,
	disable_dynamic_vram,
	verbose,
):
	return (
		"v2",
		id(getattr(model_patcher, "model", None)),
		getattr(model_patcher, "patches_uuid", None),
		str(backend),
		bool(fullgraph),
		str(mode),
		str(dynamic_shape_tracing),
		bool(compile_transformer_blocks_only),
		int(dynamo_cache_size_limit),
		bool(use_guard_filter),
		bool(disable_dynamic_vram),
		bool(verbose),
	)


def _build_structure_cache_key(
	compile_key_list,
	backend,
	fullgraph,
	mode,
	dynamic_shape_tracing,
	use_guard_filter,
	disable_dynamic_vram,
	verbose,
):
	return (
		"v1",
		tuple(compile_key_list),
		str(backend),
		bool(fullgraph),
		str(mode),
		str(dynamic_shape_tracing),
		bool(use_guard_filter),
		bool(disable_dynamic_vram),
		bool(verbose),
	)


def _cleanup_compile_memory(reset_compile_cache=False):
	try:
		gc.collect()
	except Exception:
		pass

	if reset_compile_cache:
		try:
			torch._dynamo.reset()
		except Exception:
			pass

	if torch.cuda.is_available():
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass


def _dispose_cached_output(model_patcher):
	model_patcher = _resolve_cached_output(model_patcher)
	if model_patcher is None:
		return
	try:
		model_patcher.remove_wrappers_with_key(
			comfy.patcher_extension.WrappersMP.APPLY_MODEL,
			_LAZY_COMPILE_WRAPPER_KEY,
		)
	except Exception:
		pass
	try:
		model_patcher.model_options.pop(_TORCH_COMPILE_KWARGS, None)
	except Exception:
		pass


def _resolve_cached_output(cached_output):
	if isinstance(cached_output, weakref.ReferenceType):
		return cached_output()
	return cached_output


def _prune_dead_outputs(cache):
	for cache_key, cached_output in list(cache.items()):
		if _resolve_cached_output(cached_output) is None:
			cache.pop(cache_key, None)


def _make_model_ref(shared_model):
	try:
		return weakref.ref(shared_model)
	except TypeError:
		return lambda: None


def _prepare_model_cache(shared_model):
	global _LAZY_COMPILE_LAST_MODEL_ID, _LAZY_COMPILE_LAST_MODEL_REF

	shared_model_id = id(shared_model)
	prior_model = _LAZY_COMPILE_LAST_MODEL_REF() if callable(_LAZY_COMPILE_LAST_MODEL_REF) else None
	model_changed = _LAZY_COMPILE_LAST_MODEL_ID is not None and _LAZY_COMPILE_LAST_MODEL_ID != shared_model_id
	if model_changed:
		if prior_model is not None:
			prior_cache = getattr(prior_model, _LAZY_COMPILE_OUTPUT_CACHE_KEY, None)
			if isinstance(prior_cache, dict):
				for cached_output in list(prior_cache.values()):
					cached_model_patcher = _resolve_cached_output(cached_output)
					if cached_model_patcher is not None:
						_dispose_cached_output(cached_model_patcher)
				prior_cache.clear()
			prior_structure_cache = getattr(prior_model, _LAZY_COMPILE_STRUCTURE_CACHE_KEY, None)
			if isinstance(prior_structure_cache, dict):
				prior_structure_cache.clear()
		_cleanup_compile_memory(reset_compile_cache=_LAZY_COMPILE_RESET_ON_MODEL_CHANGE)

	_LAZY_COMPILE_LAST_MODEL_REF = _make_model_ref(shared_model)
	_LAZY_COMPILE_LAST_MODEL_ID = shared_model_id
	return _get_output_cache(shared_model)


def _remember_cached_output(shared_model, cache_key, model_patcher):
	if _LAZY_COMPILE_OUTPUT_CACHE_LIMIT <= 0:
		return

	cache = _get_output_cache(shared_model)
	_prune_dead_outputs(cache)
	cache[cache_key] = weakref.ref(model_patcher)
	while len(cache) > _LAZY_COMPILE_OUTPUT_CACHE_LIMIT:
		old_key = next(iter(cache))
		if old_key == cache_key and len(cache) > 1:
			old_key = next(key for key in cache if key != cache_key)
		evicted_output = cache.pop(old_key)
		evicted_model_patcher = _resolve_cached_output(evicted_output)
		if evicted_model_patcher is not None and evicted_model_patcher is not model_patcher:
			_dispose_cached_output(evicted_model_patcher)
	_cleanup_compile_memory(reset_compile_cache=False)


def _make_output_cache_room(cache):
	if _LAZY_COMPILE_OUTPUT_CACHE_LIMIT <= 0:
		return

	_prune_dead_outputs(cache)
	evicted = False
	while len(cache) >= _LAZY_COMPILE_OUTPUT_CACHE_LIMIT:
		old_key = next(iter(cache))
		evicted_model_patcher = _resolve_cached_output(cache.pop(old_key))
		if evicted_model_patcher is not None:
			_dispose_cached_output(evicted_model_patcher)
		evicted = True
	if evicted:
		_cleanup_compile_memory(reset_compile_cache=False)


def _remember_structure_wrapper(shared_model, cache_key, wrapper):
	cache = _get_structure_cache(shared_model)
	if cache_key not in cache:
		cache.clear()
		cache[cache_key] = wrapper
	return cache[cache_key]


def _make_lazy_compile_wrapper(compile_key_list, compile_kwargs, verbose):
	compiled_modules = {}
	compiled_dispatch = None
	compile_failed = False
	reported_guard_failures = set()
	reported_cache_entry_count = 0

	def get_cache_entries():
		try:
			return tuple(torch._dynamo.eval_frame._debug_get_cache_entry_list(_dispatch_compiled_module))
		except Exception:
			return ()

	def get_guard_failures():
		if not verbose:
			return ()
		try:
			from torch._dynamo import utils as dynamo_utils
			return tuple(dynamo_utils.guard_failures.get(_dispatch_compiled_module.__code__, ()))
		except Exception:
			return ()

	def report_new_guard_failures():
		for failure in get_guard_failures():
			if failure in reported_guard_failures:
				continue
			reported_guard_failures.add(failure)
			logging.info(f"Quantized Lazy Torch Compile: Dynamo cache miss: {failure}")

	def report_pending_guard_failure(module, args, kwargs):
		if not verbose:
			return
		cache_entries = get_cache_entries()
		if not cache_entries:
			return

		local_scope = {
			"module": module,
			"args": args,
			"kwargs": kwargs,
		}
		failure_parts = []
		for cache_entry in cache_entries:
			try:
				guard_info = cache_entry.guard_manager.check_verbose(local_scope)
			except Exception:
				return
			if guard_info.result:
				return
			failure_parts.extend(guard_info.verbose_code_parts)

		for failure in failure_parts:
			failure = str(failure)
			if failure in reported_guard_failures:
				continue
			reported_guard_failures.add(failure)
			logging.info(f"Quantized Lazy Torch Compile: Pending Dynamo cache miss: {failure}")
			break

	def report_completed_dispatch(module_key, elapsed_seconds):
		nonlocal reported_cache_entry_count
		if not verbose:
			return
		cache_entry_count = len(get_cache_entries())
		if cache_entry_count <= reported_cache_entry_count:
			return
		reported_cache_entry_count = cache_entry_count
		cache_entry_label = "entry" if cache_entry_count == 1 else "entries"
		logging.info(
			"Quantized Lazy Torch Compile: Dynamo graph cache grew to "
			f"{cache_entry_count} {cache_entry_label} after "
			f"{module_key} ({elapsed_seconds:.2f}s dispatch)."
		)

	def lazy_compile_wrapper(executor, *args, **kwargs):
		nonlocal compile_failed, compiled_dispatch
		if compile_failed:
			return executor(*args, **kwargs)

		prepared_keys = []
		try:
			if compiled_dispatch is None:
				compiled_dispatch = torch.compile(_dispatch_compiled_module, **compile_kwargs)
			for key in compile_key_list:
				module = comfy.utils.get_attr(executor.class_obj, key)
				cached_module = compiled_modules.get(key)
				if cached_module is not None and cached_module[0] is module:
					continue
				compiled_modules[key] = (
					module,
					_CompiledModuleProxy(
						key,
						module,
						compiled_dispatch,
						report_pending_guard_failure if verbose else None,
						report_completed_dispatch if verbose else None,
					),
				)
				prepared_keys.append(key)
			if verbose and prepared_keys:
				_log_info_outside_progress_bar(
					"Quantized Lazy Torch Compile: prepared "
					f"{len(prepared_keys)} module(s): {', '.join(prepared_keys[:6])}"
					f"{'...' if len(prepared_keys) > 6 else ''}"
				)
		except Exception as e:
			compile_failed = True
			logging.warning(f"Quantized Lazy Torch Compile: compile failed; running uncompiled ({e}).")
			return executor(*args, **kwargs)

		original_modules = {}
		try:
			for key, (_source_module, module) in compiled_modules.items():
				original_modules[key] = comfy.utils.get_attr(executor.class_obj, key)
				comfy.utils.set_attr(executor.class_obj, key, module)
			result = executor(*args, **kwargs)
		finally:
			for key, module in original_modules.items():
				comfy.utils.set_attr(executor.class_obj, key, module)
		if verbose:
			report_new_guard_failures()
		return result

	return lazy_compile_wrapper


class INT8LazyTorchCompile:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL", {"tooltip": "Model to compile lazily at first sampling call, after Comfy object patches such as INT8 module replacement are active."}),
				"backend": (["inductor", "cudagraphs"], {"default": "inductor", "tooltip": "torch.compile backend."}),
				"fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Require a single full graph. Usually leave off for Comfy workflows."}),
				"mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default", "tooltip": "torch.compile optimization mode."}),
				"dynamic_shape_tracing": (["auto", "true", "false"], {"default": "true", "tooltip": "Use dynamic shape tracing. true is often safer for changing image sizes; false may be faster for fixed shapes."}),
				"compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile recognized transformer block lists instead of the entire diffusion model."}),
				"dynamo_cache_size_limit": ("INT", {"default": 640, "min": 0, "max": 2048, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit for this process."}),
				"use_guard_filter": ("BOOLEAN", {"default": True, "tooltip": "Ignore volatile transformer_options guards, matching Comfy's stock TorchCompileModel behavior. Disabling this can retrace every transformer block and every LoRA change."}),
				"disable_dynamic_vram": ("BOOLEAN", {"default": True, "tooltip": "Demote only this MODEL output to ComfyUI's non-dynamic patcher, matching the stock Torch Compile node. This does not disable Dynamic VRAM globally."}),
				"verbose": ("BOOLEAN", {"default": True, "tooltip": "Log compile preparation, graph-cache growth, dispatch timing, and Dynamo guard-failure diagnostics."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "apply_lazy_compile"
	CATEGORY = "loaders"
	DESCRIPTION = "Lazily apply torch.compile at first sampling call, after Comfy object patches are installed."

	def apply_lazy_compile(
		self,
		model,
		backend,
		fullgraph,
		mode,
		dynamic_shape_tracing,
		compile_transformer_blocks_only,
		dynamo_cache_size_limit,
		use_guard_filter,
		disable_dynamic_vram,
		verbose,
	):
		output_cache = _prepare_model_cache(model.model)

		has_w4a8 = _has_w4a8_modules(model)
		if has_w4a8 and not w4a8_compile_compat.is_compile_supported():
			model_patcher = _clone_for_lazy_compile(model, disable_dynamic_vram, verbose)
			_remove_compile_wrappers(model_patcher)
			logging.warning(_build_w4a8_compile_warning())
			return (model_patcher,)
		if has_w4a8 and verbose:
			logging.info(_build_w4a8_compile_info())

		has_native_int4 = _has_native_int4_modules(model)
		if has_native_int4 and not int4_compile_compat.is_compile_supported():
			model_patcher = _clone_for_lazy_compile(model, disable_dynamic_vram, verbose)
			_remove_compile_wrappers(model_patcher)
			logging.warning(_build_native_int4_compile_warning())
			return (model_patcher,)
		if has_native_int4 and verbose:
			logging.info(_build_native_int4_compile_info())

		cache_key = _build_cache_key(
			model,
			backend,
			fullgraph,
			mode,
			dynamic_shape_tracing,
			compile_transformer_blocks_only,
			dynamo_cache_size_limit,
			use_guard_filter,
			disable_dynamic_vram,
			verbose,
		)
		if _LAZY_COMPILE_OUTPUT_CACHE_LIMIT > 0:
			cached_model_patcher = _resolve_cached_output(output_cache.get(cache_key))
			if cached_model_patcher is not None:
				return (cached_model_patcher,)
			output_cache.pop(cache_key, None)
			_make_output_cache_room(output_cache)

		if not disable_dynamic_vram and callable(getattr(model, "is_dynamic", None)) and model.is_dynamic():
			logging.warning(
				"Quantized Lazy Torch Compile: Dynamic VRAM remains enabled for this MODEL; "
				"ComfyUI currently treats this combination as experimental because VBAR operations cause graph breaks."
			)

		model_patcher = _clone_for_lazy_compile(model, disable_dynamic_vram, verbose)

		diffusion_model = model_patcher.get_model_object("diffusion_model")
		compile_key_list = _get_compile_key_list(diffusion_model, bool(compile_transformer_blocks_only))
		if compile_transformer_blocks_only and _should_force_whole_model_compile(model_patcher, diffusion_model):
			compile_key_list = _WHOLE_MODEL_COMPILE_KEY_LIST
			if verbose:
				logging.info("Quantized Lazy Torch Compile: using whole-model compile for Flux-style global modulation.")

		_set_dynamo_cache_limits(dynamo_cache_size_limit)
		compile_kwargs = {
			"backend": backend,
			"fullgraph": bool(fullgraph),
			"dynamic": _get_dynamic_value(dynamic_shape_tracing),
		}
		if use_guard_filter:
			compile_options = _get_mode_options(mode) if backend == "inductor" else {}
			compile_options["guard_filter_fn"] = _skip_transformer_options_guards
			compile_kwargs["options"] = compile_options
		else:
			logging.warning(
				"Quantized Lazy Torch Compile: transformer_options guard filtering is disabled; "
				"LoRA changes and per-block indices may trigger expensive recompilation."
			)
			compile_kwargs["mode"] = mode

		_remove_compile_wrappers(model_patcher)
		structure_cache_key = _build_structure_cache_key(
			compile_key_list,
			backend,
			fullgraph,
			mode,
			dynamic_shape_tracing,
			use_guard_filter,
			disable_dynamic_vram,
			verbose,
		)
		structure_cache = _get_structure_cache(model.model)
		lazy_compile_wrapper = structure_cache.get(structure_cache_key)
		if lazy_compile_wrapper is None:
			lazy_compile_wrapper = _remember_structure_wrapper(
				model.model,
				structure_cache_key,
				_make_lazy_compile_wrapper(compile_key_list, compile_kwargs, bool(verbose)),
			)

		model_patcher.add_wrapper_with_key(
			comfy.patcher_extension.WrappersMP.APPLY_MODEL,
			_LAZY_COMPILE_WRAPPER_KEY,
			lazy_compile_wrapper,
		)
		model_patcher.model_options[_TORCH_COMPILE_KWARGS] = {
			**compile_kwargs,
			"lazy": True,
			"keys": compile_key_list,
		}

		_remember_cached_output(model.model, cache_key, model_patcher)
		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8LazyTorchCompile": INT8LazyTorchCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8LazyTorchCompile": "INT8 Lazy Torch Compile",
}
