import logging

import comfy.patcher_extension
import comfy.utils
import torch


_LAZY_COMPILE_WRAPPER_KEY = "int8_lazy_torch_compile"
_TORCH_COMPILE_KWARGS = "torch_compile_kwargs"


def _skip_transformer_options_guards(guard_entries):
	return [("transformer_options" not in entry.name) for entry in guard_entries]


def _get_dynamic_value(dynamic):
	dynamic_values = {
		"auto": None,
		"true": True,
		"false": False,
	}
	if dynamic not in dynamic_values:
		raise ValueError(f"Invalid dynamic value {dynamic}")
	return dynamic_values[dynamic]


def _get_mode_options(mode):
	if mode == "default":
		return {}

	try:
		return torch._inductor.list_mode_options(mode)
	except Exception as e:
		logging.warning(f"INT8 Lazy Torch Compile: could not load mode options for {mode}; using default mode ({e}).")
		return {}


def _get_compile_key_list(diffusion_model, compile_transformer_blocks_only):
	if not compile_transformer_blocks_only:
		return ["diffusion_model"]

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

	logging.warning("INT8 Lazy Torch Compile: no known transformer blocks found; compiling entire diffusion model.")
	return ["diffusion_model"]


def _make_lazy_compile_wrapper(compile_key_list, compile_kwargs, log_compile):
	compiled_modules = {}
	compile_failed = False

	def lazy_compile_wrapper(executor, *args, **kwargs):
		nonlocal compile_failed
		if compile_failed:
			return executor(*args, **kwargs)

		if not compiled_modules:
			try:
				for key in compile_key_list:
					module = comfy.utils.get_attr(executor.class_obj, key)
					compiled_modules[key] = torch.compile(module, **compile_kwargs)
				if log_compile:
					print(
						"[INT8 Lazy Torch Compile] Compiled "
						f"{len(compiled_modules)} module(s): {', '.join(compile_key_list[:6])}"
						f"{'...' if len(compile_key_list) > 6 else ''}"
					)
			except Exception as e:
				compile_failed = True
				logging.warning(f"INT8 Lazy Torch Compile: compile failed; running uncompiled ({e}).")
				return executor(*args, **kwargs)

		original_modules = {}
		try:
			for key, module in compiled_modules.items():
				original_modules[key] = comfy.utils.get_attr(executor.class_obj, key)
				comfy.utils.set_attr(executor.class_obj, key, module)
			return executor(*args, **kwargs)
		finally:
			for key, module in original_modules.items():
				comfy.utils.set_attr(executor.class_obj, key, module)

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
				"dynamic": (["auto", "true", "false"], {"default": "true", "tooltip": "Use dynamic shape tracing. true is often safer for changing image sizes; false may be faster for fixed shapes."}),
				"compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile recognized transformer block lists instead of the entire diffusion model."}),
				"dynamo_cache_size_limit": ("INT", {"default": 640, "min": 0, "max": 2048, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit for this process."}),
				"use_guard_filter": ("BOOLEAN", {"default": True, "tooltip": "Ignore TorchDynamo guards involving transformer_options, matching Comfy's stock TorchCompileModel behavior."}),
				"disable_dynamic_vram": ("BOOLEAN", {"default": True, "tooltip": "Clone the model with dynamic VRAM disabled when supported, matching common torch.compile practice in Comfy."}),
				"log_compile": ("BOOLEAN", {"default": True, "tooltip": "Print a one-time message listing how many modules were compiled."}),
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
		dynamic,
		compile_transformer_blocks_only,
		dynamo_cache_size_limit,
		use_guard_filter,
		disable_dynamic_vram,
		log_compile,
	):
		try:
			model_patcher = model.clone(disable_dynamic=bool(disable_dynamic_vram))
		except TypeError:
			logging.warning("INT8 Lazy Torch Compile: this ComfyUI version does not support disable_dynamic clone.")
			model_patcher = model.clone()

		diffusion_model = model_patcher.get_model_object("diffusion_model")
		compile_key_list = _get_compile_key_list(diffusion_model, bool(compile_transformer_blocks_only))

		torch._dynamo.config.cache_size_limit = int(dynamo_cache_size_limit)
		compile_kwargs = {
			"backend": backend,
			"fullgraph": bool(fullgraph),
			"dynamic": _get_dynamic_value(dynamic),
		}
		if use_guard_filter:
			compile_options = _get_mode_options(mode) if backend == "inductor" else {}
			compile_options["guard_filter_fn"] = _skip_transformer_options_guards
			compile_kwargs["options"] = compile_options
		else:
			compile_kwargs["mode"] = mode

		model_patcher.remove_wrappers_with_key(
			comfy.patcher_extension.WrappersMP.APPLY_MODEL,
			_LAZY_COMPILE_WRAPPER_KEY,
		)
		try:
			from comfy_api.torch_helpers import torch_compile as comfy_torch_compile
			model_patcher.remove_wrappers_with_key(
				comfy.patcher_extension.WrappersMP.APPLY_MODEL,
				comfy_torch_compile.COMPILE_KEY,
			)
			model_patcher.model_options.pop(comfy_torch_compile.TORCH_COMPILE_KWARGS, None)
		except Exception:
			pass

		model_patcher.add_wrapper_with_key(
			comfy.patcher_extension.WrappersMP.APPLY_MODEL,
			_LAZY_COMPILE_WRAPPER_KEY,
			_make_lazy_compile_wrapper(compile_key_list, compile_kwargs, bool(log_compile)),
		)
		model_patcher.model_options[_TORCH_COMPILE_KWARGS] = {
			**compile_kwargs,
			"lazy": True,
			"keys": compile_key_list,
		}

		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8LazyTorchCompile": INT8LazyTorchCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8LazyTorchCompile": "INT8 Lazy Torch Compile",
}
