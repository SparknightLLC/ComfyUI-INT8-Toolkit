import logging

import comfy.patcher_extension

_KERNEL_CONFIG_WRAPPER_KEY = "int8_kernel_config_sync"


def _extract_transformer_options(args, kwargs):
	transformer_options = kwargs.get("transformer_options", None)
	if transformer_options is None and len(args) > 5:
		transformer_options = args[5]
	if transformer_options is None:
		transformer_options = {}
	return transformer_options


def _kernel_config_sync_wrapper(executor, *args, **kwargs):
	transformer_options = _extract_transformer_options(args, kwargs)
	config = transformer_options.get("int8_triton_kernel_config", None)
	if isinstance(config, dict):
		try:
			from . import int8_fused_kernel
			int8_fused_kernel.set_fixed_kernel_config(config, source="transformer_options", silent=True)
		except Exception:
			pass
	return executor(*args, **kwargs)


def _ensure_kernel_config_wrapper(model_patcher):
	model_patcher.remove_wrappers_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		_KERNEL_CONFIG_WRAPPER_KEY,
	)
	model_patcher.add_wrapper_with_key(
		comfy.patcher_extension.WrappersMP.APPLY_MODEL,
		_KERNEL_CONFIG_WRAPPER_KEY,
		_kernel_config_sync_wrapper,
	)


def _to_kernel_config(block_m, block_n, block_k, group_size_m, num_warps, num_stages):
	return {
		"BLOCK_M": int(block_m),
		"BLOCK_N": int(block_n),
		"BLOCK_K": int(block_k),
		"GROUP_SIZE_M": int(group_size_m),
		"num_warps": int(num_warps),
		"num_stages": int(num_stages),
	}


class INT8KernelConfigTuner:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL", {"tooltip": "INT8 model whose Triton kernel settings should be synchronized during sampling."}),
				"run_microbench": ("BOOLEAN", {"default": False, "tooltip": "Benchmark candidate kernel settings now and use the fastest result for this model."}),
				"block_m": ("INT", {"default": 128, "min": 16, "max": 512, "step": 16, "tooltip": "Triton BLOCK_M tile size for fixed INT8 matmul kernels."}),
				"block_n": ("INT", {"default": 128, "min": 16, "max": 512, "step": 16, "tooltip": "Triton BLOCK_N tile size for fixed INT8 matmul kernels."}),
				"block_k": ("INT", {"default": 64, "min": 16, "max": 512, "step": 16, "tooltip": "Triton BLOCK_K reduction tile size for fixed INT8 matmul kernels."}),
				"group_size_m": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, "tooltip": "Triton GROUP_SIZE_M launch grouping value for fixed INT8 matmul kernels."}),
				"num_warps": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "tooltip": "Number of Triton warps per program for fixed INT8 matmul kernels."}),
				"num_stages": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "tooltip": "Number of Triton pipeline stages for fixed INT8 matmul kernels."}),
				"bench_m": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64, "tooltip": "M dimension used by the optional synthetic kernel microbenchmark."}),
				"bench_k": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 64, "tooltip": "K dimension used by the optional synthetic kernel microbenchmark."}),
				"bench_n": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 64, "tooltip": "N dimension used by the optional synthetic kernel microbenchmark."}),
				"bench_warmup": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1, "tooltip": "Warmup iterations before timing each candidate kernel config."}),
				"bench_iterations": ("INT", {"default": 6, "min": 2, "max": 100, "step": 1, "tooltip": "Timed iterations per candidate kernel config."}),
				"bench_include_scalar": ("BOOLEAN", {"default": False, "tooltip": "Include scalar-weight kernel candidates in the benchmark. Usually leave off for per-row INT8 models."}),
			}
		}

	RETURN_TYPES = ("MODEL",)
	FUNCTION = "apply_kernel_config"
	CATEGORY = "loaders"
	DESCRIPTION = "Apply INT8 Triton kernel config. Optionally run an on-demand microbench and print recommended env vars."

	def apply_kernel_config(
		self,
		model,
		run_microbench,
		block_m,
		block_n,
		block_k,
		group_size_m,
		num_warps,
		num_stages,
		bench_m,
		bench_k,
		bench_n,
		bench_warmup,
		bench_iterations,
		bench_include_scalar,
	):
		model_patcher = model.clone()

		manual_config = _to_kernel_config(
			block_m,
			block_n,
			block_k,
			group_size_m,
			num_warps,
			num_stages,
		)

		try:
			from . import int8_fused_kernel
		except Exception as e:
			logging.warning(f"INT8 Kernel Config: Triton kernel module unavailable ({e}).")
			return (model_patcher,)

		if not int8_fused_kernel.is_fixed_kernel_mode():
			logging.warning("INT8 Kernel Config: INT8_TRITON_AUTOTUNE=1 is enabled; fixed kernel config values will be ignored.")

		selected_config = manual_config
		if run_microbench:
			try:
				print(
					f"[ComfyUI-Flux2-INT8] Running kernel microbench "
					f"(M={bench_m}, K={bench_k}, N={bench_n}, warmup={bench_warmup}, iters={bench_iterations})..."
				)
				best_config, results = int8_fused_kernel.microbench_fixed_kernel_configs(
					m=bench_m,
					k=bench_k,
					n=bench_n,
					warmup=bench_warmup,
					iterations=bench_iterations,
					include_scalar=bench_include_scalar,
					extra_candidates=[manual_config],
				)
				selected_config = best_config
				print("[ComfyUI-Flux2-INT8] Microbench results (top 3):")
				for row in results[:3]:
					print(f"  avg_ms={row['avg_ms']:.3f} config={row['config']}")
			except Exception as e:
				logging.warning(f"INT8 Kernel Config: microbench failed ({e}); using manual config.")

		applied_config = int8_fused_kernel.set_fixed_kernel_config(
			selected_config,
			source="INT8KernelConfigTuner",
		)

		if "transformer_options" not in model_patcher.model_options:
			model_patcher.model_options["transformer_options"] = {}
		else:
			model_patcher.model_options["transformer_options"] = model_patcher.model_options["transformer_options"].copy()

		opts = model_patcher.model_options["transformer_options"]
		opts["int8_triton_kernel_config"] = dict(applied_config)

		_ensure_kernel_config_wrapper(model_patcher)

		print("[ComfyUI-Flux2-INT8] Recommended environment variables for persistent config:")
		for line in int8_fused_kernel.format_kernel_config_env_lines(applied_config):
			print(f"  {line}")

		return (model_patcher,)


NODE_CLASS_MAPPINGS = {
	"INT8KernelConfigTuner": INT8KernelConfigTuner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INT8KernelConfigTuner": "INT8 Kernel Config",
}
