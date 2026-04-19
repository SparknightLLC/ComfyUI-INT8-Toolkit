import os
import time
import torch

# =============================================================================
# Runtime Kernel Configuration
# =============================================================================

LOG_PREFIX = "[ComfyUI-INT8-Toolkit]"


def _read_env_int(name: str, default_value: int) -> int:
	raw_value = os.environ.get(name)
	if raw_value is None:
		return default_value
	try:
		return int(raw_value)
	except ValueError:
		print(f"{LOG_PREFIX} Invalid {name}={raw_value!r}; using {default_value}.")
		return default_value


_ENABLE_TRITON_AUTOTUNE = os.environ.get("INT8_TRITON_AUTOTUNE", "0") == "1"

_FIXED_KERNEL_CONFIG = {
	"BLOCK_M": max(16, _read_env_int("INT8_TRITON_BLOCK_M", 128)),
	"BLOCK_N": max(16, _read_env_int("INT8_TRITON_BLOCK_N", 128)),
	"BLOCK_K": max(16, _read_env_int("INT8_TRITON_BLOCK_K", 64)),
	"GROUP_SIZE_M": max(1, _read_env_int("INT8_TRITON_GROUP_SIZE_M", 8)),
	"num_warps": max(1, _read_env_int("INT8_TRITON_NUM_WARPS", 4)),
	"num_stages": max(1, _read_env_int("INT8_TRITON_NUM_STAGES", 4)),
}

import triton
import triton.language as tl
from triton.language.extra import libdevice

_AUTOTUNE_CONFIGS = [
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
	triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
]

_KERNEL_CONFIG_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "num_warps", "num_stages")


def _sanitize_kernel_config(config: dict) -> dict:
	return {
		"BLOCK_M": max(16, int(config["BLOCK_M"])),
		"BLOCK_N": max(16, int(config["BLOCK_N"])),
		"BLOCK_K": max(16, int(config["BLOCK_K"])),
		"GROUP_SIZE_M": max(1, int(config["GROUP_SIZE_M"])),
		"num_warps": max(1, int(config["num_warps"])),
		"num_stages": max(1, int(config["num_stages"])),
	}


def get_fixed_kernel_config() -> dict:
	return dict(_FIXED_KERNEL_CONFIG)


def is_fixed_kernel_mode() -> bool:
	return not _ENABLE_TRITON_AUTOTUNE


def set_fixed_kernel_config(config: dict, source: str = "runtime", silent: bool = False) -> dict:
	global _FIXED_KERNEL_CONFIG
	try:
		merged = dict(_FIXED_KERNEL_CONFIG)
		for key in _KERNEL_CONFIG_KEYS:
			if key in config:
				merged[key] = config[key]
		_FIXED_KERNEL_CONFIG = _sanitize_kernel_config(merged)
		if not silent:
			print(f"{LOG_PREFIX} Applied INT8 Triton config from {source}: {_FIXED_KERNEL_CONFIG}")
	except Exception as e:
		if not silent:
			print(f"{LOG_PREFIX} Failed to apply kernel config from {source}: {e}")
	return dict(_FIXED_KERNEL_CONFIG)


def format_kernel_config_env_lines(config: dict) -> list[str]:
	cfg = _sanitize_kernel_config(config)
	return [
		f"INT8_TRITON_BLOCK_M={cfg['BLOCK_M']}",
		f"INT8_TRITON_BLOCK_N={cfg['BLOCK_N']}",
		f"INT8_TRITON_BLOCK_K={cfg['BLOCK_K']}",
		f"INT8_TRITON_GROUP_SIZE_M={cfg['GROUP_SIZE_M']}",
		f"INT8_TRITON_NUM_WARPS={cfg['num_warps']}",
		f"INT8_TRITON_NUM_STAGES={cfg['num_stages']}",
	]


def get_candidate_kernel_configs(extra_candidates: list[dict] | None = None, include_current: bool = True) -> list[dict]:
	candidates = []
	seen = set()

	def _add(cfg):
		try:
			sanitized = _sanitize_kernel_config(cfg)
		except Exception:
			return
		fingerprint = tuple(sanitized[key] for key in _KERNEL_CONFIG_KEYS)
		if fingerprint in seen:
			return
		seen.add(fingerprint)
		candidates.append(sanitized)

	if include_current:
		_add(_FIXED_KERNEL_CONFIG)

	for cfg in _AUTOTUNE_CONFIGS:
		kwargs = getattr(cfg, "kwargs", None)
		if kwargs is None:
			continue
		_add({
			"BLOCK_M": kwargs.get("BLOCK_M", _FIXED_KERNEL_CONFIG["BLOCK_M"]),
			"BLOCK_N": kwargs.get("BLOCK_N", _FIXED_KERNEL_CONFIG["BLOCK_N"]),
			"BLOCK_K": kwargs.get("BLOCK_K", _FIXED_KERNEL_CONFIG["BLOCK_K"]),
			"GROUP_SIZE_M": kwargs.get("GROUP_SIZE_M", _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"]),
			"num_warps": getattr(cfg, "num_warps", _FIXED_KERNEL_CONFIG["num_warps"]),
			"num_stages": getattr(cfg, "num_stages", _FIXED_KERNEL_CONFIG["num_stages"]),
		})

	if extra_candidates:
		for cfg in extra_candidates:
			if isinstance(cfg, dict):
				_add(cfg)

	return candidates


@torch.no_grad()
def microbench_fixed_kernel_configs(
	m: int = 2048,
	k: int = 4096,
	n: int = 4096,
	warmup: int = 2,
	iterations: int = 6,
	include_scalar: bool = False,
	extra_candidates: list[dict] | None = None,
):
	if not is_fixed_kernel_mode():
		raise RuntimeError("INT8_TRITON_AUTOTUNE=1 is active; fixed kernel config microbench is unavailable.")

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for Triton kernel microbench.")

	m = max(64, int(m))
	k = max(64, int(k))
	n = max(64, int(n))
	warmup = max(1, int(warmup))
	iterations = max(2, int(iterations))

	original_config = get_fixed_kernel_config()
	candidates = get_candidate_kernel_configs(extra_candidates=extra_candidates, include_current=True)
	if not candidates:
		raise RuntimeError("No Triton kernel configs available for benchmarking.")

	device = torch.device("cuda")
	x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
	weight = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
	w_scale_row = torch.rand((n, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.005)
	w_scale_scalar = torch.tensor([0.01], device=device, dtype=torch.float32)
	bias = torch.randn((n,), device=device, dtype=torch.float32)

	results = []
	try:
		for cfg in candidates:
			set_fixed_kernel_config(cfg, source="microbench", silent=True)

			for _ in range(warmup):
				_ = triton_int8_linear_per_row(x, weight, w_scale_row, bias=bias, compute_dtype=torch.bfloat16)
				if include_scalar:
					_ = triton_int8_linear(x, weight, w_scale_scalar, bias=bias, compute_dtype=torch.bfloat16)
			torch.cuda.synchronize()

			start = time.perf_counter()
			for _ in range(iterations):
				_ = triton_int8_linear_per_row(x, weight, w_scale_row, bias=bias, compute_dtype=torch.bfloat16)
				if include_scalar:
					_ = triton_int8_linear(x, weight, w_scale_scalar, bias=bias, compute_dtype=torch.bfloat16)
			torch.cuda.synchronize()

			elapsed_ms = (time.perf_counter() - start) * 1000.0
			avg_ms = elapsed_ms / iterations
			results.append({
				"config": dict(cfg),
				"avg_ms": avg_ms,
			})
	finally:
		set_fixed_kernel_config(original_config, source="restore", silent=True)

	results.sort(key=lambda item: item["avg_ms"])
	best = results[0]["config"]
	return best, results

if _ENABLE_TRITON_AUTOTUNE:
	print(f"{LOG_PREFIX} Triton autotune is enabled (INT8_TRITON_AUTOTUNE=1).")
else:
	print(
		f"{LOG_PREFIX} Triton autotune is disabled; using fixed INT8 kernel config "
		f"{_FIXED_KERNEL_CONFIG}."
	)


def _kernel_strategy_decorator():
	if _ENABLE_TRITON_AUTOTUNE:
		return triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"], cache_results=True)

	def _identity(kernel):
		return kernel

	return _identity


_KERNEL_STRATEGY_DECORATOR = _kernel_strategy_decorator()

# =============================================================================
# Kernel 1: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)
# =============================================================================


@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,  # Input pointer (FP16/BF16)
    y_ptr,  # Output pointer (INT8)
    s_ptr,  # Scale pointer (FP32)
    n_elements,  # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
	# Row index we are processing
	row_idx = tl.program_id(0)

	# Pointers to the start of the row
	x_row_ptr = x_ptr + row_idx * n_elements
	y_row_ptr = y_ptr + row_idx * n_elements

	# 1. Compute Max Abs Value for the row
	offsets = tl.arange(0, BLOCK_SIZE)
	mask = offsets < n_elements

	# Load data
	x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)

	# Absolute value
	abs_x = tl.abs(x)

	# Reduction to find max
	max_val = tl.max(abs_x, axis=0)

	# 2. Compute Scale
	# scale = max_val / 127.0
	scale = tl.maximum(max_val / 127.0, 1e-30)

	# 3. Quantize
	# q = x / scale
	q_f = x / scale

	# Round and Clamp
	# FIX: Use floor(x + 0.5) for rounding. This is portable across Triton versions.
	q_i = libdevice.rint(q_f).to(tl.int32)
	q_i = tl.clamp(q_i, -128.0, 127.0)

	# 4. Store
	tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
	tl.store(s_ptr + row_idx, scale.to(tl.float32))


def triton_quantize_rowwise(x: torch.Tensor):
	"""
    Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch, 1] (float32)
    """
	rows, cols = x.shape
	y = torch.empty_like(x, dtype=torch.int8)
	s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)

	# Heuristic for block size
	BLOCK_SIZE = triton.next_power_of_2(cols)
	if BLOCK_SIZE < 128:
		BLOCK_SIZE = 128

	# Note: If cols > BLOCK_SIZE (e.g. > 8192 usually), this naive block logic needs a loop.
	# For Flux2 Klein, Z-Image, Chroma layers this appears fine afaik.

	grid = (rows, )
	_quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
	return y, s


# =============================================================================
# Kernel 2: INT8 GEMM + Fused Dequantization Epilogue
# =============================================================================


@_KERNEL_STRATEGY_DECORATOR
@triton.jit
def _int8_matmul_dequant_kernel(
        # Pointers
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        # Matrix Dimensions
        M,
        N,
        K,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        HAS_BIAS: tl.constexpr):
	"""
    Computes: C = ((A * B) * (scale_a * scale_b)) + bias
    A: [M, K] int8
    B: [N, K] int8 (Transposed physically or logically via strides)
    """
	pid = tl.program_id(axis=0)
	num_pid_m = tl.cdiv(M, BLOCK_M)
	num_pid_n = tl.cdiv(N, BLOCK_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m

	# 1. Prepare Pointers for A and B
	# A block pointer: [BLOCK_M, BLOCK_K]
	offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
	offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
	offs_k = tl.arange(0, BLOCK_K)

	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

	# 2. Main Loop (Accumulate in Int32)
	accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		# Load chunks
		a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
		b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

		# Matrix Multiply (Int8 inputs -> Int32 accum)
		accumulator += tl.dot(a, b)

		# Advance pointers
		a_ptrs += BLOCK_K * stride_ak
		b_ptrs += BLOCK_K * stride_bk

	# 3. Fused Epilogue (Dequantize & Bias)

	# Load dynamic scales
	# A Scale is per-row [M, 1]
	scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]

	# B Scale is scalar or tensor.
	scale_b = tl.load(b_scale_ptr)

	# Convert Accumulator to Float
	c = accumulator.to(tl.float32)

	# Combine scales: scale_a (broadcast columns) * scale_b
	total_scale = scale_a[:, None] * scale_b

	c = c * total_scale

	# Add Bias if present
	if HAS_BIAS:
		bias = tl.load(bias_ptr + offs_bn)  # Vector [BLOCK_N]
		c = c + bias[None, :]

	# 4. Store Result (Cast to output dtype, usually FP16)
	c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
	c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

	# We write as fp16 or bf16 implicitly by the pointer type, but explicit cast is safer
	tl.store(c_ptrs, c, mask=c_mask)


# =============================================================================
# Python Wrapper
# =============================================================================


def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=torch.float16):
	"""
    Fused pipeline for W8A8 Linear Layer.
    """
	# 1. Flatten inputs if 3D [Batch, Tokens, Dim] -> [Batch*Tokens, Dim]
	x_shape_orig = x.shape
	x_2d = x.reshape(-1, x_shape_orig[-1])

	M, K = x_2d.shape
	N = weight.shape[0]

	# 2. Kernel 1: Dynamic Activation Quantization
	#    (This is much faster than Python-loop based axiswise quant)
	x_int8, x_scale = triton_quantize_rowwise(x_2d)

	# 3. Allocate Output
	output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

	# 4. Prepare Scales for Kernel
	# Ensure weight_scale is a tensor on device
	if not isinstance(weight_scale, torch.Tensor):
		weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
	elif weight_scale.numel() == 1:
		weight_scale = weight_scale.reshape(1)

	# 5. Kernel 2: Fused GEMM + Dequant
	grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

	# Check if we have bias
	has_bias = bias is not None
	bias_ptr = bias if has_bias else x  # Dummy pointer if None
	launch_kwargs = {"HAS_BIAS": has_bias}
	if not _ENABLE_TRITON_AUTOTUNE:
		launch_kwargs.update({
			"BLOCK_M": _FIXED_KERNEL_CONFIG["BLOCK_M"],
			"BLOCK_N": _FIXED_KERNEL_CONFIG["BLOCK_N"],
			"BLOCK_K": _FIXED_KERNEL_CONFIG["BLOCK_K"],
			"GROUP_SIZE_M": _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"],
			"num_warps": _FIXED_KERNEL_CONFIG["num_warps"],
			"num_stages": _FIXED_KERNEL_CONFIG["num_stages"],
		})

	# NOTE: PyTorch Linear weights are [Out, In] (N, K).
	# The kernel expects B to be [K, N] logically.
	# Since weight is [N, K], we can treat it as [K, N] TRANSPOSED.
	# Stride of W is [K, 1]. To read as column-major [K, N], stride is [1, K].

	_int8_matmul_dequant_kernel[grid](
	    # Pointers
	    a_ptr=x_int8,
	    b_ptr=weight,
	    c_ptr=output,
	    a_scale_ptr=x_scale,
	    b_scale_ptr=weight_scale,
	    bias_ptr=bias_ptr,
	    # Shapes
	    M=M,
	    N=N,
	    K=K,
	    # Strides
	    stride_am=x_int8.stride(0),
	    stride_ak=x_int8.stride(1),
	    stride_bk=weight.stride(1),
	    stride_bn=weight.stride(0),  # Transposed access of W
	    stride_cm=output.stride(0),
	    stride_cn=output.stride(1),
	    # Meta
	    **launch_kwargs)

	# 6. Reshape output
	return output.reshape(x_shape_orig[:-1] + (N, ))


# =============================================================================
# Kernel 3: INT8 GEMM + Fused Dequant with Per-Row Weight Scales
# =============================================================================


@_KERNEL_STRATEGY_DECORATOR
@triton.jit
def _int8_matmul_dequant_per_row_kernel(
		# Pointers
		a_ptr,
		b_ptr,
		c_ptr,
		a_scale_ptr,
		b_scale_ptr,
		bias_ptr,
		# Matrix Dimensions
		M,
		N,
		K,
		# Strides
		stride_am,
		stride_ak,
		stride_bk,
		stride_bn,
		stride_cm,
		stride_cn,
		# Meta-parameters
		BLOCK_M: tl.constexpr,
		BLOCK_N: tl.constexpr,
		BLOCK_K: tl.constexpr,
		GROUP_SIZE_M: tl.constexpr,
		HAS_BIAS: tl.constexpr):
	"""
	Computes: C = ((A * B) * (scale_a[:, None] * scale_b[None, :])) + bias
	A: [M, K] int8, scale_a: [M, 1] per-row activation scales
	B: [N, K] int8, scale_b: [N, 1] per-row weight scales
	"""
	pid = tl.program_id(axis=0)
	num_pid_m = tl.cdiv(M, BLOCK_M)
	num_pid_n = tl.cdiv(N, BLOCK_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m

	# 1. Prepare Pointers for A and B
	offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
	offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
	offs_k = tl.arange(0, BLOCK_K)

	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

	# 2. Main Loop (Accumulate in Int32)
	accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
		b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
		accumulator += tl.dot(a, b)
		a_ptrs += BLOCK_K * stride_ak
		b_ptrs += BLOCK_K * stride_bk

	# 3. Fused Epilogue (Dequantize & Bias)
	scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
	scale_b = tl.load(b_scale_ptr + offs_bn)  # Vector [BLOCK_N]

	c = accumulator.to(tl.float32)
	total_scale = scale_a[:, None] * scale_b[None, :]
	c = c * total_scale

	if HAS_BIAS:
		bias = tl.load(bias_ptr + offs_bn)
		c = c + bias[None, :]

	# 4. Store Result
	c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
	c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
	tl.store(c_ptrs, c, mask=c_mask)


# =============================================================================
# Python Wrapper (Per-Row Weight Scales)
# =============================================================================


def triton_int8_linear_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None, compute_dtype=torch.float16):
	"""
	Fused pipeline for W8A8 Linear Layer with per-row weight quantization.
	weight_scale: [N, 1] per-row scales
	"""
	# 1. Flatten inputs if 3D
	x_shape_orig = x.shape
	x_2d = x.reshape(-1, x_shape_orig[-1])

	M, K = x_2d.shape
	N = weight.shape[0]

	# 2. Dynamic Activation Quantization
	x_int8, x_scale = triton_quantize_rowwise(x_2d)

	# 3. Allocate Output
	output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

	# 4. Prepare weight scales - flatten [N, 1] -> [N] for kernel
	ws = weight_scale.reshape(N).contiguous()

	# 5. Fused GEMM + Per-Row Dequant
	grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

	has_bias = bias is not None
	bias_ptr = bias if has_bias else x  # Dummy pointer if None
	launch_kwargs = {"HAS_BIAS": has_bias}
	if not _ENABLE_TRITON_AUTOTUNE:
		launch_kwargs.update({
			"BLOCK_M": _FIXED_KERNEL_CONFIG["BLOCK_M"],
			"BLOCK_N": _FIXED_KERNEL_CONFIG["BLOCK_N"],
			"BLOCK_K": _FIXED_KERNEL_CONFIG["BLOCK_K"],
			"GROUP_SIZE_M": _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"],
			"num_warps": _FIXED_KERNEL_CONFIG["num_warps"],
			"num_stages": _FIXED_KERNEL_CONFIG["num_stages"],
		})

	_int8_matmul_dequant_per_row_kernel[grid](
		a_ptr=x_int8,
		b_ptr=weight,
		c_ptr=output,
		a_scale_ptr=x_scale,
		b_scale_ptr=ws,
		bias_ptr=bias_ptr,
		M=M,
		N=N,
		K=K,
		stride_am=x_int8.stride(0),
		stride_ak=x_int8.stride(1),
		stride_bk=weight.stride(1),
		stride_bn=weight.stride(0),
		stride_cm=output.stride(0),
		stride_cn=output.stride(1),
		**launch_kwargs)

	# 6. Reshape output
	return output.reshape(x_shape_orig[:-1] + (N, ))
