import torch
import os
import logging
from torch import Tensor, nn
import torch.nn.functional as F
import comfy.model_management

_INT8_FORCE_DISABLE_TORCH_COMPILE = os.environ.get("INT8_FORCE_DISABLE_TORCH_COMPILE", "0") == "1"

# Add this at the top of your file
try:
    from .int8_fused_kernel import triton_int8_linear
    from .int8_fused_kernel import triton_int8_linear_per_row
    from .int8_fused_kernel import triton_quantize_rowwise
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    print("Triton not found, falling back to torch._int_mm")

try:
    from .quarot import build_hadamard as _quarot_build_hadamard
    from .quarot import rotate_weight as _quarot_rotate_weight
    _QUAROT_AVAILABLE = True
except Exception:
    _QUAROT_AVAILABLE = False

if _INT8_FORCE_DISABLE_TORCH_COMPILE:
    try:
        _disable_torch_compile = torch.compiler.disable
    except Exception:
        try:
            import torch._dynamo as _torch_dynamo
            _disable_torch_compile = _torch_dynamo.disable
        except Exception:
            def _disable_torch_compile(fn):
                return fn
else:
    def _disable_torch_compile(fn):
        return fn

try:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = max(0, int(os.environ.get("INT8_SMALL_BATCH_FALLBACK_MAX_ROWS", "16")))
except ValueError:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = 16

try:
    _SMALL_BATCH_FALLBACK_MIN_ROWS = max(1, int(os.environ.get("INT8_SMALL_BATCH_FALLBACK_MIN_ROWS", "2")))
except ValueError:
    _SMALL_BATCH_FALLBACK_MIN_ROWS = 2

try:
    _SMALL_LAYER_MAX_PARAMS = max(1, int(os.environ.get("INT8_SMALL_LAYER_MAX_PARAMS", "1000000")))
except ValueError:
    _SMALL_LAYER_MAX_PARAMS = 1_000_000

_ADAPTIVE_SMALL_BATCH_FALLBACK = os.environ.get("INT8_SMALL_BATCH_FALLBACK_ADAPTIVE", "1") == "1"
_DYNAMIC_LORA_DEBUG = os.environ.get("INT8_DYNAMIC_LORA_DEBUG", "0") == "1"
_DYNAMIC_LORA_BATCH = os.environ.get("INT8_DYNAMIC_LORA_BATCH", "1") == "1"
_RUNTIME_STATS_ENABLED = os.environ.get("INT8_RUNTIME_STATS", "0") == "1"
SMALL_BATCH_FALLBACK_NEVER = "never"
SMALL_BATCH_FALLBACK_SMALL_LAYERS = "only_small_layers"
SMALL_BATCH_FALLBACK_ALWAYS = "always"
SMALL_BATCH_FALLBACK_CHOICES = [
    SMALL_BATCH_FALLBACK_SMALL_LAYERS,
    SMALL_BATCH_FALLBACK_ALWAYS,
    SMALL_BATCH_FALLBACK_NEVER,
]
DEFAULT_SMALL_BATCH_FALLBACK = SMALL_BATCH_FALLBACK_SMALL_LAYERS
INT8_BACKEND_TRITON = "triton"
INT8_BACKEND_TRITON_LEGACY_UNSAFE = "triton_legacy_unsafe"
INT8_BACKEND_TORCH_INT_MM = "torch_int_mm"
INT8_BACKEND_CHOICES = [
    INT8_BACKEND_TORCH_INT_MM,
    INT8_BACKEND_TRITON,
    INT8_BACKEND_TRITON_LEGACY_UNSAFE,
]
DEFAULT_INT8_BACKEND = INT8_BACKEND_TORCH_INT_MM
OUTLIER_METHOD_NONE = "none"
OUTLIER_METHOD_QUAROT = "quarot"
OUTLIER_METHOD_HADANORM = "hadanorm"
OUTLIER_METHOD_CHOICES = [OUTLIER_METHOD_NONE, OUTLIER_METHOD_QUAROT, OUTLIER_METHOD_HADANORM]
_QUAROT_GROUP_SIZE = 128
_QUAROT_OFFSET_WARNED: set[tuple[int, int, int] | str] = set()
_HADANORM_ALPHA = 0.5
_HADANORM_EPS = 1e-6

try:
    _DYNAMIC_LORA_BATCH_MAX_RANK = max(64, int(os.environ.get("INT8_DYNAMIC_LORA_BATCH_MAX_RANK", "4096")))
except ValueError:
    _DYNAMIC_LORA_BATCH_MAX_RANK = 4096

_FLOAT8_DTYPES = tuple(
    dtype for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

# --- Quantization Utils ---

def _get_int8_compute_device(fallback_device: torch.device | None = None) -> torch.device:
    try:
        return comfy.model_management.get_torch_device()
    except Exception:
        if fallback_device is not None:
            return fallback_device
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FLOAT8_DTYPES

def _normalize_outlier_method(method) -> str:
    if not isinstance(method, str):
        return OUTLIER_METHOD_NONE

    normalized = method.strip().lower()
    if normalized in OUTLIER_METHOD_CHOICES:
        return normalized
    return OUTLIER_METHOD_NONE

def _get_module_outlier_method(module) -> str:
    method = _normalize_outlier_method(getattr(module, "_outlier_method", None))
    if method != OUTLIER_METHOD_NONE:
        return method
    if bool(getattr(module, "_use_quarot", False)):
        return OUTLIER_METHOD_QUAROT
    return OUTLIER_METHOD_NONE

def _outlier_method_uses_hadamard(method) -> bool:
    normalized = _normalize_outlier_method(method)
    return normalized in (OUTLIER_METHOD_QUAROT, OUTLIER_METHOD_HADANORM)

def _compute_hadanorm_sigma(weight: Tensor) -> Tensor:
    weight_f = weight.float()
    channel_max = weight_f.abs().amax(dim=0)
    sigma = channel_max.clamp(min=_HADANORM_EPS).pow(-(1.0 - _HADANORM_ALPHA))
    return sigma.clamp(min=_HADANORM_EPS)

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_rowwise(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Per-row (out_features-wise) INT8 quantization for weight matrices.
    Returns:
        q_weight: INT8 tensor [rows, cols]
        q_scale:  FP32 scale tensor [rows, 1]
    """
    x_for_quant = x
    if _is_float8_dtype(x_for_quant.dtype):
        # Convert FP8 weights to a quantization-friendly compute dtype first.
        x_for_quant = x_for_quant.to(torch.float16)

    if _TRITON_AVAILABLE and x_for_quant.is_cuda:
        try:
            return triton_quantize_rowwise(x_for_quant)
        except Exception:
            pass

    return quantize_int8_axiswise(x_for_quant, dim=-1)

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    if isinstance(scale, torch.Tensor) and scale.device != q.device:
        scale = scale.to(q.device, non_blocking=True)
    return q.float() * scale

def _prepack_int8_weight(weight: Tensor | None):
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8 or weight.ndim != 2:
        return None
    return weight.detach().T.contiguous()

def _get_prepacked_weight(linear_module, device: torch.device):
    packed = getattr(linear_module, "weight_packed", None)
    if not isinstance(packed, torch.Tensor):
        return None
    if packed.device != device:
        packed = packed.to(device, non_blocking=True)
    return packed

def _torch_int_mm_safe(a: Tensor, b: Tensor) -> Tensor:
    rows = int(a.shape[0])
    columns = int(b.shape[1])

    if a.is_cuda and a.shape[0] <= 16:
        pad_rows = 17 - rows
        if pad_rows > 0:
            padding = torch.zeros((pad_rows, a.shape[1]), device=a.device, dtype=a.dtype)
            a = torch.cat((a, padding), dim=0)

    if b.is_cuda and (columns % 8) != 0:
        pad_columns = 8 - (columns % 8)
        padding = torch.zeros((b.shape[0], pad_columns), device=b.device, dtype=b.dtype)
        b = torch.cat((b, padding), dim=1)

    return torch._int_mm(a, b)[:rows, :columns]

def stochastic_round_int8_delta(x: Tensor, scale: float | Tensor, seed: int = 0) -> Tensor:
    """
    Quantize a delta tensor to INT8 using stochastic rounding.
    Used for LoRA deltas to minimize quantization error.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    
    # Scale to INT8 range
    if isinstance(scale, torch.Tensor) and scale.device != x.device:
        scale = scale.to(x.device, non_blocking=True)
    x_scaled = x / scale
    
    # Stochastic rounding
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor
    del x_scaled
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_floor.shape, generator=generator, device=x.device, dtype=x_floor.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    del random_vals
    del fraction
    del x_floor
    
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# --- LinearW8A8 Core ---

@torch.no_grad()
@_disable_torch_compile
def int8_forward_dynamic(
    x: Tensor,
    weight: Tensor,
    weight_scale: float | Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype,
    use_triton: bool = True,
    weight_packed: Tensor | None = None,
    legacy_triton_unsafe: bool = False,
) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    
    # --- FAST PATH: Triton Fused Kernel ---
    if _TRITON_AVAILABLE and use_triton and x.is_cuda:
        if isinstance(weight_packed, torch.Tensor):
            return triton_int8_linear(x, weight_packed, weight_scale, bias, compute_dtype, weight_is_prepacked=True, legacy_unsafe=legacy_triton_unsafe)
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype, legacy_unsafe=legacy_triton_unsafe)

    # --- SLOW PATH: Standard PyTorch ---
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    if isinstance(weight_scale, torch.Tensor) and weight_scale.device != x.device:
        weight_scale = weight_scale.to(x.device, non_blocking=True)
    
    # INT8 Matmul (Outputs Int32)
    res = _torch_int_mm_safe(x_8, weight.T)
    
    # Dequantize: (res * weight_scale * x_scale)
    # Note: Creating intermediate Float tensors here is VRAM heavy
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

@torch.no_grad()
@_disable_torch_compile
def int8_forward_dynamic_per_row(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype,
    use_triton: bool = True,
    weight_packed: Tensor | None = None,
    legacy_triton_unsafe: bool = False,
) -> Tensor:
    """Forward with dynamic per-token activation quantization and per-row weight quantization."""

    # --- FAST PATH: Triton Fused Kernel (per-row) ---
    if _TRITON_AVAILABLE and use_triton and x.is_cuda:
        if isinstance(weight_packed, torch.Tensor):
            return triton_int8_linear_per_row(x, weight_packed, weight_scale, bias, compute_dtype, weight_is_prepacked=True, legacy_unsafe=legacy_triton_unsafe)
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype, legacy_unsafe=legacy_triton_unsafe)

    # --- SLOW PATH: Standard PyTorch ---
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    if weight_scale.device != x.device:
        weight_scale = weight_scale.to(x.device, non_blocking=True)

    # INT8 Matmul (Outputs Int32)
    res = _torch_int_mm_safe(x_8, weight.T)

    # Dequantize with per-row weight scales
    # res[i, j] = sum_k(x_8[i, k] * weight[j, k]) * x_scale[i] * weight_scale[j]
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)

    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

def _normalize_dynamic_offset(offset):
    if offset is None:
        return None
    if not isinstance(offset, (tuple, list)) or len(offset) < 3:
        return None
    try:
        return int(offset[0]), int(offset[1]), int(offset[2])
    except Exception:
        return None

def _resolve_dynamic_entry_tensors(entry, device: torch.device):
    entry_A = entry.get("A")
    entry_B = entry.get("B")
    if entry_A is None or entry_B is None:
        return None, None

    lA = entry.get("_A_cached")
    lB = entry.get("_B_cached")

    if lA is None or lA.device != device:
        lA = entry_A if entry_A.device == device else entry_A.to(device, non_blocking=True)
        entry["_A_cached"] = lA
    if lB is None or lB.device != device:
        lB = entry_B if entry_B.device == device else entry_B.to(device, non_blocking=True)
        entry["_B_cached"] = lB

    return lA, lB

def _can_batch_dynamic_entries(prepared_entries, output_length: int | None):
    if not _DYNAMIC_LORA_BATCH or len(prepared_entries) < 2:
        return False

    rank_total = 0
    input_width = None
    output_width = None
    a_dtype = None
    b_dtype = None

    for _, lA, lB in prepared_entries:
        if not isinstance(lA, torch.Tensor) or not isinstance(lB, torch.Tensor):
            return False
        if lA.ndim != 2 or lB.ndim != 2:
            return False
        if lB.shape[1] != lA.shape[0]:
            return False

        if input_width is None:
            input_width = lA.shape[1]
        elif input_width != lA.shape[1]:
            return False

        if output_width is None:
            output_width = lB.shape[0]
        elif output_width != lB.shape[0]:
            return False

        if a_dtype is None:
            a_dtype = lA.dtype
        elif a_dtype != lA.dtype:
            return False

        if b_dtype is None:
            b_dtype = lB.dtype
        elif b_dtype != lB.dtype:
            return False

        rank_total += int(lA.shape[0])
        if rank_total > _DYNAMIC_LORA_BATCH_MAX_RANK:
            return False

    if output_length is not None and output_width is not None and output_width != output_length:
        return False

    return True

def _get_small_batch_fallback_threshold(linear_module) -> int:
    mode = getattr(Int8TensorwiseOps, "small_batch_fallback_mode", SMALL_BATCH_FALLBACK_SMALL_LAYERS)
    if mode not in SMALL_BATCH_FALLBACK_CHOICES:
        mode = SMALL_BATCH_FALLBACK_SMALL_LAYERS
    if mode == SMALL_BATCH_FALLBACK_NEVER:
        return 0

    base_rows = _SMALL_BATCH_FALLBACK_MAX_ROWS
    if base_rows <= 0:
        return 0

    if not _ADAPTIVE_SMALL_BATCH_FALLBACK:
        return base_rows

    weight = getattr(linear_module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        return base_rows

    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    matmul_size = out_features * in_features
    if mode == SMALL_BATCH_FALLBACK_SMALL_LAYERS and matmul_size > _SMALL_LAYER_MAX_PARAMS:
        return 0

    rows = base_rows
    if matmul_size >= 12_000_000:
        rows = min(rows, 6)
    elif matmul_size >= 8_000_000:
        rows = min(rows, 8)
    elif matmul_size >= 4_000_000:
        rows = min(rows, 12)
    elif matmul_size <= 1_000_000:
        rows = min(32, max(rows, base_rows + 6))

    if getattr(linear_module, "_is_per_row", False):
        rows = max(_SMALL_BATCH_FALLBACK_MIN_ROWS, rows - 2)

    dynamic_entries = getattr(linear_module, "dynamic_lora_entries", None)
    dynamic_count = len(dynamic_entries) if dynamic_entries else 0
    if dynamic_count >= 4:
        rows = max(_SMALL_BATCH_FALLBACK_MIN_ROWS, rows - 4)
    elif dynamic_count >= 2:
        rows = max(_SMALL_BATCH_FALLBACK_MIN_ROWS, rows - 2)

    return max(_SMALL_BATCH_FALLBACK_MIN_ROWS, rows)

@torch.no_grad()
@_disable_torch_compile
def apply_dynamic_lora_delta(
    x_input: Tensor,
    y: Tensor,
    lora_A: Tensor | None,
    lora_B: Tensor | None,
    lora_alpha,
    lora_entries,
    device: torch.device,
    correction_x: Tensor | None = None,
) -> Tensor:
    if lora_entries:
        grouped_entries = {}
        for entry in lora_entries:
            offset_key = _normalize_dynamic_offset(entry.get("offset"))
            grouped_entries.setdefault(offset_key, []).append(entry)

        for offset_key, entries in grouped_entries.items():
            dim = None
            start = None
            length = None
            x_src = x_input
            correction_src = correction_x

            if offset_key is not None:
                dim, start, length = offset_key
                if dim == 1:
                    if not (start >= 0 and length > 0 and (start + length) <= x_input.shape[-1]):
                        if _DYNAMIC_LORA_DEBUG:
                            print(
                                f"[INT8 Dynamic LoRA] skipping invalid input offset={offset_key} "
                                f"for x shape={tuple(x_input.shape)}"
                            )
                        continue
                    x_src = x_input.narrow(-1, start, length)
                    if correction_x is not None:
                        correction_src = correction_x.narrow(-1, start, length)
                elif dim == 0:
                    if not (start >= 0 and length > 0 and (start + length) <= y.shape[-1]):
                        if _DYNAMIC_LORA_DEBUG:
                            print(
                                f"[INT8 Dynamic LoRA] skipping invalid output offset={offset_key} "
                                f"for y shape={tuple(y.shape)}"
                            )
                        continue

            prepared_entries = []
            for entry in entries:
                lA, lB = _resolve_dynamic_entry_tensors(entry, device)
                if lA is None or lB is None:
                    continue
                prepared_entries.append((entry, lA, lB))

            if not prepared_entries:
                continue

            output_length = length if dim == 0 else None
            batched = _can_batch_dynamic_entries(prepared_entries, output_length)
            if batched:
                cat_A = torch.cat([lA for _, lA, _ in prepared_entries], dim=0)
                cat_B = torch.cat([lB for _, _, lB in prepared_entries], dim=1)
                lora_x = F.linear(x_src.to(cat_A.dtype), cat_A)
                lora_y = F.linear(lora_x, cat_B).to(y.dtype)
                correction_y = None
                if correction_src is not None:
                    correction_proj = F.linear(correction_src.to(cat_A.dtype), cat_A)
                    correction_y = F.linear(correction_proj, cat_B).to(y.dtype)

                if dim == 0:
                    y.narrow(-1, start, length).add_(lora_y)
                    if correction_y is not None:
                        y.narrow(-1, start, length).add_(correction_y)
                else:
                    if lora_y.shape[-1] != y.shape[-1]:
                        if _DYNAMIC_LORA_DEBUG:
                            print(
                                f"[INT8 Dynamic LoRA] skipping mismatched batched add "
                                f"lora_y={tuple(lora_y.shape)} y={tuple(y.shape)} offset={offset_key}"
                            )
                        continue
                    y.add_(lora_y)
                    if correction_y is not None:
                        y.add_(correction_y)
                continue

            for _, lA, lB in prepared_entries:
                lora_x = F.linear(x_src.to(lA.dtype), lA)
                lora_y = F.linear(lora_x, lB).to(y.dtype)
                correction_y = None
                if correction_src is not None:
                    correction_proj = F.linear(correction_src.to(lA.dtype), lA)
                    correction_y = F.linear(correction_proj, lB).to(y.dtype)

                if dim == 0:
                    if lora_y.shape[-1] != length:
                        if _DYNAMIC_LORA_DEBUG:
                            print(
                                f"[INT8 Dynamic LoRA] skipping mismatched output slice "
                                f"offset={offset_key} lora_y={tuple(lora_y.shape)} y={tuple(y.shape)}"
                            )
                        continue
                    y.narrow(-1, start, length).add_(lora_y)
                    if correction_y is not None:
                        y.narrow(-1, start, length).add_(correction_y)
                    continue

                if lora_y.shape[-1] != y.shape[-1]:
                    if _DYNAMIC_LORA_DEBUG:
                        print(
                            f"[INT8 Dynamic LoRA] skipping mismatched full add "
                            f"lora_y={tuple(lora_y.shape)} y={tuple(y.shape)} offset={offset_key}"
                        )
                    continue

                y.add_(lora_y)
                if correction_y is not None:
                    y.add_(correction_y)

        return y

    if lora_A is None or lora_B is None:
        return y

    lA = lora_A if lora_A.device == device else lora_A.to(device, non_blocking=True)
    lB = lora_B if lora_B.device == device else lora_B.to(device, non_blocking=True)

    lora_x = F.linear(x_input.to(lA.dtype), lA)
    lora_y = F.linear(lora_x, lB)

    if lora_alpha is not None:
        lora_y = lora_y * lora_alpha

    output = y + lora_y.to(y.dtype)
    if correction_x is not None:
        correction_proj = F.linear(correction_x.to(lA.dtype), lA)
        correction_y = F.linear(correction_proj, lB)
        if lora_alpha is not None:
            correction_y = correction_y * lora_alpha
        output = output + correction_y.to(y.dtype)

    return output




# =============================================================================
# INT8 LoRA Adapter - High Precision, Low RAM Patching
# =============================================================================

def _unpack_lora_weights(weights):
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        return None

    mat1 = weights[0]
    mat2 = weights[1]
    alpha = weights[2] if len(weights) > 2 else None
    mid = weights[3] if len(weights) > 3 else None
    dora_scale = weights[4] if len(weights) > 4 else None
    reshape = weights[5] if len(weights) > 5 else None

    if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        return None

    return mat1, mat2, alpha, mid, dora_scale, reshape


def _compute_lora_scale(weights, strength):
    unpacked = _unpack_lora_weights(weights)
    if unpacked is None:
        return strength

    _, mat2, alpha, _, _, _ = unpacked
    rank = mat2.shape[0] if mat2.ndim >= 2 else 1
    if alpha is None:
        return strength

    return (alpha / rank) * strength


def _compute_fast_lora_diff(weights, target_shape, device, intermediate_dtype):
    unpacked = _unpack_lora_weights(weights)
    if unpacked is None:
        return None

    mat1, mat2, _, mid, dora_scale, reshape = unpacked
    if dora_scale is not None or reshape is not None:
        return None

    try:
        mat1_f = mat1.to(device, dtype=intermediate_dtype)
        mat2_f = mat2.to(device, dtype=intermediate_dtype)

        if mid is not None:
            mid_f = mid.to(device, dtype=intermediate_dtype)
            final_shape = [mat2_f.shape[1], mat2_f.shape[0], mid_f.shape[2], mid_f.shape[3]]
            mat2_f = (
                torch.mm(
                    mat2_f.transpose(0, 1).flatten(start_dim=1),
                    mid_f.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )

        return torch.mm(
            mat1_f.flatten(start_dim=1),
            mat2_f.flatten(start_dim=1),
        ).reshape(target_shape)
    except Exception:
        return None


def _compute_dynamic_lora_factors(weights, strength):
    unpacked = _unpack_lora_weights(weights)
    if unpacked is None:
        return None

    mat1, mat2, _, mid, dora_scale, reshape = unpacked
    if dora_scale is not None or reshape is not None:
        return None

    try:
        mat2_eff = mat2
        if mid is not None:
            final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
            mat2_eff = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mid.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )

        scale = _compute_lora_scale(weights, strength)
        return mat2_eff * scale, mat1
    except Exception:
        return None


def _get_effective_weight_scale(weight_scale, row_count, offset=None):
    if not isinstance(weight_scale, torch.Tensor) or weight_scale.numel() == 1:
        return weight_scale

    if weight_scale.shape[0] == row_count:
        return weight_scale

    if offset is not None and len(offset) >= 3:
        try:
            dim = int(offset[0])
            start = int(offset[1])
            size = int(offset[2])
            if dim == 0 and size == row_count and start >= 0 and (start + size) <= weight_scale.shape[0]:
                return weight_scale.narrow(0, start, size)
        except Exception:
            pass

    # Unknown mapping between scale rows and weight rows.
    # Fall back to scalar mean to avoid shape/device crashes.
    return weight_scale.float().mean()

def _get_effective_hadanorm_sigma(hadanorm_sigma, input_width, offset=None):
    if not isinstance(hadanorm_sigma, torch.Tensor):
        return None

    if hadanorm_sigma.numel() == input_width:
        return hadanorm_sigma

    normalized = _normalize_dynamic_offset(offset)
    if normalized is None:
        return None

    dim, start, size = normalized
    if dim != 1 or size != input_width:
        return None
    if start < 0 or (start + size) > hadanorm_sigma.numel():
        return None
    return hadanorm_sigma.narrow(0, start, size)

def _rotate_activation_runtime(x: Tensor, h_matrix: Tensor, group_size: int) -> Tensor:
    original_shape = x.shape
    feature_count = original_shape[-1]
    group_count = feature_count // group_size
    grouped_x = x.reshape(*original_shape[:-1], group_count, group_size)
    return torch.matmul(grouped_x, h_matrix).reshape(original_shape)

def _hadamard_offset_supported(offset, input_width: int, group_size: int) -> bool:
    normalized = _normalize_dynamic_offset(offset)
    if normalized is None:
        return True

    dim, start, size = normalized
    if dim == 0:
        return True
    if dim != 1:
        return False

    if size != input_width:
        return False
    if start < 0 or size <= 0:
        return False
    if (start % group_size) != 0 or (size % group_size) != 0:
        return False
    return True

def _transform_weight_like_for_outlier_method(
    weight_like: Tensor,
    outlier_method,
    comp_device: torch.device,
    hadanorm_sigma: Tensor | None = None,
    offset=None,
) -> Tensor:
    method = _normalize_outlier_method(outlier_method)
    if method == OUTLIER_METHOD_NONE:
        return weight_like
    if weight_like.ndim < 2 or weight_like.shape[1] % _QUAROT_GROUP_SIZE != 0:
        return weight_like
    if not _outlier_method_uses_hadamard(method):
        return weight_like
    if not _hadamard_offset_supported(offset, weight_like.shape[1], _QUAROT_GROUP_SIZE):
        normalized = _normalize_dynamic_offset(offset)
        key = normalized if normalized is not None else "unsupported_offset"
        if key not in _QUAROT_OFFSET_WARNED:
            _QUAROT_OFFSET_WARNED.add(key)
            logging.warning(
                f"[INT8 {method}] skipping transform for unsupported offset={normalized} "
                f"weight_shape={tuple(weight_like.shape)}"
            )
        return weight_like

    try:
        if not _QUAROT_AVAILABLE:
            return weight_like
        rotate_dtype = torch.float32 if weight_like.dtype in (torch.float16, torch.bfloat16) else weight_like.dtype
        weight_work = weight_like.to(comp_device, dtype=rotate_dtype)
        if method == OUTLIER_METHOD_HADANORM:
            sigma = _get_effective_hadanorm_sigma(hadanorm_sigma, weight_like.shape[1], offset=offset)
            if sigma is None:
                logging.warning(
                    f"[INT8 HadaNorm] missing sigma for transformed weight_shape={tuple(weight_like.shape)} "
                    f"offset={_normalize_dynamic_offset(offset)}"
                )
                return weight_like
            sigma = sigma.to(comp_device, dtype=rotate_dtype, non_blocking=True).view(1, -1)
            weight_work = weight_work * sigma
        h_matrix = _quarot_build_hadamard(_QUAROT_GROUP_SIZE, device=comp_device, dtype=rotate_dtype)
        rotated = _quarot_rotate_weight(weight_work, h_matrix, group_size=_QUAROT_GROUP_SIZE)
        return rotated.to(weight_like.dtype)
    except Exception:
        return weight_like

def _apply_outlier_activation_transform(
    x: Tensor,
    outlier_method,
    hadamard: Tensor | None = None,
    hadanorm_sigma: Tensor | None = None,
):
    method = _normalize_outlier_method(outlier_method)
    if method == OUTLIER_METHOD_NONE or not isinstance(hadamard, torch.Tensor):
        return x, None
    if x.shape[-1] % _QUAROT_GROUP_SIZE != 0:
        return x, None

    transform_dtype = torch.float32 if x.device.type == "cpu" and x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_work = x.to(transform_dtype) if x.dtype != transform_dtype else x

    if method == OUTLIER_METHOD_HADANORM:
        if not isinstance(hadanorm_sigma, torch.Tensor):
            return x, None
        sigma = hadanorm_sigma
        if sigma.device != x_work.device or sigma.dtype != x_work.dtype:
            sigma = sigma.to(x_work.device, dtype=x_work.dtype, non_blocking=True)
        x_work = x_work / sigma.view(*([1] * (x_work.ndim - 1)), -1)

    h_matrix = hadamard
    if h_matrix.device != x_work.device or h_matrix.dtype != x_work.dtype:
        h_matrix = h_matrix.to(x_work.device, dtype=x_work.dtype, non_blocking=True)

    x_rotated = _rotate_activation_runtime(x_work, h_matrix, _QUAROT_GROUP_SIZE)
    if method == OUTLIER_METHOD_HADANORM:
        correction_source = x_rotated.mean(dim=-2, keepdim=True)
        return x_rotated - correction_source, correction_source

    return x_rotated, None

def _apply_int8_delta_inplace(weight, delta_f, weight_scale, seed, offset=None):
    comp_device = _get_int8_compute_device(weight.device)
    delta_dev = delta_f.to(comp_device)
    effective_scale = _get_effective_weight_scale(weight_scale, delta_dev.shape[0], offset)
    delta_int8 = stochastic_round_int8_delta(delta_dev, effective_scale, seed)
    del delta_dev
    res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
    del delta_int8
    patched_weight = torch.clamp(res, -128, 127).to(torch.int8)
    del res
    if patched_weight.device != weight.device:
        patched_weight = patched_weight.to(weight.device)
    weight.copy_(patched_weight)
    del patched_weight
    return weight

try:
    from comfy.weight_adapter.lora import LoRAAdapter
    _LORA_ADAPTER_AVAILABLE = True
except ImportError:
    _LORA_ADAPTER_AVAILABLE = False

try:
    from comfy.weight_adapter.base import WeightAdapterBase
    _WEIGHT_ADAPTER_BASE_AVAILABLE = True
except ImportError:
    _WEIGHT_ADAPTER_BASE_AVAILABLE = False

if _LORA_ADAPTER_AVAILABLE:
    class INT8LoRAPatchAdapter(LoRAAdapter):
        """
        Specialized LoRA adapter that patches INT8 weights IN-PLACE in INT8 space.
        """
        def __init__(self, loaded_keys, weights, weight_scale, seed=0, use_quarot=False, outlier_method=None, hadanorm_sigma=None):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed
            resolved_method = _normalize_outlier_method(outlier_method)
            if resolved_method == OUTLIER_METHOD_NONE and use_quarot:
                resolved_method = OUTLIER_METHOD_QUAROT
            self.outlier_method = resolved_method
            self.hadanorm_sigma = hadanorm_sigma

        def _calculate_weight_fallback(self, weight, key, strength, strength_model, offset, function, intermediate_dtype, original_weight):
            if weight.dtype != torch.int8:
                return super().calculate_weight(
                    weight,
                    key,
                    strength,
                    strength_model,
                    offset,
                    function,
                    intermediate_dtype,
                    original_weight,
                )

            device = weight.device
            comp_device = _get_int8_compute_device(device)
            effective_scale = _get_effective_weight_scale(self.weight_scale, weight.shape[0], offset)
            base_weight_f = dequantize(weight.to(comp_device), effective_scale).to(intermediate_dtype)
            original_base_f = base_weight_f.clone()

            patched_weight_f = super().calculate_weight(
                base_weight_f,
                key,
                strength,
                strength_model,
                offset,
                function,
                intermediate_dtype,
                original_weight,
            )

            delta_f = patched_weight_f - original_base_f
            del patched_weight_f
            del original_base_f
            del base_weight_f
            delta_f = _transform_weight_like_for_outlier_method(
                delta_f,
                self.outlier_method,
                comp_device,
                hadanorm_sigma=self.hadanorm_sigma,
                offset=offset,
            )
            final_weight = _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
            del delta_f
            return final_weight

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            device = weight.device
            comp_device = _get_int8_compute_device(device)
            lora_diff = _compute_fast_lora_diff(self.weights, weight.shape, comp_device, intermediate_dtype)

            if lora_diff is None:
                return self._calculate_weight_fallback(
                    weight,
                    key,
                    strength,
                    strength_model,
                    offset,
                    function,
                    intermediate_dtype,
                    original_weight,
                )

            scale = _compute_lora_scale(self.weights, strength)
            if weight.dtype == torch.int8:
                delta_f = lora_diff * scale
                del lora_diff
                delta_f = _transform_weight_like_for_outlier_method(
                    delta_f,
                    self.outlier_method,
                    comp_device,
                    hadanorm_sigma=self.hadanorm_sigma,
                    offset=offset,
                )
                final_weight = _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
                del delta_f
                return final_weight
            else:
                final_weight = weight + (lora_diff * scale).to(weight.device, weight.dtype)
                del lora_diff
                return final_weight

    class INT8MergedLoRAPatchAdapter(LoRAAdapter):
        """
        Adapter that merges multiple LoRAs in float space BEFORE applying a single
        stochastic rounding step. This is much more precise for LoRA stacks.
        """
        def __init__(self, patches, weight_scale, seed=0, use_quarot=False, outlier_method=None, hadanorm_sigma=None):
            # We need to satisfy the base LoRAAdapter constructor.
            # We use the first patch's keys/weights as a reference.
            first_patch_adapter = patches[0][0]
            super().__init__(first_patch_adapter.loaded_keys, first_patch_adapter.weights)
            
            # patches is a list of (adapter, strength)
            self.patches = patches
            self.weight_scale = weight_scale
            self.seed = seed
            resolved_method = _normalize_outlier_method(outlier_method)
            if resolved_method == OUTLIER_METHOD_NONE and use_quarot:
                resolved_method = OUTLIER_METHOD_QUAROT
            self.outlier_method = resolved_method
            self.hadanorm_sigma = hadanorm_sigma

        def _calculate_weight_fallback(self, weight, key, strength_model, offset, function, intermediate_dtype, original_weight):
            if weight.dtype == torch.int8:
                device = weight.device
                comp_device = _get_int8_compute_device(device)
                effective_scale = _get_effective_weight_scale(self.weight_scale, weight.shape[0], offset)
                base_weight_f = dequantize(weight.to(comp_device), effective_scale).to(intermediate_dtype)
                original_base_f = base_weight_f.clone()
                patched_weight_f = base_weight_f
            else:
                patched_weight_f = weight.to(dtype=intermediate_dtype).clone()
                original_base_f = None

            for adapter, lora_strength in self.patches:
                patched_weight_f = adapter.calculate_weight(
                    patched_weight_f,
                    key,
                    lora_strength,
                    strength_model,
                    offset,
                    function,
                    intermediate_dtype,
                    original_weight,
                )

            if weight.dtype == torch.int8:
                delta_f = patched_weight_f - original_base_f
                del patched_weight_f
                del original_base_f
                delta_f = _transform_weight_like_for_outlier_method(
                    delta_f,
                    self.outlier_method,
                    comp_device,
                    hadanorm_sigma=self.hadanorm_sigma,
                    offset=offset,
                )
                final_weight = _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
                del delta_f
                return final_weight

            final_weight = patched_weight_f.to(weight.device, weight.dtype)
            del patched_weight_f
            return final_weight

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            # Note: 'strength' from ComfyUI is ignored here as we use internal lora_strengths
            device = weight.device
            comp_device = _get_int8_compute_device(device)
            
            total_delta_f = None
            
            for adapter, lora_strength in self.patches:
                if not isinstance(adapter, LoRAAdapter):
                    return self._calculate_weight_fallback(
                        weight,
                        key,
                        strength_model,
                        offset,
                        function,
                        intermediate_dtype,
                        original_weight,
                    )

                delta = _compute_fast_lora_diff(adapter.weights, weight.shape, comp_device, intermediate_dtype)
                if delta is None:
                    return self._calculate_weight_fallback(
                        weight,
                        key,
                        strength_model,
                        offset,
                        function,
                        intermediate_dtype,
                        original_weight,
                    )

                scale = _compute_lora_scale(adapter.weights, lora_strength)
                
                if total_delta_f is None:
                    total_delta_f = delta * scale
                else:
                    total_delta_f += delta * scale
                del delta
            
            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                # One single stochastic rounding step for all combined LoRAs
                total_delta_f = _transform_weight_like_for_outlier_method(
                    total_delta_f,
                    self.outlier_method,
                    comp_device,
                    hadanorm_sigma=self.hadanorm_sigma,
                    offset=offset,
                )
                final_weight = _apply_int8_delta_inplace(weight, total_delta_f, self.weight_scale, self.seed, offset)
                del total_delta_f
                return final_weight
            else:
                final_weight = weight + total_delta_f.to(device, weight.dtype)
                del total_delta_f
                return final_weight

if _WEIGHT_ADAPTER_BASE_AVAILABLE:
    class INT8WeightPatchAdapter(WeightAdapterBase):
        name = "int8_weight_patch"

        def __init__(self, base_adapter, weight_scale, seed=0, use_quarot=False, outlier_method=None, hadanorm_sigma=None):
            self.base_adapter = base_adapter
            self.weight_scale = weight_scale
            self.seed = seed
            resolved_method = _normalize_outlier_method(outlier_method)
            if resolved_method == OUTLIER_METHOD_NONE and use_quarot:
                resolved_method = OUTLIER_METHOD_QUAROT
            self.outlier_method = resolved_method
            self.hadanorm_sigma = hadanorm_sigma
            self.loaded_keys = getattr(base_adapter, "loaded_keys", set())
            self.weights = getattr(base_adapter, "weights", ())

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            if weight.dtype != torch.int8:
                return self.base_adapter.calculate_weight(
                    weight,
                    key,
                    strength,
                    strength_model,
                    offset,
                    function,
                    intermediate_dtype,
                    original_weight,
                )

            comp_device = _get_int8_compute_device(weight.device)
            effective_scale = _get_effective_weight_scale(self.weight_scale, weight.shape[0], offset)
            base_weight_f = dequantize(weight.to(comp_device), effective_scale).to(intermediate_dtype)
            original_base_f = base_weight_f.clone()

            patched_weight_f = self.base_adapter.calculate_weight(
                base_weight_f,
                key,
                strength,
                strength_model,
                offset,
                function,
                intermediate_dtype,
                original_weight,
            )
            if patched_weight_f is None:
                return weight

            delta_f = patched_weight_f - original_base_f
            del patched_weight_f
            del original_base_f
            del base_weight_f
            delta_f = _transform_weight_like_for_outlier_method(
                delta_f,
                self.outlier_method,
                comp_device,
                hadanorm_sigma=self.hadanorm_sigma,
                offset=offset,
            )
            final_weight = _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
            del delta_f
            return final_weight
else:
    INT8WeightPatchAdapter = None


# =============================================================================
# Dynamic LoRA Synchronization Hook
# =============================================================================

class DynamicLoRAHook:
    """
    Hook registered on the diffusion_model to synchronize dynamic LoRA attributes
    with the current ModelPatcher context at the start of each forward pass.
    """
    def __init__(self):
        self.current_lora_id = None

    @staticmethod
    def _compute_lora_id(dynamic_loras):
        if not dynamic_loras:
            return None

        signature = []
        # Prefer a loader-created UUID so cloned dicts keep a stable identity while
        # newly loaded or changed LoRA stacks force recomposition.
        for entry in dynamic_loras:
            patch_uuid = entry.get("patch_uuid", None)
            if patch_uuid is not None:
                signature.append((str(patch_uuid), float(entry.get("strength", 0.0))))
                continue

            patches = entry.get("patches", {})
            if isinstance(patches, dict):
                patch_items = tuple(
                    sorted(
                        (
                            str(raw_key),
                            id(adapter),
                        )
                        for raw_key, adapter in patches.items()
                    )
                )
            else:
                patch_items = ()
            signature.append((
                entry.get("name", ""),
                float(entry.get("strength", 0.0)),
                patch_items,
            ))

        return hash(tuple(
            signature
        ))

    @classmethod
    @_disable_torch_compile
    def sync_from_transformer_options(cls, diffusion_model, transformer_options):
        if transformer_options is None:
            transformer_options = {}

        dynamic_loras = transformer_options.get("dynamic_loras", [])
        target_modules = []

        if diffusion_model is not None:
            target_modules.append(diffusion_model)
            orig_mod = getattr(diffusion_model, "_orig_mod", None)
            if orig_mod is not None:
                target_modules.append(orig_mod)

        for module in target_modules:
            hook = cls.register(module)
            lora_id = cls._compute_lora_id(dynamic_loras)
            if lora_id == hook.current_lora_id:
                continue
            hook.apply_composition(module, dynamic_loras)
            hook.current_lora_id = lora_id

    @_disable_torch_compile
    def pre_forward(self, module, input_args, input_kwargs):
        # 1. Try to find transformer_options
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            # Fallback for models that pass it in context
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]
        
        dynamic_loras = transformer_options.get("dynamic_loras", [])
        
        # 2. Generate a stable ID for this set of LoRAs
        lora_id = self._compute_lora_id(dynamic_loras)
        
        if lora_id == self.current_lora_id:
            return None # Already synchronized
            
        # 3. Synchronize all linear layers
        self.apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    @_disable_torch_compile
    def apply_composition(self, diffusion_model, dynamic_loras):
        def normalize_patch_key(raw_key):
            key = raw_key[0] if isinstance(raw_key, tuple) else raw_key
            if not isinstance(key, str):
                return None

            if key.endswith(".weight"):
                key = key[:-7]

            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model."):]
            elif key.startswith("model.diffusion_model."):
                key = key[len("model.diffusion_model."):]
            elif key.startswith("model."):
                key = key[len("model."):]

            if key.startswith("_orig_mod.diffusion_model."):
                key = key[len("_orig_mod.diffusion_model."):]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]

            return key

        def normalize_module_name(module_name):
            name = module_name
            if name.startswith("_orig_mod.diffusion_model."):
                name = name[len("_orig_mod.diffusion_model."):]
            elif name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
            elif name.startswith("diffusion_model."):
                name = name[len("diffusion_model."):]
            elif name.startswith("model.diffusion_model."):
                name = name[len("model.diffusion_model."):]
            elif name.startswith("model."):
                name = name[len("model."):]
            return name

        # Pre-group patches by layer
        layer_patches = {}
        if dynamic_loras:
            for entry in dynamic_loras:
                strength = entry["strength"]
                for key, adapter in entry["patches"].items():
                    normalized_key = normalize_patch_key(key)
                    if normalized_key is None:
                        continue
                    offset = key[1] if isinstance(key, tuple) and len(key) > 1 else None
                    if normalized_key not in layer_patches:
                        layer_patches[normalized_key] = []
                    layer_patches[normalized_key].append((adapter, strength, offset))

        # Update all modules
        candidate_modules = 0
        matched_modules = 0
        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            candidate_modules += 1
            
            normalized_name = normalize_module_name(name)
            patches = layer_patches.get(normalized_name)
            
            if not patches:
                module.dynamic_lora_entries = None
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            # Compose
            matched_modules += 1
            entries = []
            module_outlier_method = _get_module_outlier_method(module)
            hadanorm_sigma = getattr(module, "hadanorm_sigma", None)
            for adapter, strength, offset in patches:
                if not _LORA_ADAPTER_AVAILABLE or not isinstance(adapter, LoRAAdapter):
                    if _DYNAMIC_LORA_DEBUG:
                        logging.warning(f"[INT8 Dynamic LoRA] skipping non-LoRA adapter for module={normalized_name}")
                    continue

                factors = _compute_dynamic_lora_factors(adapter.weights, strength)
                if factors is None:
                    if _DYNAMIC_LORA_DEBUG:
                        logging.warning(
                            f"[INT8 Dynamic LoRA] skipping unsupported LoRA format for module={normalized_name} "
                            f"offset={offset}"
                        )
                    continue

                curr_A, curr_B = factors
                curr_A = _transform_weight_like_for_outlier_method(
                    curr_A,
                    module_outlier_method,
                    curr_A.device,
                    hadanorm_sigma=hadanorm_sigma,
                    offset=offset,
                )

                entries.append({
                    "A": curr_A,
                    "B": curr_B,
                    "offset": offset,
                })

            if entries:
                module_weight = getattr(module, "weight", None)
                device = module_weight.device if isinstance(module_weight, torch.Tensor) else torch.device("cpu")
                module.dynamic_lora_entries = [
                    {
                        "A": entry["A"].to(device),
                        "B": entry["B"].to(device),
                        "offset": entry["offset"],
                    }
                    for entry in entries
                ]
            else:
                module.dynamic_lora_entries = None

            # Keep legacy fields unset to force offset-aware entry path.
            module.lora_A = None
            module.lora_B = None
            module.lora_alpha = None

        if _DYNAMIC_LORA_DEBUG:
            print(
                f"[INT8 Dynamic LoRA] candidate_modules={candidate_modules} "
                f"patch_keys={len(layer_patches)} matched_modules={matched_modules}"
            )

    @classmethod
    def register(cls, diffusion_model):
        if not hasattr(diffusion_model, "_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
        return diffusion_model._dynamic_lora_hook


# =============================================================================
# Int8TensorwiseOps - ComfyUI Custom Operations
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """
        Custom ComfyUI operations for INT8 tensorwise quantization.
        """
        excluded_names = []
        dynamic_quantize = False # Manual toggle for on-the-fly quantization
        outlier_method = OUTLIER_METHOD_NONE
        use_triton = True
        runtime_backend = DEFAULT_INT8_BACKEND
        runtime_uses_triton = DEFAULT_INT8_BACKEND in (INT8_BACKEND_TRITON, INT8_BACKEND_TRITON_LEGACY_UNSAFE)
        runtime_uses_legacy_triton = False
        prepack_int8_weights = False
        small_batch_fallback_mode = SMALL_BATCH_FALLBACK_SMALL_LAYERS
        _is_prequantized = None # Global flag for current load
        _otf_progress_total = None
        _otf_progress_processed = 0
        _otf_progress_quantized = 0
        _otf_progress_outlier = 0
        _otf_progress_last_bucket = -1
        _runtime_stats = {}

        @classmethod
        def reset_otf_progress(cls):
            cls._otf_progress_total = None
            cls._otf_progress_processed = 0
            cls._otf_progress_quantized = 0
            cls._otf_progress_outlier = 0
            cls._otf_progress_last_bucket = -1

        @classmethod
        def _init_otf_progress(cls, state_dict):
            if cls._otf_progress_total is not None:
                return

            total_estimate = 1
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if not key.endswith("weight"):
                    continue
                if value.dtype in (torch.float16, torch.bfloat16, torch.float32) or _is_float8_dtype(value.dtype):
                    total_estimate += 1

            cls._otf_progress_total = max(1, int(total_estimate))
            print(f"[INT8 OTF] Starting quantization scan (estimated layers: {cls._otf_progress_total})")

        @classmethod
        def _update_otf_progress(cls, *, quantized: bool, outlier_adjusted: bool):
            total = cls._otf_progress_total if cls._otf_progress_total is not None else 1
            cls._otf_progress_processed += 1
            if quantized:
                cls._otf_progress_quantized += 1
            if outlier_adjusted:
                cls._otf_progress_outlier += 1

            percent = min(100, int((cls._otf_progress_processed * 100) / max(1, total)))
            bucket = percent // 5
            if bucket != cls._otf_progress_last_bucket:
                cls._otf_progress_last_bucket = bucket
                print(
                    f"[INT8 OTF] {percent:3d}% "
                    f"({cls._otf_progress_processed}/{total}) "
                    f"quantized={cls._otf_progress_quantized} "
                    f"outlier_adjusted={cls._otf_progress_outlier}"
                )

        @classmethod
        def summarize_otf_progress(cls):
            if cls._otf_progress_processed <= 0:
                return
            print(
                "[INT8 OTF] Complete "
                f"(processed={cls._otf_progress_processed}, "
                f"quantized={cls._otf_progress_quantized}, "
                f"outlier_adjusted={cls._otf_progress_outlier})"
            )

        @classmethod
        def reset_runtime_stats(cls):
            cls._runtime_stats = {
                "linear_calls": 0,
                "triton": 0,
                "triton_legacy_unsafe": 0,
                "torch_int_mm": 0,
                "small_batch_fallback": 0,
                "prepacked": 0,
            }

        @classmethod
        def _increment_runtime_stat(cls, key):
            if not _RUNTIME_STATS_ENABLED:
                return
            try:
                if torch.compiler.is_compiling():
                    return
            except Exception:
                pass
            if not cls._runtime_stats:
                cls.reset_runtime_stats()
            cls._runtime_stats[key] = cls._runtime_stats.get(key, 0) + 1

        @classmethod
        def print_runtime_stats(cls, prefix="[INT8 Runtime]"):
            if not _RUNTIME_STATS_ENABLED:
                return
            print(
                f"{prefix} backend={cls.runtime_backend} "
                f"small_batch_fallback={cls.small_batch_fallback_mode} "
                f"prepack_int8_weights={cls.prepack_int8_weights} "
                f"calls={cls._runtime_stats}"
            )
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self.register_buffer("weight_packed", None)
                self.register_buffer("quarot_hadamard", None)
                self.register_buffer("hadanorm_sigma", None)
                self._is_quantized = False
                self._is_per_row = False
                self._use_quarot = False
                self._outlier_method = OUTLIER_METHOD_NONE
                self.compute_dtype = torch.bfloat16
                self.dynamic_lora_entries = None
                self.lora_A = None
                self.lora_B = None
                self.lora_alpha = None
            
            def reset_parameters(self):
                return None
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                hadamard_key = prefix + "quarot_hadamard"
                hadanorm_sigma_key = prefix + "hadanorm_sigma"
                bias_key = prefix + "bias"
                
                weight_scale = state_dict.pop(scale_key, None)
                stored_hadamard = state_dict.pop(hadamard_key, None)
                stored_hadanorm_sigma = state_dict.pop(hadanorm_sigma_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Load Quantized
                        self._is_quantized = True
                        self._use_quarot = False
                        self.quarot_hadamard = None
                        self.hadanorm_sigma = None
                        self._outlier_method = OUTLIER_METHOD_NONE
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        self.weight_packed = _prepack_int8_weight(weight_tensor) if Int8TensorwiseOps.prepack_int8_weights else None
                        Int8TensorwiseOps._is_prequantized = True # Found a quantized layer
                        
                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self.weight_scale = weight_scale.float().reshape(1)
                                self._is_per_row = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.weight_scale = weight_scale.float()
                                self._is_per_row = True
                            else:
                                self.weight_scale = weight_scale.float()
                                self._is_per_row = False
                        else:
                            self.weight_scale = torch.tensor([float(weight_scale)], dtype=torch.float32)
                            self._is_per_row = False

                        if isinstance(stored_hadanorm_sigma, torch.Tensor):
                            self.hadanorm_sigma = stored_hadanorm_sigma.float()
                            self._outlier_method = OUTLIER_METHOD_HADANORM
                        if isinstance(stored_hadamard, torch.Tensor):
                            self.quarot_hadamard = stored_hadamard.float()
                            if self._outlier_method == OUTLIER_METHOD_NONE:
                                self._use_quarot = True
                                self._outlier_method = OUTLIER_METHOD_QUAROT
                            
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32) or _is_float8_dtype(weight_tensor.dtype):
                        # Load High-Precision
                        # Detect if the model is pre-quantized if we don't know yet
                        if Int8TensorwiseOps._is_prequantized is None:
                            # Robust detection: scan keys and a sample of values
                            is_prequant = False
                            for k in state_dict.keys():
                                if "weight_scale" in k or "comfy_quant" in k:
                                    is_prequant = True
                                    break
                            
                            if not is_prequant:
                                # Fallback: scan a sample of values for int8 tensors
                                for i, v in enumerate(state_dict.values()):
                                    if i > 200: break # Safety limit
                                    if getattr(v, "dtype", None) == torch.int8:
                                        is_prequant = True
                                        break
                            Int8TensorwiseOps._is_prequantized = is_prequant

                        track_progress = Int8TensorwiseOps.dynamic_quantize and not Int8TensorwiseOps._is_prequantized
                        if track_progress:
                            Int8TensorwiseOps._init_otf_progress(state_dict)

                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        quantized_now = False
                        outlier_adjusted_now = False
                        
                        if is_excluded or is_dim1 or Int8TensorwiseOps._is_prequantized or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self._is_per_row = False
                            self._use_quarot = False
                            self.quarot_hadamard = None
                            self.hadanorm_sigma = None
                            self._outlier_method = OUTLIER_METHOD_NONE
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                            self.weight_packed = None
                            #print("Not quantizing", prefix)
                        else:
                            # Quantize on the fly (per-row, including FP8 -> INT8).
                            device = _get_int8_compute_device(weight_tensor.device)
                            w_gpu = weight_tensor.to(device, non_blocking=True)
                            outlier_method = _normalize_outlier_method(Int8TensorwiseOps.outlier_method)
                            if _is_float8_dtype(w_gpu.dtype):
                                w_gpu = w_gpu.to(torch.float16 if device.type == "cuda" else torch.float32)
                            elif outlier_method != OUTLIER_METHOD_NONE and w_gpu.dtype in (torch.float16, torch.bfloat16):
                                w_gpu = w_gpu.float()
                            self._use_quarot = False
                            self.quarot_hadamard = None
                            self.hadanorm_sigma = None
                            self._outlier_method = OUTLIER_METHOD_NONE

                            if (
                                outlier_method != OUTLIER_METHOD_NONE
                                and _QUAROT_AVAILABLE
                                and w_gpu.ndim == 2
                                and w_gpu.shape[1] % _QUAROT_GROUP_SIZE == 0
                            ):
                                try:
                                    h_matrix = _quarot_build_hadamard(_QUAROT_GROUP_SIZE, device=device, dtype=w_gpu.dtype)
                                    if outlier_method == OUTLIER_METHOD_HADANORM:
                                        hadanorm_sigma = _compute_hadanorm_sigma(w_gpu).to(device=device, dtype=w_gpu.dtype)
                                        w_gpu = w_gpu * hadanorm_sigma.view(1, -1)
                                        self.hadanorm_sigma = hadanorm_sigma.detach().cpu()
                                    w_gpu = _quarot_rotate_weight(w_gpu, h_matrix, group_size=_QUAROT_GROUP_SIZE)
                                    self.quarot_hadamard = h_matrix.detach().cpu()
                                    self._use_quarot = outlier_method == OUTLIER_METHOD_QUAROT
                                    self._outlier_method = outlier_method
                                except Exception:
                                    self._use_quarot = False
                                    self.quarot_hadamard = None
                                    self.hadanorm_sigma = None
                                    self._outlier_method = OUTLIER_METHOD_NONE

                            q_weight, q_scale = quantize_int8_rowwise(w_gpu)
                            #print("Quantizing", prefix)
                            
                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.weight_packed = _prepack_int8_weight(self.weight) if Int8TensorwiseOps.prepack_int8_weights else None
                            self.weight_scale = (
                                q_scale.cpu()
                                if isinstance(q_scale, torch.Tensor)
                                else torch.tensor([float(q_scale)], dtype=torch.float32)
                            )
                            self._is_quantized = True
                            self._is_per_row = self.weight_scale.dim() == 2 and self.weight_scale.shape[1] == 1
                            quantized_now = True
                            outlier_adjusted_now = self._outlier_method != OUTLIER_METHOD_NONE

                        if track_progress:
                            Int8TensorwiseOps._update_otf_progress(
                                quantized=quantized_now,
                                outlier_adjusted=outlier_adjusted_now,
                            )
                    else:
                        self._is_quantized = False
                        self._is_per_row = False
                        self._use_quarot = False
                        self.quarot_hadamard = None
                        self.hadanorm_sigma = None
                        self._outlier_method = OUTLIER_METHOD_NONE
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        self.weight_packed = None
                else:
                    missing_keys.append(weight_key)
                
                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

            def _replace_weight(self, new_weight, inplace_update=False):
                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)
                self.weight_packed = (
                    _prepack_int8_weight(self.weight)
                    if self._is_quantized and Int8TensorwiseOps.prepack_int8_weights
                    else None
                )

            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                target_device = _weight.device if isinstance(_weight, torch.Tensor) else self.weight.device
                if self.weight.device == target_device:
                    return self.weight.clone()
                return self.weight.to(target_device, non_blocking=True).clone()

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight

                    if inplace_update:
                        self._replace_weight(new_weight, inplace_update=True)
                    else:
                        self._replace_weight(new_weight)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight

                    if inplace_update:
                        self._replace_weight(out_weight, inplace_update=True)
                    else:
                        self._replace_weight(out_weight)
                    return

                # Re-quantize if fallback occurred
                new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
                
                if return_weight:
                    return new_weight

                if inplace_update:
                    self._replace_weight(new_weight, inplace_update=True)
                else:
                    self._replace_weight(new_weight)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None: return None
                
                new_bias = out_bias
                if return_weight:
                    return new_bias

                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(new_bias)
                else:
                    self.bias = nn.Parameter(new_bias, requires_grad=False)

            @_disable_torch_compile
            def forward(self, x: Tensor) -> Tensor:
                """Fast forward using torch._int_mm for quantized weights."""
                
                if not self._is_quantized:
                    weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out
                
                # 1. Move weight/bias/scale to device (non_blocking)
                weight = self.weight if self.weight.device == x.device else self.weight.to(x.device, non_blocking=True)
                if self.bias is None:
                    bias = None
                else:
                    bias = self.bias if self.bias.device == x.device else self.bias.to(x.device, non_blocking=True)
                
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    if w_scale.device != x.device:
                        w_scale = w_scale.to(x.device, non_blocking=True)
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                x_shape = x.shape
                outlier_method = _get_module_outlier_method(self)
                x_transformed, correction_source = _apply_outlier_activation_transform(
                    x,
                    outlier_method,
                    hadamard=self.quarot_hadamard,
                    hadanorm_sigma=self.hadanorm_sigma,
                )
                x_2d = x_transformed.reshape(-1, x_shape[-1])

                use_triton = Int8TensorwiseOps.runtime_uses_triton
                legacy_triton_unsafe = Int8TensorwiseOps.runtime_uses_legacy_triton
                weight_packed = (
                    _get_prepacked_weight(self, x.device)
                    if use_triton and Int8TensorwiseOps.prepack_int8_weights
                    else None
                )
                
                small_batch_threshold = _get_small_batch_fallback_threshold(self)
                use_small_batch_fallback = small_batch_threshold > 0 and x_2d.shape[0] <= small_batch_threshold
                correction_2d = None
                input_2d = x_2d
                matmul_bias = bias
                if correction_source is not None:
                    correction_2d = correction_source.reshape(-1, correction_source.shape[-1])
                    input_2d = torch.cat((x_2d, correction_2d), dim=0)
                    matmul_bias = None

                if use_small_batch_fallback:
                    Int8TensorwiseOps._increment_runtime_stat("linear_calls")
                    Int8TensorwiseOps._increment_runtime_stat("small_batch_fallback")
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(input_2d.dtype)
                    bias_typed = matmul_bias.to(input_2d.dtype) if matmul_bias is not None else None
                    y = F.linear(input_2d, w_float, bias_typed)
                else:
                    Int8TensorwiseOps._increment_runtime_stat("linear_calls")
                    if use_triton:
                        if legacy_triton_unsafe:
                            Int8TensorwiseOps._increment_runtime_stat("triton_legacy_unsafe")
                        else:
                            Int8TensorwiseOps._increment_runtime_stat("triton")
                        if isinstance(weight_packed, torch.Tensor):
                            Int8TensorwiseOps._increment_runtime_stat("prepacked")
                    else:
                        Int8TensorwiseOps._increment_runtime_stat("torch_int_mm")

                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(
                            input_2d,
                            weight,
                            w_scale,
                            matmul_bias,
                            compute_dtype,
                            use_triton=use_triton,
                            weight_packed=weight_packed,
                            legacy_triton_unsafe=legacy_triton_unsafe,
                        )
                    else:
                        y = int8_forward_dynamic(
                            input_2d,
                            weight,
                            w_scale,
                            matmul_bias,
                            compute_dtype,
                            use_triton=use_triton,
                            weight_packed=weight_packed,
                            legacy_triton_unsafe=legacy_triton_unsafe,
                        )

                if correction_2d is not None:
                    y, correction = y.split((x_2d.shape[0], correction_2d.shape[0]), dim=0)
                    if bias is not None:
                        y = y + bias.to(y.dtype)
                    y_view = y.reshape(*x_shape[:-1], y.shape[-1])
                    correction = correction.reshape(*correction_source.shape[:-1], correction.shape[-1])
                    y_view = y_view + correction.to(y_view.dtype)
                else:
                    y_view = y.reshape(*x_shape[:-1], y.shape[-1])
                
                # Dynamic LoRA Path
                y_view = apply_dynamic_lora_delta(
                    x_input=x_transformed,
                    y=y_view,
                    lora_A=self.lora_A,
                    lora_B=self.lora_B,
                    lora_alpha=self.lora_alpha,
                    lora_entries=self.dynamic_lora_entries,
                    device=x.device,
                    correction_x=correction_source,
                )
                
                return y_view.reshape(*x_shape[:-1], y_view.shape[-1])
        
        # Pass-through for other layers
        class GroupNorm(manual_cast.GroupNorm): pass
        class LayerNorm(manual_cast.LayerNorm): pass
        class Conv2d(manual_cast.Conv2d): pass
        class Conv3d(manual_cast.Conv3d): pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding): pass
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2: return cls.Conv2d(*args, **kwargs)
            elif dims == 3: return cls.Conv3d(*args, **kwargs)
            else: raise ValueError(f"unsupported dimensions: {dims}")
