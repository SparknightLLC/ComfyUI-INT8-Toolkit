import torch
import os
import logging
from torch import Tensor, nn
import torch.nn.functional as F

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
    _disable_torch_compile = torch.compiler.disable
except Exception:
    def _disable_torch_compile(fn):
        return fn

try:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = max(0, int(os.environ.get("INT8_SMALL_BATCH_FALLBACK_MAX_ROWS", "16")))
except ValueError:
    _SMALL_BATCH_FALLBACK_MAX_ROWS = 16

_DYNAMIC_LORA_DEBUG = os.environ.get("INT8_DYNAMIC_LORA_DEBUG", "0") == "1"

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

def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FLOAT8_DTYPES

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
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# --- LinearW8A8 Core ---

@torch.no_grad()
@_disable_torch_compile
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    
    # --- FAST PATH: Triton Fused Kernel ---
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    if isinstance(weight_scale, torch.Tensor) and weight_scale.device != x.device:
        weight_scale = weight_scale.to(x.device, non_blocking=True)
    
    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)
    
    # Dequantize: (res * weight_scale * x_scale)
    # Note: Creating intermediate Float tensors here is VRAM heavy
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

@torch.no_grad()
@_disable_torch_compile
def int8_forward_dynamic_per_row(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization and per-row weight quantization."""

    # --- FAST PATH: Triton Fused Kernel (per-row) ---
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    if weight_scale.device != x.device:
        weight_scale = weight_scale.to(x.device, non_blocking=True)

    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)

    # Dequantize with per-row weight scales
    # res[i, j] = sum_k(x_8[i, k] * weight[j, k]) * x_scale[i] * weight_scale[j]
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)

    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled

@torch.no_grad()
@_disable_torch_compile
def apply_dynamic_lora_delta(
    x_2d: Tensor,
    y: Tensor,
    lora_A: Tensor | None,
    lora_B: Tensor | None,
    lora_alpha,
    lora_entries,
    device: torch.device,
) -> Tensor:
    if lora_entries:
        for entry in lora_entries:
            entry_A = entry.get("A")
            entry_B = entry.get("B")
            offset = entry.get("offset")
            if entry_A is None or entry_B is None:
                continue

            # Cache per-device tensors to avoid repeated CPU->GPU copies every forward.
            lA = entry.get("_A_cached")
            lB = entry.get("_B_cached")
            if lA is None or lA.device != device:
                lA = entry_A if entry_A.device == device else entry_A.to(device, non_blocking=True)
                entry["_A_cached"] = lA
            if lB is None or lB.device != device:
                lB = entry_B if entry_B.device == device else entry_B.to(device, non_blocking=True)
                entry["_B_cached"] = lB
            x_src = x_2d

            if offset is not None and len(offset) >= 3 and int(offset[0]) == 1:
                start = int(offset[1])
                length = int(offset[2])
                if start >= 0 and length > 0 and (start + length) <= x_2d.shape[-1]:
                    x_src = x_2d.narrow(-1, start, length)
                else:
                    if _DYNAMIC_LORA_DEBUG:
                        print(f"[INT8 Dynamic LoRA] skipping invalid input offset={offset} for x shape={tuple(x_2d.shape)}")
                    continue

            lora_x = F.linear(x_src.to(lA.dtype), lA)
            lora_y = F.linear(lora_x, lB).to(y.dtype)

            if offset is not None and len(offset) >= 3 and int(offset[0]) == 0:
                start = int(offset[1])
                length = int(offset[2])
                if lora_y.shape[-1] != length:
                    if _DYNAMIC_LORA_DEBUG:
                        print(
                            f"[INT8 Dynamic LoRA] skipping mismatched output slice "
                            f"offset={offset} lora_y={tuple(lora_y.shape)} y={tuple(y.shape)}"
                        )
                    continue
                if start >= 0 and length > 0 and (start + length) <= y.shape[-1]:
                    y.narrow(-1, start, length).add_(lora_y)
                else:
                    if _DYNAMIC_LORA_DEBUG:
                        print(f"[INT8 Dynamic LoRA] skipping invalid output offset={offset} for y shape={tuple(y.shape)}")
                continue

            if lora_y.shape[-1] != y.shape[-1]:
                if _DYNAMIC_LORA_DEBUG:
                    print(
                        f"[INT8 Dynamic LoRA] skipping mismatched full add "
                        f"lora_y={tuple(lora_y.shape)} y={tuple(y.shape)} offset={offset}"
                    )
                continue

            y.add_(lora_y)

        return y

    if lora_A is None or lora_B is None:
        return y

    lA = lora_A if lora_A.device == device else lora_A.to(device, non_blocking=True)
    lB = lora_B if lora_B.device == device else lora_B.to(device, non_blocking=True)

    lora_x = F.linear(x_2d.to(lA.dtype), lA)
    lora_y = F.linear(lora_x, lB)

    if lora_alpha is not None:
        lora_y = lora_y * lora_alpha

    return y + lora_y.to(y.dtype)




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


def _apply_int8_delta_inplace(weight, delta_f, weight_scale, seed, offset=None):
    comp_device = torch.device("cuda") if torch.cuda.is_available() else weight.device
    delta_dev = delta_f.to(comp_device)
    effective_scale = _get_effective_weight_scale(weight_scale, delta_dev.shape[0], offset)
    delta_int8 = stochastic_round_int8_delta(delta_dev, effective_scale, seed)
    res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
    patched_weight = torch.clamp(res, -128, 127).to(torch.int8)
    if patched_weight.device != weight.device:
        patched_weight = patched_weight.to(weight.device)
    weight.copy_(patched_weight)
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
        def __init__(self, loaded_keys, weights, weight_scale, seed=0):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed

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
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
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
            return _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
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
                return _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
            else:
                return weight + (lora_diff * scale).to(weight.device, weight.dtype)

    class INT8MergedLoRAPatchAdapter(LoRAAdapter):
        """
        Adapter that merges multiple LoRAs in float space BEFORE applying a single
        stochastic rounding step. This is much more precise for LoRA stacks.
        """
        def __init__(self, patches, weight_scale, seed=0):
            # We need to satisfy the base LoRAAdapter constructor.
            # We use the first patch's keys/weights as a reference.
            first_patch_adapter = patches[0][0]
            super().__init__(first_patch_adapter.loaded_keys, first_patch_adapter.weights)
            
            # patches is a list of (adapter, strength)
            self.patches = patches
            self.weight_scale = weight_scale
            self.seed = seed

        def _calculate_weight_fallback(self, weight, key, strength_model, offset, function, intermediate_dtype, original_weight):
            if weight.dtype == torch.int8:
                device = weight.device
                comp_device = torch.device("cuda") if torch.cuda.is_available() else device
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
                return _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)

            return patched_weight_f.to(weight.device, weight.dtype)

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            # Note: 'strength' from ComfyUI is ignored here as we use internal lora_strengths
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
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
            
            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                # One single stochastic rounding step for all combined LoRAs
                return _apply_int8_delta_inplace(weight, total_delta_f, self.weight_scale, self.seed, offset)
            else:
                return weight + total_delta_f.to(device, weight.dtype)

if _WEIGHT_ADAPTER_BASE_AVAILABLE:
    class INT8WeightPatchAdapter(WeightAdapterBase):
        name = "int8_weight_patch"

        def __init__(self, base_adapter, weight_scale, seed=0):
            self.base_adapter = base_adapter
            self.weight_scale = weight_scale
            self.seed = seed
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

            comp_device = torch.device("cuda") if torch.cuda.is_available() else weight.device
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
            return _apply_int8_delta_inplace(weight, delta_f, self.weight_scale, self.seed, offset)
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
        # Use stable values so per-step dict/list cloning does not force recompose.
        return hash(tuple(
            (
                entry.get("name", ""),
                float(entry.get("strength", 0.0)),
                len(entry.get("patches", {})),
            )
            for entry in dynamic_loras
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
        _is_prequantized = None # Global flag for current load
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self._is_quantized = False
                self._is_per_row = False
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
                bias_key = prefix + "bias"
                
                weight_scale = state_dict.pop(scale_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Load Quantized
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
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

                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1 or Int8TensorwiseOps._is_prequantized or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self._is_per_row = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                            #print("Not quantizing", prefix)
                        else:
                            # Quantize on the fly (per-row, including FP8 -> INT8).
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            w_gpu = weight_tensor.to(device, non_blocking=True)
                            q_weight, q_scale = quantize_int8_rowwise(w_gpu)
                            #print("Quantizing", prefix)
                            
                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.weight_scale = (
                                q_scale.cpu()
                                if isinstance(q_scale, torch.Tensor)
                                else torch.tensor([float(q_scale)], dtype=torch.float32)
                            )
                            self._is_quantized = True
                            self._is_per_row = self.weight_scale.dim() == 2 and self.weight_scale.shape[1] == 1
                    else:
                        self._is_quantized = False
                        self._is_per_row = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)
                
                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

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
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight

                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # Re-quantize if fallback occurred
                new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
                
                if return_weight:
                    return new_weight

                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

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
                x_2d = x.reshape(-1, x_shape[-1])
                
                use_small_batch_fallback = _SMALL_BATCH_FALLBACK_MAX_ROWS > 0 and x_2d.shape[0] <= _SMALL_BATCH_FALLBACK_MAX_ROWS
                if use_small_batch_fallback:
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)
                else:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                
                # Dynamic LoRA Path
                y = apply_dynamic_lora_delta(
                    x_2d=x_2d,
                    y=y,
                    lora_A=self.lora_A,
                    lora_B=self.lora_B,
                    lora_alpha=self.lora_alpha,
                    lora_entries=self.dynamic_lora_entries,
                    device=x.device,
                )
                
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
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
