"""Temporary torch.compile compatibility boundary for comfy-kitchen W4A8.

Retirement path:
1. Upstream registers ``comfy_kitchen::w4a8_int8_linear`` with a FakeTensor
   implementation.
2. ``get_compile_support_source`` detects that operator and the Toolkit routes
   W4A8 through the regular ``F.linear`` layout dispatch again.
3. This module and the corresponding fallback warning can then be removed.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F


_UPSTREAM_OP_NAME = "comfy_kitchen::w4a8_int8_linear"
_TOOLKIT_OP_NAME = "comfyui_quantization_toolkit::w4a8_int8_linear"
_TORCH_STATE_ATTRIBUTE = "_comfyui_quantization_toolkit_w4a8_compile_compat"
_SHIM_REVISION = 1
SUPPORT_SOURCE_UPSTREAM = "upstream"
SUPPORT_SOURCE_TOOLKIT = "toolkit_shim"
_OUTPUT_DTYPES = {
	"bfloat16": torch.bfloat16,
	"float16": torch.float16,
	"float32": torch.float32,
}

try:
	_torch_is_compiling = torch.compiler.is_compiling
except Exception:
	try:
		_torch_is_compiling = torch._dynamo.is_compiling
	except Exception:
		def _torch_is_compiling():
			return False


def _get_registered_op(namespace, name):
	try:
		return getattr(getattr(torch.ops, namespace), name)
	except AttributeError:
		return None


def _has_fake_kernel(operator_name):
	try:
		return bool(torch._C._dispatch_has_kernel_for_dispatch_key(operator_name, "Meta"))
	except Exception:
		return False


def _get_upstream_compile_op():
	operator = _get_registered_op("comfy_kitchen", "w4a8_int8_linear")
	if operator is None or not _has_fake_kernel(_UPSTREAM_OP_NAME):
		return None
	return operator


def _load_upstream_python_linear():
	from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

	return w4a8_int8_linear


def _normalize_output_dtype_name(dtype):
	for name, candidate in _OUTPUT_DTYPES.items():
		if dtype == candidate:
			return name
	raise ValueError(f"W4A8 compile does not support output dtype {dtype}")


def _register_toolkit_op():
	existing_state = getattr(torch, _TORCH_STATE_ATTRIBUTE, None)
	if isinstance(existing_state, dict) and existing_state.get("operator") is not None:
		if existing_state.get("revision") != _SHIM_REVISION:
			return None, "A different Toolkit W4A8 compatibility shim revision is active; restart ComfyUI."
		return existing_state["operator"], existing_state.get("error")

	if _get_registered_op("comfyui_quantization_toolkit", "w4a8_int8_linear") is not None:
		return None, (
			f"{_TOOLKIT_OP_NAME} is already registered without the Toolkit compatibility state; "
			"restart ComfyUI to restore a known operator registration."
		)

	try:
		upstream_python_linear = _load_upstream_python_linear()

		@torch.library.custom_op(_TOOLKIT_OP_NAME, mutates_args=())
		def toolkit_w4a8_int8_linear(
			x: torch.Tensor,
			qweight: torch.Tensor,
			s_rel: torch.Tensor,
			s_channel: torch.Tensor,
			codebook: Optional[torch.Tensor],
			correction: Optional[torch.Tensor],
			bias: Optional[torch.Tensor],
			group_size: int,
			convrot_groupsize: int,
			output_dtype_name: str,
		) -> torch.Tensor:
			return upstream_python_linear(
				x,
				qweight,
				s_rel,
				s_channel,
				codebook=codebook,
				correction=correction,
				bias=bias,
				group_size=group_size,
				convrot_groupsize=convrot_groupsize,
				out_dtype=_OUTPUT_DTYPES[output_dtype_name],
			)

		@toolkit_w4a8_int8_linear.register_fake
		def toolkit_w4a8_int8_linear_fake(
			x,
			qweight,
			s_rel,
			s_channel,
			codebook,
			correction,
			bias,
			group_size,
			convrot_groupsize,
			output_dtype_name,
		):
			return x.new_empty(
				(*x.shape[:-1], qweight.shape[0]),
				dtype=_OUTPUT_DTYPES[output_dtype_name],
			)

		state = {
			"operator": toolkit_w4a8_int8_linear,
			"error": None,
			"revision": _SHIM_REVISION,
		}
		setattr(torch, _TORCH_STATE_ATTRIBUTE, state)
		return toolkit_w4a8_int8_linear, None
	except Exception as exception:
		error = f"{type(exception).__name__}: {exception}"
		logging.warning(f"W4A8 torch.compile compatibility shim is unavailable ({error}).")
		return None, error


if _get_upstream_compile_op() is None:
	_toolkit_op, _toolkit_op_error = _register_toolkit_op()
else:
	_toolkit_op, _toolkit_op_error = None, None


def _detect_compile_support_source():
	if _get_upstream_compile_op() is not None:
		return SUPPORT_SOURCE_UPSTREAM
	if _toolkit_op is not None and _has_fake_kernel(_TOOLKIT_OP_NAME):
		return SUPPORT_SOURCE_TOOLKIT
	return None


_compile_support_source = _detect_compile_support_source()


def get_compile_support_source():
	return _compile_support_source


def get_compile_support_error():
	if get_compile_support_source() is not None:
		return None
	return _toolkit_op_error or "No compiler-safe W4A8 operator with a FakeTensor implementation is available."


def is_compile_supported():
	return get_compile_support_source() is not None


def _is_compiling():
	return bool(_torch_is_compiling())


def call_toolkit_w4a8_int8_linear(
	x,
	qweight,
	s_rel,
	s_channel,
	codebook,
	correction,
	bias,
	group_size,
	convrot_groupsize,
	output_dtype,
):
	if _toolkit_op is None:
		raise RuntimeError(get_compile_support_error())
	return _toolkit_op(
		x,
		qweight,
		s_rel,
		s_channel,
		codebook,
		correction,
		bias,
		int(group_size),
		int(convrot_groupsize),
		_normalize_output_dtype_name(output_dtype),
	)


def native_w4a8_linear(x, weight, bias):
	"""Run W4A8 eagerly, or cross the opaque Toolkit op while Dynamo traces."""
	if get_compile_support_source() == SUPPORT_SOURCE_UPSTREAM or not _is_compiling():
		return F.linear(x, weight, bias)

	params = weight._params
	return call_toolkit_w4a8_int8_linear(
		x,
		weight._qdata,
		params.scale,
		params.s_channel,
		params.codebook,
		params.correction,
		bias,
		params.group_size,
		params.convrot_groupsize,
		params.orig_dtype,
	)
