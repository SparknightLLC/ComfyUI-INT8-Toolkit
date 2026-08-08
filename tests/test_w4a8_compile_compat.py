import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

compat = importlib.import_module(f"{PACKAGE_NAME}.w4a8_compile_compat")
quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")


@unittest.skipUnless(quant._NATIVE_W4A8_AVAILABLE, "Native ComfyUI W4A8 layout is unavailable")
class W4A8CompileCompatTests(unittest.TestCase):
	def tearDown(self):
		try:
			torch._dynamo.reset()
		except Exception:
			pass

	def make_quantized_weight(self, dtype=torch.float32, device="cpu"):
		return quant.quantize_native_w4a8(
			torch.randn(16, 256, dtype=dtype, device=device),
		)

	def make_quantized_module(self):
		module = quant.Int8TensorwiseOps.Linear(
			256,
			16,
			bias=False,
			device="cpu",
			dtype=torch.float32,
		)
		q_weight = self.make_quantized_weight()
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = None
		module._is_quantized = True
		module._quant_format = quant.W4A8_FORMAT
		module._convrot_groupsize = q_weight._params.convrot_groupsize
		module._w4a8_group_size = q_weight._params.group_size
		module.eval()
		return module

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_toolkit_op_matches_native_eager_output(self):
		q_weight = self.make_quantized_weight()
		x = torch.randn(3, 256, dtype=torch.float32)
		bias = torch.randn(16, dtype=torch.float32)
		params = q_weight._params

		expected = F.linear(x, q_weight, bias)
		actual = compat.call_toolkit_w4a8_int8_linear(
			x,
			q_weight._qdata,
			params.scale,
			params.s_channel,
			params.codebook,
			params.correction,
			bias,
			params.group_size,
			params.convrot_groupsize,
			params.orig_dtype,
		)

		torch.testing.assert_close(actual, expected)

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_toolkit_op_supports_asymmetric_correction_metadata(self):
		q_weight = quant.QuantizedTensor.from_float(
			torch.randn(16, 256, dtype=torch.float32),
			"AsymW4A8Int8Layout",
			symmetric=False,
			codebook=False,
		)
		params = q_weight._params
		x = torch.randn(2, 3, 256, dtype=torch.float32)

		self.assertIsNotNone(params.correction)
		expected = F.linear(x, q_weight)
		actual = compat.call_toolkit_w4a8_int8_linear(
			x,
			q_weight._qdata,
			params.scale,
			params.s_channel,
			params.codebook,
			params.correction,
			None,
			params.group_size,
			params.convrot_groupsize,
			params.orig_dtype,
		)

		torch.testing.assert_close(actual, expected)

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_toolkit_op_registration_passes_pytorch_opcheck(self):
		q_weight = quant.QuantizedTensor.from_float(
			torch.randn(16, 256, dtype=torch.float32),
			"AsymW4A8Int8Layout",
			group_size=16,
			convrot_groupsize=256,
			scale_dtype=torch.float32,
		)
		params = q_weight._params
		arguments = (
			torch.randn(3, 256, dtype=torch.float32),
			q_weight._qdata,
			params.scale,
			params.s_channel,
			params.codebook,
			params.correction,
			None,
			params.group_size,
			params.convrot_groupsize,
			"float32",
		)

		results = torch.library.opcheck(
			compat._toolkit_op,
			arguments,
			test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
			raise_exception=True,
		)

		self.assertTrue(all(result == "SUCCESS" for result in results.values()))

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_fake_tensor_output_metadata_matches_linear(self):
		from torch._subclasses.fake_tensor import FakeTensorMode

		with FakeTensorMode():
			x = torch.empty(2, 3, 256, dtype=torch.float16, device="cuda")
			qweight = torch.empty(16, 128, dtype=torch.int8, device="cuda")
			s_rel = torch.empty(16, 16, dtype=torch.float8_e4m3fn, device="cuda")
			s_channel = torch.empty(16, dtype=torch.float32, device="cuda")
			output = compat.call_toolkit_w4a8_int8_linear(
				x,
				qweight,
				s_rel,
				s_channel,
				None,
				None,
				None,
				16,
				256,
				torch.bfloat16,
			)

		self.assertEqual(tuple(output.shape), (2, 3, 16))
		self.assertEqual(output.dtype, torch.bfloat16)
		self.assertEqual(output.device.type, "cuda")

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_quantized_module_compiles_fullgraph_with_dynamic_rows(self):
		module = self.make_quantized_module()
		compiled_module = torch.compile(module, backend="eager", fullgraph=True, dynamic=True)

		for row_count in (3, 5):
			with self.subTest(row_count=row_count):
				x = torch.randn(row_count, 256, dtype=torch.float32)
				expected = module(x)
				actual = compiled_module(x)

				torch.testing.assert_close(actual, expected)

	def test_eager_inference_keeps_native_layout_dispatch(self):
		x = torch.randn(3, 4)
		weight = torch.randn(5, 4)
		expected = object()

		with mock.patch.object(compat, "_compile_support_source", compat.SUPPORT_SOURCE_TOOLKIT):
			with mock.patch.object(compat, "_is_compiling", return_value=False):
				with mock.patch.object(compat.F, "linear", return_value=expected) as linear:
					actual = compat.native_w4a8_linear(x, weight, None)

		self.assertIs(actual, expected)
		linear.assert_called_once_with(x, weight, None)

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit W4A8 compile custom op is unavailable")
	def test_compiling_routes_plain_storage_through_toolkit_op(self):
		x = torch.randn(3, 4)
		params = SimpleNamespace(
			scale=torch.ones(5, 1),
			s_channel=torch.ones(5),
			codebook=None,
			correction=None,
			group_size=16,
			convrot_groupsize=256,
			orig_dtype=torch.bfloat16,
		)
		weight = SimpleNamespace(
			_qdata=torch.ones(5, 2, dtype=torch.int8),
			_params=params,
		)
		expected = object()

		with mock.patch.object(compat, "_compile_support_source", compat.SUPPORT_SOURCE_TOOLKIT):
			with mock.patch.object(compat, "_is_compiling", return_value=True):
				with mock.patch.object(
					compat,
					"call_toolkit_w4a8_int8_linear",
					return_value=expected,
				) as toolkit_linear:
					actual = compat.native_w4a8_linear(x, weight, None)

		self.assertIs(actual, expected)
		toolkit_linear.assert_called_once_with(
			x,
			weight._qdata,
			params.scale,
			params.s_channel,
			None,
			None,
			None,
			16,
			256,
			torch.bfloat16,
		)


if __name__ == "__main__":
	unittest.main()
