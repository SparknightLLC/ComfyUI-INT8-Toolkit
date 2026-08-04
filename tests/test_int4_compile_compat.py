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

compat = importlib.import_module(f"{PACKAGE_NAME}.int4_compile_compat")
quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")


@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
class INT4CompileCompatTests(unittest.TestCase):
	def tearDown(self):
		try:
			torch._dynamo.reset()
		except Exception:
			pass

	def make_quantized_weight(self):
		return quant.quantize_native_int4(torch.randn(16, 256, dtype=torch.float32))

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
		module.weight_scale = q_weight._params.scale
		module._is_quantized = True
		module._quant_format = "convrot_w4a4"
		module.eval()
		return module

	def replace_quantized_weight(self, module):
		q_weight = self.make_quantized_weight()
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = q_weight._params.scale

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_toolkit_op_matches_native_eager_output(self):
		q_weight = self.make_quantized_weight()
		x = torch.randn(3, 256, dtype=torch.float32)
		bias = torch.randn(16, dtype=torch.float32)

		expected = F.linear(x, q_weight, bias)
		actual = compat.call_toolkit_convrot_w4a4_linear(
			x,
			q_weight._qdata,
			q_weight._params.scale,
			bias,
			q_weight._params.convrot_groupsize,
			q_weight._params.quant_group_size,
			q_weight._params.linear_dtype,
		)

		torch.testing.assert_close(actual, expected)

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_toolkit_op_registration_passes_pytorch_opcheck(self):
		q_weight = self.make_quantized_weight()
		arguments = (
			torch.randn(3, 256, dtype=torch.float32),
			q_weight._qdata,
			q_weight._params.scale,
			None,
			q_weight._params.convrot_groupsize,
			q_weight._params.quant_group_size,
			q_weight._params.linear_dtype,
		)

		results = torch.library.opcheck(
			compat._toolkit_op,
			arguments,
			test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
			raise_exception=True,
		)

		self.assertTrue(all(result == "SUCCESS" for result in results.values()))

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_fake_tensor_output_metadata_matches_linear(self):
		from torch._subclasses.fake_tensor import FakeTensorMode

		with FakeTensorMode():
			x = torch.empty(2, 3, 256, dtype=torch.bfloat16, device="cuda")
			qweight = torch.empty(16, 128, dtype=torch.int8, device="cuda")
			wscales = torch.empty(16, dtype=torch.float32, device="cuda")
			output = compat.call_toolkit_convrot_w4a4_linear(
				x,
				qweight,
				wscales,
				None,
				256,
				64,
				"int4",
			)

		self.assertEqual(tuple(output.shape), (2, 3, 16))
		self.assertEqual(output.dtype, torch.bfloat16)
		self.assertEqual(output.device.type, "cuda")

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_quantized_module_compiles_fullgraph_with_dynamic_rows(self):
		module = self.make_quantized_module()
		compiled_module = torch.compile(module, backend="eager", fullgraph=True, dynamic=True)

		for row_count in (3, 5):
			with self.subTest(row_count=row_count):
				x = torch.randn(row_count, 256, dtype=torch.float32)
				expected = module(x)
				actual = compiled_module(x)

				torch.testing.assert_close(actual, expected)

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_requantized_weight_reuses_graph_and_updates_output(self):
		module = self.make_quantized_module()
		compile_count = 0

		def compile_backend(graph_module, example_inputs):
			nonlocal compile_count
			compile_count += 1
			return graph_module.forward

		compiled_module = torch.compile(module, backend=compile_backend, fullgraph=True, dynamic=True)
		x = torch.randn(3, 256, dtype=torch.float32)
		first_output = compiled_module(x)

		self.replace_quantized_weight(module)
		expected = module(x)
		actual = compiled_module(x)

		self.assertEqual(compile_count, 1)
		self.assertFalse(torch.equal(actual, first_output))
		torch.testing.assert_close(actual, expected)

	def test_eager_inference_keeps_native_layout_dispatch(self):
		x = torch.randn(3, 4)
		weight = torch.randn(5, 4)
		expected = object()

		with mock.patch.object(compat, "_compile_support_source", compat.SUPPORT_SOURCE_TOOLKIT):
			with mock.patch.object(compat, "_is_compiling", return_value=False):
				with mock.patch.object(compat.F, "linear", return_value=expected) as linear:
					actual = compat.native_int4_linear(x, weight, None)

		self.assertIs(actual, expected)
		linear.assert_called_once_with(x, weight, None)

	def test_upstream_operator_retires_toolkit_runtime_route(self):
		x = torch.randn(3, 4)
		weight = torch.randn(5, 4)
		expected = object()

		with mock.patch.object(compat, "_compile_support_source", compat.SUPPORT_SOURCE_UPSTREAM):
			with mock.patch.object(compat, "_is_compiling", return_value=True):
				with mock.patch.object(compat, "call_toolkit_convrot_w4a4_linear") as toolkit_linear:
					with mock.patch.object(compat.F, "linear", return_value=expected) as linear:
						actual = compat.native_int4_linear(x, weight, None)

		self.assertIs(actual, expected)
		linear.assert_called_once_with(x, weight, None)
		toolkit_linear.assert_not_called()

	@unittest.skipUnless(compat._toolkit_op is not None, "Toolkit INT4 compile custom op is unavailable")
	def test_compiling_routes_plain_storage_through_toolkit_op(self):
		x = torch.randn(3, 4)
		params = SimpleNamespace(
			scale=torch.ones(5),
			convrot_groupsize=256,
			quant_group_size=64,
			linear_dtype="int4",
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
					"call_toolkit_convrot_w4a4_linear",
					return_value=expected,
				) as toolkit_linear:
					actual = compat.native_int4_linear(x, weight, None)

		self.assertIs(actual, expected)
		toolkit_linear.assert_called_once_with(
			x,
			weight._qdata,
			params.scale,
			None,
			256,
			64,
			"int4",
		)


if __name__ == "__main__":
	unittest.main()
