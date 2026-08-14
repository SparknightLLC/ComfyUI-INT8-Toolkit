import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")
loader = importlib.import_module(f"{PACKAGE_NAME}.int8_unet_loader")
model_adapter = importlib.import_module(f"{PACKAGE_NAME}.int8_model_adapter")
quantization_policy = importlib.import_module(f"{PACKAGE_NAME}.quantization_policy")


class QuantizationModeTests(unittest.TestCase):
	def setUp(self):
		self.quantization_state = {
			"dynamic_quantize": quant.Int8TensorwiseOps.dynamic_quantize,
			"_is_prequantized": quant.Int8TensorwiseOps._is_prequantized,
			"quantization_mode": quant.Int8TensorwiseOps.quantization_mode,
			"keep_float_names": list(quant.Int8TensorwiseOps.keep_float_names),
			"int4_sensitive_names": list(quant.Int8TensorwiseOps.int4_sensitive_names),
			"int4_mixed_selected_names": set(quant.Int8TensorwiseOps.int4_mixed_selected_names),
			"int4_mixed_ratio": quant.Int8TensorwiseOps.int4_mixed_ratio,
		}
		quant.Int8TensorwiseOps.reset_otf_progress()

	def tearDown(self):
		quant.Int8TensorwiseOps.reset_otf_progress()
		for name, value in self.quantization_state.items():
			setattr(quant.Int8TensorwiseOps, name, value)

	def test_int_mm_dispatch_preserves_aligned_and_unaligned_results(self):
		for rows, inner, columns in ((17, 8, 8), (3, 7, 5)):
			left = torch.randint(-8, 9, (rows, inner), dtype=torch.int8)
			right = torch.randint(-8, 9, (inner, columns), dtype=torch.int8)

			actual = quant._torch_int_mm_dispatch(left, right, output_columns=columns)
			expected = left.to(torch.int32) @ right.to(torch.int32)

			self.assertTrue(torch.equal(actual, expected))

	def test_int8_convrot_uses_native_runtime_and_preserves_dynamic_lora(self):
		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = torch.nn.Parameter(torch.randint(-8, 9, (16, 256), dtype=torch.int8), requires_grad=False)
		module.weight_scale = torch.ones((16, 1), dtype=torch.float32)
		module._is_quantized = True
		module._quant_format = "int8_tensorwise"
		module._is_per_row = True
		module._outlier_method = quant.OUTLIER_METHOD_CONVROT
		module._convrot_groupsize = 256
		module.quarot_hadamard = quant._build_outlier_hadamard(
			quant.OUTLIER_METHOD_CONVROT,
			256,
			device="cpu",
			dtype=torch.float32,
		)

		x = torch.randn(3, 256, dtype=torch.float32)
		lora_a = torch.randn(4, 256, dtype=torch.float32)
		lora_b = torch.randn(16, 4, dtype=torch.float32)
		rotated_lora_a = quant._transform_weight_like_for_outlier_method(
			lora_a,
			quant.OUTLIER_METHOD_CONVROT,
			torch.device("cpu"),
		)
		module.dynamic_lora_entries = [{"A": rotated_lora_a, "B": lora_b, "offset": None}]
		base_output = torch.randn(3, 16, dtype=torch.bfloat16)
		expected_base = base_output.clone()

		with mock.patch.object(quant, "_native_int8_convrot_linear", return_value=base_output) as native_linear:
			actual = module(x)

		native_linear.assert_called_once()
		expected = expected_base + torch.nn.functional.linear(
			torch.nn.functional.linear(x, lora_a),
			lora_b,
		).to(base_output.dtype)
		torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

	def test_linear_cast_cleanup_runs_when_execution_raises(self):
		for quant_format, kernel_owner, kernel_name in (
			(None, quant.F, "linear"),
			("convrot_w4a4", quant, "native_int4_linear"),
			(quant.W4A8_FORMAT, quant, "native_w4a8_linear"),
		):
			with self.subTest(quant_format=quant_format or "float"):
				module = quant.Int8TensorwiseOps.Linear(4, 3, bias=False, device="cpu", dtype=torch.float32)
				module._is_quantized = quant_format is not None
				module._quant_format = quant_format
				offload_stream = object()

				with mock.patch(
					"comfy.ops.cast_bias_weight",
					return_value=(module.weight, module.bias, offload_stream),
				), mock.patch("comfy.ops.uncast_bias_weight") as uncast_bias_weight, mock.patch.object(
					kernel_owner,
					kernel_name,
					side_effect=RuntimeError("execution failed"),
				):
					with self.assertRaisesRegex(RuntimeError, "execution failed"):
						module(torch.randn(2, 4, dtype=torch.float32))

				uncast_bias_weight.assert_called_once_with(
					module,
					module.weight,
					module.bias,
					offload_stream,
				)

	def test_native_int8_convrot_metadata_round_trip_preserves_layer_group_size(self):
		state_dict = {
			"weight": torch.randint(-8, 9, (16, 2688), dtype=torch.int8),
			"weight_scale": torch.ones((16, 1), dtype=torch.float32),
			"comfy_quant": quant._encode_comfy_quant_config({
				"format": "int8_tensorwise",
				"convrot": True,
				"convrot_groupsize": 64,
			}),
		}
		module = quant.Int8TensorwiseOps.Linear(2688, 16, bias=False, device="cpu", dtype=torch.float32)

		module.load_state_dict(state_dict, strict=False)

		self.assertEqual(module._convrot_groupsize, 64)
		self.assertEqual(tuple(module.quarot_hadamard.shape), (64, 64))
		self.assertEqual(module._outlier_method, quant.OUTLIER_METHOD_CONVROT)
		with mock.patch.object(
			quant,
			"_native_int8_convrot_linear",
			return_value=torch.zeros((2, 16), dtype=torch.bfloat16),
		) as native_linear:
			module(torch.randn(2, 2688, dtype=torch.float32))
		native_linear.assert_called_once()
		self.assertEqual(native_linear.call_args.args[-1], 64)

		saved_state_dict = module.state_dict()
		saved_config = quant._decode_comfy_quant_config(saved_state_dict["comfy_quant"])
		self.assertEqual(saved_config["convrot_groupsize"], 64)
		self.assertNotIn("quarot_hadamard", saved_state_dict)

		reloaded = quant.Int8TensorwiseOps.Linear(2688, 16, bias=False, device="cpu", dtype=torch.float32)
		reloaded.load_state_dict(saved_state_dict, strict=False)
		self.assertEqual(reloaded._convrot_groupsize, 64)
		self.assertEqual(tuple(reloaded.quarot_hadamard.shape), (64, 64))

	def test_native_int8_convrot_load_rejects_invalid_group_metadata(self):
		for group_size, message in (
			(0, "must be positive"),
			(256, "2688 is not divisible"),
			(32, "must be a power of 4"),
			("invalid", "invalid ConvRot group size"),
		):
			with self.subTest(group_size=group_size):
				module = quant.Int8TensorwiseOps.Linear(2688, 16, bias=False, device="cpu", dtype=torch.float32)
				state_dict = {
					"weight": torch.randint(-8, 9, (16, 2688), dtype=torch.int8),
					"weight_scale": torch.ones((16, 1), dtype=torch.float32),
					"comfy_quant": quant._encode_comfy_quant_config({
						"format": "int8_tensorwise",
						"convrot": True,
						"convrot_groupsize": group_size,
					}),
				}

				with self.assertRaisesRegex(ValueError, message):
					module.load_state_dict(state_dict, strict=False)

	def test_native_int8_convrot_save_rejects_inconsistent_hadamard_shape(self):
		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = nn.Parameter(torch.randint(-8, 9, (16, 256), dtype=torch.int8), requires_grad=False)
		module.weight_scale = torch.ones((16, 1), dtype=torch.float32)
		module._is_quantized = True
		module._quant_format = "int8_tensorwise"
		module._outlier_method = quant.OUTLIER_METHOD_CONVROT
		module._convrot_groupsize = 256
		module.quarot_hadamard = quant._build_outlier_hadamard(
			quant.OUTLIER_METHOD_CONVROT,
			64,
			device="cpu",
			dtype=torch.float32,
		)

		with self.assertRaisesRegex(ValueError, "Hadamard shape must be"):
			module.state_dict()

	def test_native_int8_convrot_save_rejects_nondivisible_group_size(self):
		module = quant.Int8TensorwiseOps.Linear(2688, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = nn.Parameter(torch.randint(-8, 9, (16, 2688), dtype=torch.int8), requires_grad=False)
		module.weight_scale = torch.ones((16, 1), dtype=torch.float32)
		module._is_quantized = True
		module._quant_format = "int8_tensorwise"
		module._outlier_method = quant.OUTLIER_METHOD_CONVROT
		module._convrot_groupsize = 256
		module.quarot_hadamard = quant._build_outlier_hadamard(
			quant.OUTLIER_METHOD_CONVROT,
			256,
			device="cpu",
			dtype=torch.float32,
		)

		with self.assertRaisesRegex(ValueError, "2688 is not divisible"):
			module.state_dict()

	def test_dynamic_lora_hook_supports_baked_native_int8_linear(self):
		class DiffusionModel(nn.Module):
			def __init__(self):
				super().__init__()
				self.layer = nn.Linear(4, 3, bias=False)
				self.layer.quant_format = "int8_tensorwise"

		diffusion_model = DiffusionModel()
		input_tensor = torch.randn(2, 4)
		base_output = diffusion_model.layer(input_tensor).detach().clone()
		lora_up = torch.randn(3, 2)
		lora_down = torch.randn(2, 4)
		adapter = quant.LoRAAdapter(
			[],
			(lora_up, lora_down, None, None, None, None),
		)
		dynamic_loras = [{
			"name": "native.safetensors",
			"strength": 0.75,
			"patches": {"diffusion_model.layer.weight": adapter},
			"patch_uuid": "native-test",
		}]

		quant.DynamicLoRAHook().apply_composition(diffusion_model, dynamic_loras)
		actual = diffusion_model.layer(input_tensor)
		expected_delta = torch.nn.functional.linear(
			torch.nn.functional.linear(input_tensor, lora_down * 0.75),
			lora_up,
		)

		self.assertTrue(hasattr(diffusion_model.layer, quant._NATIVE_DYNAMIC_LORA_HOOK_ATTRIBUTE))
		torch.testing.assert_close(actual, base_output + expected_delta)

	def test_native_dynamic_lora_combines_compatible_factors_without_changing_delta(self):
		class DiffusionModel(nn.Module):
			def __init__(self):
				super().__init__()
				self.layer = nn.Linear(4, 3, bias=False)
				self.layer.quant_format = "int8_tensorwise"

		diffusion_model = DiffusionModel()
		input_tensor = torch.randn(2, 4)
		base_output = diffusion_model.layer(input_tensor).detach().clone()
		adapters = []
		expected_delta = torch.zeros_like(base_output)
		for index, (rank, strength, alpha) in enumerate(((2, 0.75, 1.0), (3, -0.4, 6.0))):
			lora_up = torch.randn(3, rank)
			lora_down = torch.randn(rank, 4)
			adapters.append({
				"name": f"native-{index}.safetensors",
				"strength": strength,
				"patches": {
					"diffusion_model.layer.weight": quant.LoRAAdapter(
						[],
						(lora_up, lora_down, alpha, None, None, None),
					),
				},
				"patch_uuid": f"native-combined-{index}",
			})
			expected_delta += torch.nn.functional.linear(
				torch.nn.functional.linear(input_tensor, lora_down * ((alpha / rank) * strength)),
				lora_up,
			)

		quant.DynamicLoRAHook().apply_composition(diffusion_model, adapters)
		actual = diffusion_model.layer(input_tensor)

		self.assertEqual(len(diffusion_model.layer.dynamic_lora_entries), 1)
		torch.testing.assert_close(actual, base_output + expected_delta, rtol=1e-5, atol=1e-5)

	def test_dynamic_lora_hook_matches_linear_inside_compiled_module_wrapper(self):
		class CompiledModule(nn.Module):
			def __init__(self):
				super().__init__()
				self._orig_mod = nn.Module()
				self._orig_mod.layer = nn.Linear(4, 3, bias=False)
				self._orig_mod.layer.quant_format = "int8_tensorwise"

		class DiffusionModel(nn.Module):
			def __init__(self):
				super().__init__()
				self.blocks = nn.ModuleList([CompiledModule()])

		diffusion_model = DiffusionModel()
		lora_up = torch.randn(3, 2)
		lora_down = torch.randn(2, 4)
		adapter = quant.LoRAAdapter(
			[],
			(lora_up, lora_down, None, None, None, None),
		)
		dynamic_loras = [{
			"name": "compiled.safetensors",
			"strength": 1.0,
			"patches": {"diffusion_model.blocks.0.layer.weight": adapter},
			"patch_uuid": "compiled-test",
		}]

		quant.DynamicLoRAHook().apply_composition(diffusion_model, dynamic_loras)

		self.assertEqual(
			len(diffusion_model.blocks[0]._orig_mod.layer.dynamic_lora_entries),
			1,
		)

	def test_dynamic_lora_hook_unwraps_hidden_lazy_compile_source(self):
		class SourceBlock(nn.Module):
			def __init__(self):
				super().__init__()
				self.layer = nn.Linear(4, 3, bias=False)
				self.layer.quant_format = "int8_tensorwise"

		class HiddenCompiledModule(nn.Module):
			def __init__(self, source_module):
				super().__init__()
				object.__setattr__(self, "_source_module", source_module)

		class DiffusionModel(nn.Module):
			def __init__(self, source_block):
				super().__init__()
				self.blocks = nn.ModuleList([HiddenCompiledModule(source_block)])

		source_block = SourceBlock()
		diffusion_model = DiffusionModel(source_block)
		lora_up = torch.randn(3, 2)
		lora_down = torch.randn(2, 4)
		adapter = quant.LoRAAdapter(
			[],
			(lora_up, lora_down, None, None, None, None),
		)
		dynamic_loras = [{
			"name": "hidden-compiled.safetensors",
			"strength": 1.0,
			"patches": {"diffusion_model.blocks.0.layer.weight": adapter},
			"patch_uuid": "hidden-compiled-test",
		}]

		quant.DynamicLoRAHook().apply_composition(diffusion_model, dynamic_loras)

		self.assertEqual(len(source_block.layer.dynamic_lora_entries), 1)

	def test_legacy_modes_are_not_exposed_or_normalized(self):
		for mode in ("none", "convrot", "quarot", "hadanorm"):
			self.assertNotIn(mode, quant.QUANTIZATION_MODE_CHOICES)
			self.assertEqual(quant.normalize_quantization_mode(mode), "int8")

	def test_w4a8_is_exposed_as_a_distinct_low_bit_mode(self):
		self.assertIn(quant.QUANTIZATION_MODE_W4A8, quant.QUANTIZATION_MODE_CHOICES)
		self.assertTrue(quant.quantization_mode_is_int4(quant.QUANTIZATION_MODE_W4A8))
		self.assertEqual(
			quant.quantization_mode_outlier_method(quant.QUANTIZATION_MODE_W4A8),
			quant.OUTLIER_METHOD_NONE,
		)

	def test_model_adapter_rejects_w4a8_without_native_layout(self):
		with mock.patch.object(model_adapter, "native_w4a8_available", return_value=False):
			with self.assertRaisesRegex(RuntimeError, "ComfyUI 0.32.0 or newer"):
				model_adapter.INT8ModelAdapter().apply_quantization(
					object(),
					enable_quantization=model_adapter.QUANTIZATION_CONTROL_ALWAYS,
					model_type=model_adapter.AUTO_MODEL_TYPE,
					quantization_mode=quant.QUANTIZATION_MODE_W4A8,
					log_progress=False,
				)

	@unittest.skipUnless(quant._NATIVE_W4A8_AVAILABLE, "Native ComfyUI W4A8 layout is unavailable")
	def test_native_w4a8_state_dict_round_trip_and_requantization(self):
		q_weight = quant.quantize_native_w4a8(torch.randn(16, 256, dtype=torch.float32))
		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = None
		module._is_quantized = True
		module._quant_format = quant.W4A8_FORMAT
		module._convrot_groupsize = q_weight._params.convrot_groupsize
		module._w4a8_group_size = q_weight._params.group_size

		state_dict = module.state_dict()
		self.assertIn("weight_s_rel", state_dict)
		self.assertIn("weight_s_channel", state_dict)
		self.assertEqual(
			quant._decode_comfy_quant_config(state_dict["comfy_quant"])["format"],
			quant.W4A8_FORMAT,
		)

		loaded = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		loaded.load_state_dict(state_dict, strict=False)
		self.assertEqual(loaded._quant_format, quant.W4A8_FORMAT)
		self.assertEqual(loaded.weight._layout_cls, "AsymW4A8Int8Layout")

		requantized = loaded.set_weight(
			loaded.convert_weight(loaded.weight) + 0.01,
			seed=17,
			return_weight=True,
		)
		self.assertEqual(requantized._layout_cls, "AsymW4A8Int8Layout")

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_native_int4_state_dict_round_trip(self):
		weight = torch.randn(16, 256, dtype=torch.float32)
		q_weight = quant.quantize_native_int4(weight)

		module = quant.Int8TensorwiseOps.Linear(
			256,
			16,
			bias=True,
			device="cpu",
			dtype=torch.float32,
		)
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = q_weight._params.scale
		module._is_quantized = True
		module._quant_format = "convrot_w4a4"
		module._convrot_groupsize = 256
		module._quant_group_size = 64
		module._linear_dtype = "int4"

		state_dict = module.state_dict()
		self.assertEqual(tuple(state_dict["weight"].shape), (16, 128))

		loaded = quant.Int8TensorwiseOps.Linear(
			256,
			16,
			bias=True,
			device="cpu",
			dtype=torch.float32,
		)
		loaded.load_state_dict(state_dict, strict=False)

		self.assertEqual(loaded._quant_format, "convrot_w4a4")
		self.assertEqual(loaded.weight._layout_cls, "TensorCoreConvRotW4A4Layout")
		output = loaded(torch.randn(3, 256, dtype=torch.float32))
		self.assertEqual(tuple(output.shape), (3, 16))
		self.assertTrue(torch.isfinite(output).all())

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_native_int4_int8_compute_metadata_round_trip(self):
		q_weight = quant.quantize_native_int4(
			torch.randn(16, 256, dtype=torch.float32),
			linear_dtype="int8",
		)
		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = q_weight._params.scale
		module._is_quantized = True
		module._quant_format = "convrot_w4a4"

		loaded = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		loaded.load_state_dict(module.state_dict(), strict=False)
		self.assertEqual(loaded._linear_dtype, "int8")
		self.assertEqual(loaded.weight._params.linear_dtype, "int8")

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_native_int4_dynamic_lora_preserves_runtime_delta(self):
		q_weight = quant.quantize_native_int4(torch.randn(16, 256, dtype=torch.float32))
		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
		module.weight_scale = q_weight._params.scale
		module._is_quantized = True
		module._quant_format = "convrot_w4a4"

		x = torch.randn(3, 256, dtype=torch.float32)
		base_output = module(x)
		lora_a = torch.randn(4, 256, dtype=torch.float32)
		lora_b = torch.randn(16, 4, dtype=torch.float32)
		module.dynamic_lora_entries = [{"A": lora_a, "B": lora_b, "offset": None}]

		actual = module(x)
		expected = base_output + torch.nn.functional.linear(torch.nn.functional.linear(x, lora_a), lora_b)
		torch.testing.assert_close(actual, expected)

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_int4_mixed_keeps_sensitive_layer_in_int8(self):
		quant.Int8TensorwiseOps.dynamic_quantize = True
		quant.Int8TensorwiseOps._is_prequantized = False
		quant.Int8TensorwiseOps.quantization_mode = quant.QUANTIZATION_MODE_INT4_MIXED
		quant.Int8TensorwiseOps.keep_float_names = []
		quant.Int8TensorwiseOps.int4_sensitive_names = ["sensitive"]
		quant.Int8TensorwiseOps._init_otf_progress({
			"bulk.weight": torch.randn(16, 256),
			"sensitive.weight": torch.randn(16, 256),
		})

		bulk = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		bulk._load_from_state_dict(
			{"bulk.weight": torch.randn(16, 256)},
			"bulk.",
			{},
			False,
			[],
			[],
			[],
		)

		sensitive = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		sensitive._load_from_state_dict(
			{"sensitive.weight": torch.randn(16, 256)},
			"sensitive.",
			{},
			False,
			[],
			[],
			[],
		)

		self.assertEqual(bulk._quant_format, "convrot_w4a4")
		self.assertEqual(sensitive._quant_format, "int8_tensorwise")

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_int4_full_promotes_sensitive_layer_to_int4(self):
		quant.Int8TensorwiseOps.dynamic_quantize = True
		quant.Int8TensorwiseOps._is_prequantized = False
		quant.Int8TensorwiseOps.quantization_mode = quant.QUANTIZATION_MODE_INT4_FULL
		quant.Int8TensorwiseOps.keep_float_names = []
		quant.Int8TensorwiseOps.int4_sensitive_names = ["sensitive"]

		module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
		module._load_from_state_dict(
			{"sensitive.weight": torch.randn(16, 256)},
			"sensitive.",
			{},
			False,
			[],
			[],
			[],
		)

		self.assertEqual(module._quant_format, "convrot_w4a4")

	@unittest.skipUnless(quant._NATIVE_INT4_AVAILABLE, "Native ComfyUI INT4 layout is unavailable")
	def test_int4_modes_preserve_keep_float_layers(self):
		quant.Int8TensorwiseOps.dynamic_quantize = True
		quant.Int8TensorwiseOps.keep_float_names = ["protected"]
		quant.Int8TensorwiseOps.int4_sensitive_names = []

		for quantization_mode in (
			quant.QUANTIZATION_MODE_INT4_MIXED,
			quant.QUANTIZATION_MODE_INT4_FULL,
		):
			with self.subTest(quantization_mode=quantization_mode):
				quant.Int8TensorwiseOps._is_prequantized = False
				quant.Int8TensorwiseOps.quantization_mode = quantization_mode
				weight = torch.randn(16, 256)
				module = quant.Int8TensorwiseOps.Linear(256, 16, bias=False, device="cpu", dtype=torch.float32)
				module._load_from_state_dict(
					{"protected.weight": weight.clone()},
					"protected.",
					{},
					False,
					[],
					[],
					[],
				)

				self.assertFalse(module._is_quantized)
				self.assertIsNone(module._quant_format)
				torch.testing.assert_close(module.weight, weight)

	def test_architecture_presets_protect_residual_writeback_paths(self):
		krea_preset = loader.get_model_type_quantization_preset(loader.MODEL_TYPE_KREA2)
		anima_preset = loader.get_model_type_quantization_preset("anima")
		minimax_preset = loader.get_model_type_quantization_preset(loader.MODEL_TYPE_MINIMAX_H3)

		self.assertIn("first", krea_preset["keep_float"])
		self.assertEqual(krea_preset["int4_sensitive"], ("attn.wo", "mlp.down"))
		self.assertIn("final_layer", anima_preset["keep_float"])
		self.assertEqual(
			anima_preset["int4_sensitive"],
			("self_attn.output_proj", "cross_attn.output_proj", "mlp.layer2"),
		)
		self.assertEqual(
			minimax_preset["keep_float"],
			(
				"video_patch_proj", "audio_patch_proj", "condition_proj", "time_embedder",
				"token_refiner", "adaln_proj", "final_layer",
			),
		)
		self.assertEqual(minimax_preset["int4_sensitive"], ("attn.out_proj", "mlp.fc2"))

	def test_model_adapter_auto_detects_minimax_h3(self):
		class MiniMaxH3(nn.Module):
			def __init__(self):
				super().__init__()
				self.video_patch_proj = nn.Linear(4, 4)
				self.audio_patch_proj = nn.Linear(4, 4)
				self.condition_proj = nn.Linear(4, 4)
				self.token_refiner = nn.Sequential(nn.Linear(4, 4))
				self.blocks = nn.ModuleList([nn.Module()])
				self.blocks[0].adaln_proj = nn.Sequential(nn.Linear(4, 4))

		self.assertEqual(
			model_adapter._infer_model_type_from_modules(MiniMaxH3()),
			loader.MODEL_TYPE_MINIMAX_H3,
		)

	def test_architecture_ratios_cover_residual_writeback_paths(self):
		krea_names = [
			f"blocks.{block}.{path}"
			for block in range(28)
			for path in (
				"attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo",
				"mlp.gate", "mlp.up", "mlp.down",
			)
		]
		anima_names = [
			f"blocks.{block}.{path}"
			for block in range(28)
			for path in (
				"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
				"cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
				"mlp.layer1", "mlp.layer2",
			)
		]
		krea_preset = loader.get_model_type_quantization_preset(loader.MODEL_TYPE_KREA2)
		anima_preset = loader.get_model_type_quantization_preset("anima")

		krea_selected = quantization_policy.select_int4_mixed_layers(
			krea_names,
			0.25,
			krea_preset["int4_sensitive"],
		)
		anima_selected = quantization_policy.select_int4_mixed_layers(
			anima_names,
			0.3,
			anima_preset["int4_sensitive"],
		)

		self.assertEqual(len(krea_selected), 56)
		self.assertTrue(all(name.endswith(("attn.wo", "mlp.down")) for name in krea_selected))
		self.assertEqual(len(anima_selected), 84)
		self.assertTrue(all(name.endswith(("output_proj", "mlp.layer2")) for name in anima_selected))

	def test_mixed_ratio_selects_a_stable_model_agnostic_budget(self):
		layer_names = [f"blocks.{index}.linear" for index in range(10)]

		first = quantization_policy.select_int4_mixed_layers(
			layer_names,
			0.2,
			priority_patterns=("blocks.9.",),
		)
		second = quantization_policy.select_int4_mixed_layers(
			layer_names,
			0.2,
			priority_patterns=("blocks.9.",),
		)

		self.assertEqual(first, second)
		self.assertEqual(len(first), 2)
		self.assertIn("blocks.9.linear", first)

	def test_mixed_ratio_boundaries_select_none_or_all(self):
		layer_names = [f"layer.{index}" for index in range(5)]

		self.assertEqual(
			quantization_policy.select_int4_mixed_layers(layer_names, 0.0),
			(),
		)
		self.assertEqual(
			quantization_policy.select_int4_mixed_layers(layer_names, 1.0),
			tuple(layer_names),
		)

	def test_mixed_ratio_profiles_are_nested_and_preserve_the_default_selection(self):
		layer_names = [f"layer.{index}" for index in range(100)]
		lower_profile = set(quantization_policy.select_int4_mixed_layers(layer_names, 0.15))
		default_profile = set(quantization_policy.select_int4_mixed_layers(layer_names, 0.2))
		higher_profile = set(quantization_policy.select_int4_mixed_layers(layer_names, 0.3))

		self.assertLess(lower_profile, default_profile)
		self.assertLess(default_profile, higher_profile)
		self.assertEqual(
			set(quantization_policy.select_int4_mixed_layers(layer_names[:10], 0.2)),
			{"layer.2", "layer.7"},
		)

	def test_otf_mixed_profile_selects_the_requested_fraction(self):
		quant.Int8TensorwiseOps.quantization_mode = quant.QUANTIZATION_MODE_INT4_MIXED
		quant.Int8TensorwiseOps.keep_float_names = []
		quant.Int8TensorwiseOps.int4_sensitive_names = ["layer.4"]
		quant.Int8TensorwiseOps.int4_mixed_ratio = 0.4
		state_dict = {
			f"layer.{index}.weight": torch.randn(16, 256)
			for index in range(5)
		}

		quant.Int8TensorwiseOps._init_otf_progress(state_dict)

		self.assertEqual(len(quant.Int8TensorwiseOps.int4_mixed_selected_names), 2)
		self.assertIn("layer.4", quant.Int8TensorwiseOps.int4_mixed_selected_names)

	def test_public_quantization_inputs_do_not_use_legacy_int8_names(self):
		adapter_inputs = model_adapter.INT8ModelAdapter.INPUT_TYPES()["required"]
		loader_inputs = loader.UNetLoaderINTW8A8.INPUT_TYPES()["required"]

		self.assertIn("enable_quantization", adapter_inputs)
		self.assertNotIn("enable_int8", adapter_inputs)
		self.assertIn("prepack_weights", adapter_inputs)
		self.assertNotIn("prepack_int8_weights", adapter_inputs)
		self.assertIn("prepack_weights", loader_inputs)
		self.assertNotIn("prepack_int8_weights", loader_inputs)
		self.assertEqual(
			adapter_inputs["enable_quantization"],
			(
				model_adapter.QUANTIZATION_CONTROL_CHOICES,
				{
					"default": model_adapter.QUANTIZATION_CONTROL_AS_NEEDED,
					"tooltip": "as_needed converts FP8 and floating-point inputs, but leaves MODEL inputs containing Toolkit-supported INT8, W4A4, or W4A8 layers unchanged. always converts remaining eligible layers; bypass returns the MODEL unchanged.",
				},
			),
		)
		self.assertEqual(adapter_inputs["int4_mixed_ratio"][1]["default"], 0.2)
		self.assertEqual(loader_inputs["int4_mixed_ratio"][1]["default"], 0.2)
		self.assertEqual(model_adapter.INT8ModelAdapter.FUNCTION, "apply_quantization")

	def test_as_needed_returns_a_supported_model_unchanged(self):
		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Sequential(nn.Linear(4, 4))
		shared_model.diffusion_model[0]._is_quantized = True
		model = SimpleNamespace(model=shared_model, object_patches={}, object_patches_backup={})

		output = model_adapter.INT8ModelAdapter().apply_quantization(
			model,
			enable_quantization=model_adapter.QUANTIZATION_CONTROL_AS_NEEDED,
			model_type=model_adapter.AUTO_MODEL_TYPE,
			quantization_mode=quant.QUANTIZATION_MODE_INT8,
			log_progress=False,
		)[0]

		self.assertIs(output, model)

	def test_as_needed_detects_supported_object_patches(self):
		quantized_module = nn.Linear(4, 4)
		quantized_module._is_quantized = True
		model = SimpleNamespace(
			model=SimpleNamespace(diffusion_model=nn.Module()),
			object_patches={"diffusion_model.block": quantized_module},
			object_patches_backup={},
		)

		self.assertTrue(model_adapter._model_has_quantized_modules(model))

	def test_as_needed_recognizes_native_int8_but_not_fp8(self):
		native_int8_module = nn.Linear(4, 4)
		native_int8_module.quant_format = "int8_tensorwise"
		native_w4a8_module = nn.Linear(4, 4)
		native_w4a8_module.quant_format = "asym_w4a8_int8"
		native_fp8_module = nn.Linear(4, 4)
		native_fp8_module.quant_format = "float8_e4m3fn"

		self.assertTrue(model_adapter._module_has_target_quantization(native_int8_module))
		self.assertTrue(model_adapter._module_has_target_quantization(native_w4a8_module))
		self.assertFalse(model_adapter._module_has_target_quantization(native_fp8_module))

	def test_bypass_returns_model_without_inspection(self):
		model = object()

		output = model_adapter.INT8ModelAdapter().apply_quantization(
			model,
			enable_quantization=model_adapter.QUANTIZATION_CONTROL_BYPASS,
			model_type=model_adapter.AUTO_MODEL_TYPE,
			quantization_mode=quant.QUANTIZATION_MODE_INT8,
		)[0]

		self.assertIs(output, model)


if __name__ == "__main__":
	unittest.main()
