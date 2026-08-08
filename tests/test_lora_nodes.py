import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from comfy.weight_adapter.lora import LoRAAdapter

lora_nodes = importlib.import_module(f"{PACKAGE_NAME}.int8_lora")
lora_dynamic = importlib.import_module(f"{PACKAGE_NAME}.int8_dynamic_lora")
lora_patching = importlib.import_module(f"{PACKAGE_NAME}.int8_lora_patching")
quant = importlib.import_module(f"{PACKAGE_NAME}.int8_quant")


class QuantizedLoraNodeTests(unittest.TestCase):
	def test_display_names_use_standardized_lora_terms(self):
		expected_names = {
			"INT8LoraLoader": "Load LoRA (Quantized)",
			"INT8LoraLoaderStack": "Load LoRA Stack (Quantized)",
			"QuantizedLoraConfig": "LoRA Stack Entry (Quantized)",
			"QuantizedLoraPatcher": "Apply LoRA Stack (Quantized)",
		}

		self.assertEqual(lora_nodes.NODE_DISPLAY_NAME_MAPPINGS, expected_names)

	def test_stack_entry_node_uses_path_terminology(self):
		input_types = lora_nodes.QuantizedLoraConfig.INPUT_TYPES()

		self.assertIn("path", input_types["required"])
		self.assertNotIn("lora_name", input_types["required"])

	def test_stack_entry_node_returns_lora_spec(self):
		output = lora_nodes.QuantizedLoraConfig.execute("folder/test.safetensors", 0.75)

		self.assertEqual(output[0].path, "folder/test.safetensors")
		self.assertEqual(output[0].strength, 0.75)

	def test_apply_stack_node_uses_zero_minimum_autogrow_inputs(self):
		input_types = lora_nodes.QuantizedLoraPatcher.INPUT_TYPES()
		autogrow = input_types["optional"]["loras"][1]

		self.assertEqual(list(input_types["required"]), ["mode", "model"])
		self.assertEqual(autogrow["template"]["min"], 0)
		self.assertEqual(autogrow["template"]["names"][0], "lora_1")
		self.assertEqual(autogrow["template"]["names"][-1], "lora_100")

	def test_apply_stack_node_preserves_input_order_and_skips_zero_strength(self):
		model = object()
		loras = {
			"lora_3": lora_nodes.QuantizedLoraSpec("third.safetensors", -1.0),
			"lora_2": lora_nodes.QuantizedLoraSpec("disabled.safetensors", 0.0),
			"lora_1": lora_nodes.QuantizedLoraSpec("first.safetensors", 0.5),
		}

		with mock.patch.object(
			lora_nodes.INT8LoraLoaderStack,
			"apply_loras",
			return_value=("patched",),
		) as apply_loras:
			output = lora_nodes.QuantizedLoraPatcher.execute(
				model=model,
				mode=lora_nodes.LORA_MODE_DYNAMIC,
				loras=loras,
			)

		self.assertEqual(output, ("patched",))
		apply_loras.assert_called_once_with(
			lora_nodes.LORA_MODE_DYNAMIC,
			model,
			[("first.safetensors", 0.5), ("third.safetensors", -1.0)],
		)

	def test_lora_entry_summary_reports_effective_paths_and_strengths(self):
		entries = [
			("folder/first.safetensors", 0.5),
			("folder/second.safetensors", -1.0),
		]

		self.assertEqual(
			lora_nodes._summarize_lora_entries(entries),
			"folder/first.safetensors@0.5, folder/second.safetensors@-1",
		)

	def test_new_and_legacy_stacks_patch_native_quantized_modules_identically(self):
		class ModelPatcher:
			def __init__(self, quantization_format):
				quantized_module = SimpleNamespace(quant_format=quantization_format)
				self.model = SimpleNamespace(diffusion_model=SimpleNamespace(modules=lambda: ()))
				self.object_patches = {"diffusion_model.block": quantized_module}
				self.object_patches_backup = {}
				self.patch_calls = []

			def clone(self):
				return ModelPatcher(self.object_patches["diffusion_model.block"].quant_format)

			def add_patches(self, patch_dict, strength):
				self.patch_calls.append((patch_dict, strength))

		adapter_a = LoRAAdapter([], (object(), object(), None, None, None, None))
		adapter_b = LoRAAdapter([], (object(), object(), None, None, None, None))
		adapters = {
			"first.safetensors": adapter_a,
			"second.safetensors": adapter_b,
		}

		def load_lora(data, key_map, log_missing=True):
			return {"diffusion_model.block.weight": adapters[data]}

		for quantization_format in ("int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8"):
			with self.subTest(quantization_format=quantization_format):
				model = ModelPatcher(quantization_format)
				loras = {
					"lora_1": lora_nodes.QuantizedLoraSpec("first.safetensors", 0.5),
					"lora_2": lora_nodes.QuantizedLoraSpec("second.safetensors", 0.75),
				}

				with mock.patch.object(lora_nodes, "_get_key_map", return_value={}):
					with mock.patch.object(lora_nodes.folder_paths, "get_full_path", side_effect=lambda _kind, name: name):
						with mock.patch.object(lora_nodes.comfy.utils, "load_torch_file", side_effect=lambda path, safe_load=True: path):
							with mock.patch.object(lora_nodes.comfy.lora, "load_lora", side_effect=load_lora):
								legacy_output = lora_nodes.INT8LoraLoaderStack().apply_stack(
									lora_nodes.LORA_MODE_STOCHASTIC,
									model,
									lora_1="first.safetensors",
									strength_1=0.5,
									lora_2="second.safetensors",
									strength_2=0.75,
								)[0]
								new_output = lora_nodes.QuantizedLoraPatcher.execute(
									model=model,
									mode=lora_nodes.LORA_MODE_STOCHASTIC,
									loras=loras,
								)[0]

				for output in (legacy_output, new_output):
					patch_calls = [call for call in output.patch_calls if call[0]]
					self.assertEqual(
						patch_calls,
						[
							({"diffusion_model.block.weight": adapter_a}, 0.5),
							({"diffusion_model.block.weight": adapter_b}, 0.75),
						],
					)

	def test_stochastic_stack_keeps_mixed_precision_fallback_adapters_native(self):
		class ModelPatcher:
			def __init__(self):
				self.model = SimpleNamespace(diffusion_model=SimpleNamespace(modules=lambda: ()))
				self.object_patches = {
					"diffusion_model.int8_layer": SimpleNamespace(quant_format="int8_tensorwise"),
					"diffusion_model.fp8_layer": SimpleNamespace(quant_format="float8_e4m3fn"),
				}
				self.object_patches_backup = {}
				self.patch_calls = []

			def clone(self):
				return ModelPatcher()

			def add_patches(self, patch_dict, strength):
				self.patch_calls.append((patch_dict, strength))

		adapter_a = LoRAAdapter([], (object(), object(), None, None, None, None))
		adapter_b = LoRAAdapter([], (object(), object(), None, None, None, None))
		adapters = {"a.safetensors": adapter_a, "b.safetensors": adapter_b}

		with mock.patch.object(lora_nodes, "_get_key_map", return_value={}):
			with mock.patch.object(lora_nodes.folder_paths, "get_full_path", side_effect=lambda _kind, name: name):
				with mock.patch.object(lora_nodes.comfy.utils, "load_torch_file", side_effect=lambda path, safe_load=True: path):
					with mock.patch.object(
						lora_nodes.comfy.lora,
						"load_lora",
						side_effect=lambda data, key_map, log_missing=True: {
							"diffusion_model.fp8_layer.weight": adapters[data],
						},
					):
						output = lora_nodes.INT8LoraLoaderStack().apply_loras(
							lora_nodes.LORA_MODE_STOCHASTIC,
							ModelPatcher(),
							[("a.safetensors", -0.5), ("b.safetensors", 0.75)],
						)[0]

		patch_calls = [call for call in output.patch_calls if call[0]]
		self.assertEqual(
			patch_calls,
			[
				({"diffusion_model.fp8_layer.weight": adapter_a}, -0.5),
				({"diffusion_model.fp8_layer.weight": adapter_b}, 0.75),
			],
		)

	def test_int8_adapters_support_comfy_dynamic_vram_prefetch_reconstruction(self):
		def prefetch(adapter):
			return lora_nodes.comfy.lora.prefetch_prepared_value(
				adapter,
				counter=[0],
				destination=None,
				stream=None,
				copy=False,
			)

		first_adapter = LoRAAdapter(
			{"first"},
			(torch.randn(3, 2), torch.randn(2, 4), 1.0, None, None, None),
		)
		second_adapter = LoRAAdapter(
			{"second"},
			(torch.randn(3, 1), torch.randn(1, 4), None, None, None, None),
		)
		weight_scale = torch.full((3, 1), 0.25)

		adapters = [
			quant.INT8LoRAPatchAdapter(
				first_adapter.loaded_keys,
				first_adapter.weights,
				weight_scale,
				seed=17,
				outlier_method=quant.OUTLIER_METHOD_CONVROT,
			),
			quant.INT8MergedLoRAPatchAdapter(
				[(first_adapter, -0.5), (second_adapter, 0.75)],
				weight_scale,
				seed=23,
			),
			quant.INT8WeightPatchAdapter(
				first_adapter,
				weight_scale,
				seed=31,
			),
		]

		for adapter in adapters:
			with self.subTest(adapter=type(adapter).__name__):
				reconstructed = prefetch(adapter)
				self.assertIsInstance(reconstructed, type(adapter))
				self.assertEqual(reconstructed.seed, adapter.seed)
				torch.testing.assert_close(reconstructed.weight_scale, adapter.weight_scale)

		base_weight = torch.randn(3, 4)
		identity = lambda value: value
		for adapter in adapters:
			reconstructed = prefetch(adapter)
			expected = adapter.calculate_weight(
				base_weight.clone(),
				"diffusion_model.layer.weight",
				-0.6,
				1.0,
				None,
				identity,
			)
			actual = reconstructed.calculate_weight(
				base_weight.clone(),
				"diffusion_model.layer.weight",
				-0.6,
				1.0,
				None,
				identity,
			)
			torch.testing.assert_close(actual, expected)

	def test_plain_stochastic_stack_uses_stable_source_order(self):
		adapter_z = LoRAAdapter([], (object(), object(), None, None, None, None))
		adapter_a = LoRAAdapter([], (object(), object(), None, None, None, None))
		lora_patching._set_lora_source_key(adapter_z, "z.safetensors")
		lora_patching._set_lora_source_key(adapter_a, "a.safetensors")

		merged_adapter = lora_patching._create_stochastic_stack_adapter(
			[(adapter_z, 0.5), (adapter_a, 0.75)],
			weight_scale=1.0,
			seed=318008,
		)

		self.assertEqual(
			merged_adapter.patches,
			[(adapter_a, 0.75), (adapter_z, 0.5)],
		)

	def test_dynamic_mode_falls_back_to_standard_patches_for_mixed_precision_layers(self):
		int8_module = SimpleNamespace(quant_format="int8_tensorwise")
		w4a8_module = SimpleNamespace(quant_format="asym_w4a8_int8")
		fp8_module = SimpleNamespace(quant_format="float8_e4m3fn")
		model_patcher = SimpleNamespace(
			object_patches={
				"diffusion_model.int8_layer": int8_module,
				"diffusion_model.w4a8_layer": w4a8_module,
				"diffusion_model.fp8_layer": fp8_module,
			},
			object_patches_backup={},
		)
		int8_adapter = LoRAAdapter([], (object(), object(), None, None, None, None))
		fp8_adapter = LoRAAdapter([], (object(), object(), None, None, None, None))
		patch_dict = {
			"diffusion_model.int8_layer.weight": int8_adapter,
			"diffusion_model.w4a8_layer.weight": int8_adapter,
			"diffusion_model.fp8_layer.weight": fp8_adapter,
		}

		dynamic_patches, static_patches = lora_dynamic._partition_dynamic_patches(
			model_patcher,
			patch_dict,
			module_cache={},
		)

		self.assertEqual(dynamic_patches, {"diffusion_model.int8_layer.weight": int8_adapter})
		self.assertEqual(
			static_patches,
			{
				"diffusion_model.w4a8_layer.weight": int8_adapter,
				"diffusion_model.fp8_layer.weight": fp8_adapter,
			},
		)

	def test_dynamic_mode_warns_when_w4a8_uses_standard_fallback(self):
		model_patcher = SimpleNamespace(
			object_patches={
				"diffusion_model.block": SimpleNamespace(quant_format="asym_w4a8_int8"),
			},
			object_patches_backup={},
			model=SimpleNamespace(diffusion_model=SimpleNamespace(modules=lambda: ())),
		)

		with self.assertLogs(level="WARNING") as captured:
			lora_dynamic._warn_if_w4a8_dynamic_fallback(model_patcher)

		self.assertIn("W4A8 runtime patching is not supported", captured.output[0])
		self.assertIn("Standard LoRA patch path", captured.output[0])


if __name__ == "__main__":
	unittest.main()
