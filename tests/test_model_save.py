import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

model_save = importlib.import_module(f"{PACKAGE_NAME}.int8_model_save")


class ModelSaveTests(unittest.TestCase):
	def test_default_filename_uses_quantized_model_scope(self):
		filename_input = model_save.INT8ModelSave.INPUT_TYPES()["required"]["filename_prefix"]
		self.assertEqual(filename_input[1]["default"], "quantized_models/Quantized_Model")

	def test_unquantized_model_is_rejected(self):
		model_patcher = SimpleNamespace(
			model=torch.nn.Sequential(torch.nn.Linear(4, 4)),
			object_patches={},
			object_patches_backup={},
		)

		with self.assertRaisesRegex(ValueError, "Place Enable Quantization on MODEL"):
			model_save._validate_quantized_model(model_patcher)

	def test_pending_toolkit_quantized_object_patch_is_accepted(self):
		quantized_module = torch.nn.Linear(4, 4, bias=False)
		quantized_module.weight = torch.nn.Parameter(
			torch.ones((4, 4), dtype=torch.int8),
			requires_grad=False,
		)
		quantized_module._is_quantized = True
		quantized_module._quant_format = "int8_tensorwise"
		model_patcher = SimpleNamespace(
			model=torch.nn.Sequential(torch.nn.Linear(4, 4)),
			object_patches={"diffusion_model.block.linear": quantized_module},
			object_patches_backup={},
		)

		model_save._validate_quantized_model(model_patcher)

	def test_dynamic_lora_is_rejected(self):
		quantized_module = torch.nn.Linear(4, 4, bias=False)
		quantized_module.weight = torch.nn.Parameter(
			torch.ones((4, 4), dtype=torch.int8),
			requires_grad=False,
		)
		quantized_module._is_quantized = True
		quantized_module._quant_format = "int8_tensorwise"
		model_patcher = SimpleNamespace(
			model=torch.nn.Sequential(torch.nn.Linear(4, 4)),
			object_patches={"diffusion_model.block.linear": quantized_module},
			object_patches_backup={},
			get_attachment=lambda key: (("Dynamic", "example.safetensors", 1.0),),
		)

		with self.assertRaisesRegex(ValueError, "cannot serialize Dynamic LoRAs"):
			model_save._validate_quantized_model(model_patcher)


if __name__ == "__main__":
	unittest.main()
