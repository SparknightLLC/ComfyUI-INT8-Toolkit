import importlib
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"
PACKAGE_NAME = Path(__file__).resolve().parents[1].name
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from comfy.model_patcher import ModelPatcher

model_adapter = importlib.import_module(f"{PACKAGE_NAME}.int8_model_adapter")


class ModelAdapterCacheTests(unittest.TestCase):
	def setUp(self):
		self.last_model_ref = model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_REF
		self.last_model_id = model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_ID
		self.cache_limit = model_adapter._INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_REF = None
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_ID = None
		model_adapter._INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT = 1

	def tearDown(self):
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_REF = self.last_model_ref
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_ID = self.last_model_id
		model_adapter._INT8_MODEL_ADAPTER_OUTPUT_CACHE_LIMIT = self.cache_limit

	def test_output_caches_are_local_to_each_base_model(self):
		first_model = torch.nn.Module()
		second_model = torch.nn.Module()

		first_cache = model_adapter._get_output_cache(first_model)
		second_cache = model_adapter._get_output_cache(second_model)

		self.assertIsNot(first_cache, second_cache)

	def test_architecture_change_clears_prior_adapter_output(self):
		first_model = torch.nn.Module()
		second_model = torch.nn.Module()
		first_cache = model_adapter._get_output_cache(first_model)
		cached_model_patcher = mock.Mock()
		first_cache[("first",)] = cached_model_patcher
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_REF = weakref.ref(first_model)
		model_adapter._INT8_MODEL_ADAPTER_LAST_MODEL_ID = id(first_model)

		with mock.patch.object(model_adapter, "_cleanup_adapter_cache_memory") as cleanup_memory:
			model_adapter._prepare_model_cache(second_model)

		self.assertEqual(first_cache, {})
		cached_model_patcher.unpatch_model.assert_called_once_with(unpatch_weights=True)
		cleanup_memory.assert_called_once_with()

	def test_cache_room_is_created_before_requantization(self):
		cached_model_patcher = mock.Mock()
		cache = {("first",): cached_model_patcher}

		with mock.patch.object(model_adapter, "_cleanup_adapter_cache_memory") as cleanup_memory:
			model_adapter._make_output_cache_room(cache)

		self.assertEqual(cache, {})
		cached_model_patcher.unpatch_model.assert_called_once_with(unpatch_weights=True)
		cleanup_memory.assert_called_once_with()

	def test_cache_eviction_does_not_copy_quantized_weight_backups_into_source_modules(self):
		shared_model = torch.nn.Module()
		shared_model.diffusion_model = torch.nn.Module()
		shared_model.diffusion_model.layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)
		shared_model.model_lowvram = False
		shared_model.lowvram_patch_counter = 0
		shared_model.model_loaded_weight_memory = 1
		shared_model.model_offload_buffer_memory = 0
		shared_model.current_weight_patches_uuid = "quantized"
		shared_model.device = torch.device("cpu")

		original_module = shared_model.diffusion_model.layer
		base_model_patcher = ModelPatcher(shared_model, torch.device("cpu"), torch.device("cpu"))
		quantized_model_patcher = base_model_patcher.clone()
		quantized_module = torch.nn.Linear(4, 4, bias=False)
		quantized_module.weight = torch.nn.Parameter(
			torch.ones((4, 4), dtype=torch.int8),
			requires_grad=False,
		)
		quantized_model_patcher.add_object_patch("diffusion_model.layer", quantized_module)
		quantized_model_patcher.patch_model(load_weights=False)
		quantized_model_patcher.backup["diffusion_model.layer.weight"] = SimpleNamespace(
			weight=quantized_module.weight.detach().clone(),
			inplace_update=False,
		)

		model_adapter._dispose_cached_adapter_output(quantized_model_patcher)
		replacement_model_patcher = base_model_patcher.clone()
		replacement_model_patcher.unpatch_model(unpatch_weights=True)

		self.assertIs(shared_model.diffusion_model.layer, original_module)
		self.assertEqual(shared_model.diffusion_model.layer.weight.dtype, torch.float16)
		self.assertEqual(replacement_model_patcher.backup, {})

	def test_prior_adapter_reset_fully_unpatches_weights(self):
		quantized_module = model_adapter.Int8TensorwiseOps.Linear(4, 4, bias=False)
		original_module = torch.nn.Linear(4, 4, bias=False)
		shared_model = torch.nn.Module()
		shared_model.diffusion_model = torch.nn.Module()
		shared_model.diffusion_model.layer = quantized_module
		setattr(
			shared_model,
			model_adapter._INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY,
			{"diffusion_model.layer": original_module},
		)
		model_patcher = SimpleNamespace(
			model=shared_model,
			object_patches={"diffusion_model.layer": quantized_module},
			object_patches_backup={},
			get_attachment=lambda _key: None,
			set_attachments=lambda _key, _value: None,
			unpatch_model=mock.Mock(),
		)

		model_adapter._reset_prior_int8_object_patches(model_patcher)

		model_patcher.unpatch_model.assert_called_once_with(unpatch_weights=True)

	def test_prior_adapter_modules_are_restored_even_when_the_live_class_changed(self):
		original_module = torch.nn.Linear(4, 4, bias=False)
		stale_module = torch.nn.Linear(4, 4, bias=False)
		diffusion_model = torch.nn.Module()
		diffusion_model.layer = stale_module
		shared_model = torch.nn.Module()
		shared_model.diffusion_model = diffusion_model
		setattr(
			shared_model,
			model_adapter._INT8_MODEL_ADAPTER_ORIGINAL_MODULES_KEY,
			{"diffusion_model.layer": original_module},
		)

		model_patcher = SimpleNamespace(
			model=shared_model,
			object_patches={},
			object_patches_backup={},
			get_attachment=lambda _key: None,
			set_attachments=lambda _key, _value: None,
			unpatch_model=mock.Mock(),
		)

		restored_count = model_adapter._reset_prior_int8_object_patches(model_patcher)

		self.assertEqual(restored_count, 1)
		self.assertIs(shared_model.diffusion_model.layer, original_module)

	def test_checkpoint_full_precision_mm_source_hint_does_not_block_requantization(self):
		diffusion_model = torch.nn.Module()
		diffusion_model.regular = torch.nn.Linear(256, 16, bias=False)
		diffusion_model.checkpoint_full_precision = torch.nn.Linear(256, 16, bias=False)
		diffusion_model.checkpoint_full_precision._full_precision_mm_config = True

		candidate_names = {
			module_name
			for module_name, _module in model_adapter._collect_int8_candidates(diffusion_model, ())
		}

		self.assertIn("regular", candidate_names)
		self.assertIn("checkpoint_full_precision", candidate_names)

	def test_anima_dtype_guard_does_not_replace_preprocessor(self):
		class DiffusionModel(torch.nn.Module):
			def preprocess_text_embeds(self, text_embeds, text_ids):
				return text_embeds + text_ids.to(text_embeds.dtype)

		diffusion_model = DiffusionModel()
		adapter_state = {"model_type": "anima", "log_progress": False}
		transformer_options = {"int8_model_adapter": adapter_state}
		text_embeds = torch.ones(1, dtype=torch.float32)
		text_ids = torch.ones(1, dtype=torch.long)

		class Executor:
			class_obj = SimpleNamespace(diffusion_model=diffusion_model)

			def __call__(self, *args, **kwargs):
				self.assert_preprocessor_is_unmodified()
				return diffusion_model.preprocess_text_embeds(text_embeds, kwargs["t5xxl_ids"])

			def assert_preprocessor_is_unmodified(self):
				if "preprocess_text_embeds" in diffusion_model.__dict__:
					raise AssertionError("Anima preprocessor was replaced on the module instance")

		result = model_adapter._int8_model_adapter_notice_wrapper(
			Executor(),
			transformer_options=transformer_options,
			t5xxl_ids=text_ids,
		)

		self.assertTrue(torch.equal(result, torch.full((1,), 2.0)))
		self.assertNotIn("preprocess_text_embeds", diffusion_model.__dict__)

	def test_anima_dtype_guard_rejects_float_token_ids_before_forward(self):
		adapter_state = {"model_type": "anima", "log_progress": False}
		transformer_options = {"int8_model_adapter": adapter_state}
		executor = mock.Mock()
		executor.class_obj = SimpleNamespace(diffusion_model=torch.nn.Module())

		with self.assertRaisesRegex(RuntimeError, "Anima requires integer token ids"):
			model_adapter._int8_model_adapter_notice_wrapper(
				executor,
				transformer_options=transformer_options,
				t5xxl_ids=torch.ones(1, dtype=torch.float32),
			)

		executor.assert_not_called()


if __name__ == "__main__":
	unittest.main()
