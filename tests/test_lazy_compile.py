import functools
import importlib
import gc
import sys
import unittest
import weakref
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

import comfy.patcher_extension
import comfy.ops

lazy_compile = importlib.import_module(f"{PACKAGE_NAME}.int8_lazy_compile")


class LazyCompileTests(unittest.TestCase):
	def setUp(self):
		self.last_model_ref = lazy_compile._LAZY_COMPILE_LAST_MODEL_REF
		self.last_model_id = lazy_compile._LAZY_COMPILE_LAST_MODEL_ID
		self.cache_limit = lazy_compile._LAZY_COMPILE_OUTPUT_CACHE_LIMIT
		self.reset_on_model_change = lazy_compile._LAZY_COMPILE_RESET_ON_MODEL_CHANGE
		lazy_compile._LAZY_COMPILE_LAST_MODEL_REF = None
		lazy_compile._LAZY_COMPILE_LAST_MODEL_ID = None
		lazy_compile._LAZY_COMPILE_OUTPUT_CACHE_LIMIT = 1
		lazy_compile._LAZY_COMPILE_RESET_ON_MODEL_CHANGE = True

	def tearDown(self):
		lazy_compile._LAZY_COMPILE_LAST_MODEL_REF = self.last_model_ref
		lazy_compile._LAZY_COMPILE_LAST_MODEL_ID = self.last_model_id
		lazy_compile._LAZY_COMPILE_OUTPUT_CACHE_LIMIT = self.cache_limit
		lazy_compile._LAZY_COMPILE_RESET_ON_MODEL_CHANGE = self.reset_on_model_change

	def test_native_int4_object_patch_is_detected(self):
		int4_module = torch.nn.Linear(4, 4)
		int4_module._quant_format = "convrot_w4a4"
		model_patcher = SimpleNamespace(
			object_patches={"diffusion_model.block.linear": int4_module},
			object_patches_backup={},
			model=SimpleNamespace(diffusion_model=torch.nn.Module()),
		)

		self.assertTrue(lazy_compile._has_native_int4_modules(model_patcher))

	def test_w4a8_object_patch_is_detected(self):
		w4a8_module = torch.nn.Linear(4, 4)
		w4a8_module._quant_format = "asym_w4a8_int8"
		model_patcher = SimpleNamespace(
			object_patches={"diffusion_model.block.linear": w4a8_module},
			object_patches_backup={},
			model=SimpleNamespace(diffusion_model=torch.nn.Module()),
		)

		self.assertTrue(lazy_compile._has_w4a8_modules(model_patcher))

	def test_w4a8_warning_explains_upstream_compile_limitation(self):
		with mock.patch.object(lazy_compile.importlib.metadata, "version", return_value="0.2.28"):
			with mock.patch.object(
				lazy_compile.w4a8_compile_compat,
				"get_compile_support_error",
				return_value="test registration failure",
			):
				message = lazy_compile._build_w4a8_compile_warning()

		self.assertIn("Installed comfy-kitchen version: 0.2.28", message)
		self.assertIn("AsymW4A8Int8Layout", message)
		self.assertIn("torch.library custom operator", message)
		self.assertIn("FakeTensor implementation", message)
		self.assertIn("test registration failure", message)
		self.assertIn("returned uncompiled", message)

	def test_plain_model_is_not_detected_as_int4(self):
		model_patcher = SimpleNamespace(
			object_patches={},
			object_patches_backup={},
			model=SimpleNamespace(diffusion_model=torch.nn.Sequential(torch.nn.Linear(4, 4))),
		)

		self.assertFalse(lazy_compile._has_native_int4_modules(model_patcher))

	def test_compile_logging_input_is_named_verbose(self):
		required_inputs = lazy_compile.INT8LazyTorchCompile.INPUT_TYPES()["required"]

		self.assertIn("verbose", required_inputs)
		self.assertNotIn("log_compile", required_inputs)
		self.assertIn("dynamic_shape_tracing", required_inputs)
		self.assertNotIn("dynamic", required_inputs)

	def test_compile_preparation_log_clears_active_progress_bar(self):
		with mock.patch.object(lazy_compile.tqdm, "external_write_mode") as external_write_mode:
			with mock.patch.object(lazy_compile.logging, "info") as info:
				lazy_compile._log_info_outside_progress_bar("prepared")

		external_write_mode.assert_called_once_with(file=sys.stderr)
		info.assert_called_once_with("prepared")

	def test_static_clone_normalizes_mixed_dtype_unquantized_modules_once(self):
		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		shared_model.diffusion_model.first = comfy.ops.disable_weight_init.Linear(4, 4, dtype=torch.float32)
		shared_model.diffusion_model.first.weight = nn.Parameter(
			torch.randn(4, 4, dtype=torch.float32),
			requires_grad=False,
		)
		shared_model.diffusion_model.first.bias = nn.Parameter(
			torch.randn(4, dtype=torch.float32),
			requires_grad=False,
		)
		shared_model.diffusion_model.matching = comfy.ops.disable_weight_init.Linear(4, 4, dtype=torch.float16)
		shared_model.get_dtype_inference = lambda: torch.float16
		model = SimpleNamespace(
			model=shared_model,
			object_patches={},
			object_patches_backup={},
			get_model_object=lambda name: shared_model.diffusion_model,
		)

		input_tensor = torch.randn(2, 4, dtype=torch.float16)
		with self.assertRaisesRegex(RuntimeError, "mat1 and mat2 must have the same dtype"):
			shared_model.diffusion_model.first(input_tensor)

		normalized_count = lazy_compile._normalize_static_clone_dtypes(model)
		output = shared_model.diffusion_model.first(input_tensor)

		self.assertEqual(normalized_count, 1)
		self.assertEqual(shared_model.diffusion_model.first.weight.dtype, torch.float16)
		self.assertEqual(shared_model.diffusion_model.first.bias.dtype, torch.float16)
		self.assertFalse(shared_model.diffusion_model.first.comfy_cast_weights)
		self.assertFalse(shared_model.diffusion_model.matching.comfy_cast_weights)
		self.assertEqual(output.dtype, torch.float16)

	def test_static_clone_does_not_modify_replaced_or_quantized_modules(self):
		class CastableLinear(nn.Linear):
			def __init__(self):
				super().__init__(4, 4, dtype=torch.float32)
				self.comfy_cast_weights = False

		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		shared_model.diffusion_model.replaced = CastableLinear()
		shared_model.diffusion_model.quantized = CastableLinear()
		shared_model.diffusion_model.quantized._is_quantized = True
		shared_model.get_dtype_inference = lambda: torch.float16
		model = SimpleNamespace(
			model=shared_model,
			object_patches={"diffusion_model.replaced": nn.Identity()},
			object_patches_backup={},
			get_model_object=lambda name: shared_model.diffusion_model,
		)

		normalized_count = lazy_compile._normalize_static_clone_dtypes(model)

		self.assertEqual(normalized_count, 0)
		self.assertFalse(shared_model.diffusion_model.replaced.comfy_cast_weights)
		self.assertFalse(shared_model.diffusion_model.quantized.comfy_cast_weights)

	def test_dtype_repair_is_limited_to_dynamic_vram_demotion(self):
		static_clone = SimpleNamespace()
		model = SimpleNamespace(
			is_dynamic=lambda: False,
			clone=lambda disable_dynamic=False: static_clone,
		)

		with mock.patch.object(lazy_compile, "_normalize_static_clone_dtypes") as normalize_dtypes:
			output = lazy_compile._clone_for_lazy_compile(model, True, True)

		self.assertIs(output, static_clone)
		normalize_dtypes.assert_not_called()

	def test_native_int4_warning_explains_upstream_compile_limitation(self):
		with mock.patch.object(lazy_compile.importlib.metadata, "version", return_value="0.2.22"):
			with mock.patch.object(
				lazy_compile.int4_compile_compat,
				"get_compile_support_error",
				return_value="test registration failure",
			):
				message = lazy_compile._build_native_int4_compile_warning()

		self.assertIn("Installed comfy-kitchen version: 0.2.22", message)
		self.assertIn("TensorCoreConvRotW4A4Layout", message)
		self.assertIn("torch.library custom operator", message)
		self.assertIn("FakeTensor implementation", message)
		self.assertIn("test registration failure", message)
		self.assertIn("returned uncompiled", message)
		self.assertIn(lazy_compile._COMFY_KITCHEN_PROJECT_URL, message)

	def test_native_int4_warning_is_not_gated_by_verbose(self):
		class ModelPatcher:
			def __init__(self, shared_model, int4_module):
				self.model = shared_model
				self.object_patches = {"diffusion_model.block": int4_module}
				self.object_patches_backup = {}
				self.model_options = {}

			def clone(self, disable_dynamic=False):
				return ModelPatcher(self.model, self.object_patches["diffusion_model.block"])

			def remove_wrappers_with_key(self, wrapper_type, key):
				return

		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		int4_module = nn.Linear(4, 4)
		int4_module._quant_format = "convrot_w4a4"
		model = ModelPatcher(shared_model, int4_module)

		with mock.patch.object(lazy_compile, "_prepare_model_cache", return_value={}):
			with mock.patch.object(lazy_compile.int4_compile_compat, "is_compile_supported", return_value=False):
				with mock.patch.object(
					lazy_compile.int4_compile_compat,
					"get_compile_support_error",
					return_value="test registration failure",
				):
					with mock.patch.object(lazy_compile.logging, "warning") as warning:
						output = lazy_compile.INT8LazyTorchCompile().apply_lazy_compile(
							model,
							backend="inductor",
							fullgraph=False,
							mode="default",
							dynamic_shape_tracing="true",
							compile_transformer_blocks_only=True,
							dynamo_cache_size_limit=640,
							use_guard_filter=True,
							disable_dynamic_vram=True,
							verbose=False,
						)[0]

		self.assertIsNot(output, model)
		warning.assert_called_once()
		self.assertIn("Native ConvRot INT4 detected", warning.call_args.args[0])

	def test_native_int4_uses_lazy_compile_when_compat_is_available(self):
		class ModelPatcher:
			def __init__(self, shared_model, int4_module):
				self.model = shared_model
				self.patches_uuid = "int4"
				self.object_patches = {"diffusion_model.blocks.0.linear": int4_module}
				self.object_patches_backup = {}
				self.model_options = {"transformer_options": {}}
				self.wrappers = {}

			def clone(self, disable_dynamic=False):
				clone = ModelPatcher(self.model, self.object_patches["diffusion_model.blocks.0.linear"])
				clone.model_options = self.model_options.copy()
				return clone

			def get_model_object(self, name):
				if name != "diffusion_model":
					raise KeyError(name)
				return self.model.diffusion_model

			def remove_wrappers_with_key(self, wrapper_type, key):
				self.wrappers.pop((wrapper_type, key), None)

			def add_wrapper_with_key(self, wrapper_type, key, wrapper):
				self.wrappers[(wrapper_type, key)] = wrapper

		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		shared_model.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		int4_module = nn.Linear(4, 4)
		int4_module._quant_format = "convrot_w4a4"
		model = ModelPatcher(shared_model, int4_module)
		compile_wrapper = object()

		with mock.patch.object(lazy_compile.int4_compile_compat, "is_compile_supported", return_value=True):
			with mock.patch.object(lazy_compile, "_make_lazy_compile_wrapper", return_value=compile_wrapper):
				with mock.patch.object(lazy_compile, "_cleanup_compile_memory"):
					with mock.patch.object(lazy_compile.logging, "warning") as warning:
						output = lazy_compile.INT8LazyTorchCompile().apply_lazy_compile(
							model,
							backend="inductor",
							fullgraph=False,
							mode="default",
							dynamic_shape_tracing="true",
							compile_transformer_blocks_only=True,
							dynamo_cache_size_limit=640,
							use_guard_filter=True,
							disable_dynamic_vram=True,
							verbose=False,
						)[0]

		wrapper_key = (comfy.patcher_extension.WrappersMP.APPLY_MODEL, lazy_compile._LAZY_COMPILE_WRAPPER_KEY)
		self.assertIs(output.wrappers[wrapper_key], compile_wrapper)
		warning.assert_not_called()

	def test_w4a8_uses_lazy_compile_when_compat_is_available(self):
		class ModelPatcher:
			def __init__(self, shared_model, w4a8_module):
				self.model = shared_model
				self.patches_uuid = "w4a8"
				self.object_patches = {"diffusion_model.blocks.0.linear": w4a8_module}
				self.object_patches_backup = {}
				self.model_options = {"transformer_options": {}}
				self.wrappers = {}

			def clone(self, disable_dynamic=False):
				clone = ModelPatcher(self.model, self.object_patches["diffusion_model.blocks.0.linear"])
				clone.model_options = self.model_options.copy()
				return clone

			def get_model_object(self, name):
				if name != "diffusion_model":
					raise KeyError(name)
				return self.model.diffusion_model

			def remove_wrappers_with_key(self, wrapper_type, key):
				self.wrappers.pop((wrapper_type, key), None)

			def add_wrapper_with_key(self, wrapper_type, key, wrapper):
				self.wrappers[(wrapper_type, key)] = wrapper

		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		shared_model.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		w4a8_module = nn.Linear(4, 4)
		w4a8_module._quant_format = "asym_w4a8_int8"
		model = ModelPatcher(shared_model, w4a8_module)
		compile_wrapper = object()

		with mock.patch.object(lazy_compile.w4a8_compile_compat, "is_compile_supported", return_value=True):
			with mock.patch.object(lazy_compile, "_make_lazy_compile_wrapper", return_value=compile_wrapper):
				with mock.patch.object(lazy_compile, "_cleanup_compile_memory"):
					with mock.patch.object(lazy_compile.logging, "warning") as warning:
						output = lazy_compile.INT8LazyTorchCompile().apply_lazy_compile(
							model,
							backend="inductor",
							fullgraph=False,
							mode="default",
							dynamic_shape_tracing="true",
							compile_transformer_blocks_only=True,
							dynamo_cache_size_limit=640,
							use_guard_filter=True,
							disable_dynamic_vram=True,
							verbose=False,
						)[0]

		wrapper_key = (comfy.patcher_extension.WrappersMP.APPLY_MODEL, lazy_compile._LAZY_COMPILE_WRAPPER_KEY)
		self.assertIs(output.wrappers[wrapper_key], compile_wrapper)
		warning.assert_not_called()

	def test_output_caches_are_local_to_each_base_model(self):
		first_model = torch.nn.Module()
		second_model = torch.nn.Module()

		first_cache = lazy_compile._get_output_cache(first_model)
		second_cache = lazy_compile._get_output_cache(second_model)

		self.assertIsNot(first_cache, second_cache)

	def test_output_cache_does_not_retain_base_model(self):
		class CachedOutput:
			pass

		shared_model = torch.nn.Module()
		cached_output = CachedOutput()
		cached_output.model = shared_model
		lazy_compile._remember_cached_output(shared_model, ("output",), cached_output)
		shared_model_ref = weakref.ref(shared_model)
		cached_output_ref = weakref.ref(cached_output)

		del cached_output
		del shared_model
		gc.collect()

		self.assertIsNone(cached_output_ref())
		self.assertIsNone(shared_model_ref())

	def test_structure_caches_are_local_to_each_base_model(self):
		first_model = torch.nn.Module()
		second_model = torch.nn.Module()

		first_cache = lazy_compile._get_structure_cache(first_model)
		second_cache = lazy_compile._get_structure_cache(second_model)

		self.assertIsNot(first_cache, second_cache)

	def test_structure_wrapper_survives_output_eviction(self):
		shared_model = torch.nn.Module()
		structure_key = ("structure",)
		wrapper = object()
		cached_output = SimpleNamespace()
		output_cache = lazy_compile._get_output_cache(shared_model)
		output_cache[("lora_a",)] = cached_output
		lazy_compile._remember_structure_wrapper(shared_model, structure_key, wrapper)

		with mock.patch.object(lazy_compile, "_dispose_cached_output"):
			with mock.patch.object(lazy_compile, "_cleanup_compile_memory"):
				lazy_compile._make_output_cache_room(output_cache)

		self.assertIs(lazy_compile._get_structure_cache(shared_model)[structure_key], wrapper)

	def test_changed_patch_uuid_reuses_structural_wrapper(self):
		class ModelPatcher:
			def __init__(self, shared_model, patches_uuid):
				self.model = shared_model
				self.patches_uuid = patches_uuid
				self.object_patches = {}
				self.object_patches_backup = {}
				self.model_options = {"transformer_options": {}}
				self.wrappers = {}

			def clone(self, disable_dynamic=False):
				clone = ModelPatcher(self.model, self.patches_uuid)
				clone.model_options = self.model_options.copy()
				return clone

			def get_model_object(self, name):
				if name != "diffusion_model":
					raise KeyError(name)
				return self.model.diffusion_model

			def remove_wrappers_with_key(self, wrapper_type, key):
				self.wrappers.pop((wrapper_type, key), None)

			def add_wrapper_with_key(self, wrapper_type, key, wrapper):
				self.wrappers[(wrapper_type, key)] = wrapper

		shared_model = nn.Module()
		shared_model.diffusion_model = nn.Module()
		shared_model.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		first_input = ModelPatcher(shared_model, "lora_a")
		second_input = ModelPatcher(shared_model, "lora_b")
		structural_wrapper = object()
		apply_kwargs = {
			"backend": "inductor",
			"fullgraph": False,
			"mode": "default",
			"dynamic_shape_tracing": "true",
			"compile_transformer_blocks_only": True,
			"dynamo_cache_size_limit": 640,
			"use_guard_filter": True,
			"disable_dynamic_vram": True,
			"verbose": True,
		}

		with mock.patch.object(lazy_compile, "_make_lazy_compile_wrapper", return_value=structural_wrapper) as make_wrapper:
			with mock.patch.object(lazy_compile, "_cleanup_compile_memory"):
				first_output = lazy_compile.INT8LazyTorchCompile().apply_lazy_compile(first_input, **apply_kwargs)[0]
				second_output = lazy_compile.INT8LazyTorchCompile().apply_lazy_compile(second_input, **apply_kwargs)[0]

		wrapper_key = (comfy.patcher_extension.WrappersMP.APPLY_MODEL, lazy_compile._LAZY_COMPILE_WRAPPER_KEY)
		self.assertIs(second_output.wrappers[wrapper_key], structural_wrapper)
		self.assertNotIn(wrapper_key, first_output.wrappers)
		make_wrapper.assert_called_once()

	def test_same_model_eviction_preserves_dynamo_cache(self):
		shared_model = torch.nn.Module()
		first_output = mock.Mock()
		second_output = mock.Mock()

		with mock.patch.object(lazy_compile, "_dispose_cached_output") as dispose_output:
			with mock.patch.object(lazy_compile, "_cleanup_compile_memory") as cleanup_memory:
				lazy_compile._remember_cached_output(shared_model, ("first",), first_output)
				lazy_compile._remember_cached_output(shared_model, ("second",), second_output)

		dispose_output.assert_called_once_with(first_output)
		cleanup_memory.assert_called_with(reset_compile_cache=False)

	def test_cache_room_is_created_before_replacement_compile(self):
		cached_output = SimpleNamespace()
		cache = {("first",): cached_output}

		with mock.patch.object(lazy_compile, "_dispose_cached_output") as dispose_output:
			with mock.patch.object(lazy_compile, "_cleanup_compile_memory") as cleanup_memory:
				lazy_compile._make_output_cache_room(cache)

		self.assertEqual(cache, {})
		dispose_output.assert_called_once_with(cached_output)
		cleanup_memory.assert_called_once_with(reset_compile_cache=False)

	def test_architecture_change_clears_prior_cache_and_resets_dynamo(self):
		first_model = torch.nn.Module()
		second_model = torch.nn.Module()
		cached_output = SimpleNamespace()
		first_cache = lazy_compile._get_output_cache(first_model)
		first_cache[("first",)] = cached_output
		first_structure_cache = lazy_compile._get_structure_cache(first_model)
		first_structure_cache[("structure",)] = object()
		lazy_compile._LAZY_COMPILE_LAST_MODEL_REF = weakref.ref(first_model)
		lazy_compile._LAZY_COMPILE_LAST_MODEL_ID = id(first_model)

		with mock.patch.object(lazy_compile, "_dispose_cached_output") as dispose_output:
			with mock.patch.object(lazy_compile, "_cleanup_compile_memory") as cleanup_memory:
				lazy_compile._prepare_model_cache(second_model)

		self.assertEqual(first_cache, {})
		self.assertEqual(first_structure_cache, {})
		dispose_output.assert_called_once_with(cached_output)
		cleanup_memory.assert_called_once_with(reset_compile_cache=True)

	def test_compiled_wrapper_reuses_graph_with_replaced_leaf_weights(self):
		class LinearLeaf(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.weight = nn.Parameter(torch.full((2, 2), value), requires_grad=False)

			def forward(self, value):
				return value @ self.weight.T

		class Block(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.linear = LinearLeaf(value)

			def forward(self, value):
				return self.linear(value)

		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return self.class_obj.diffusion_model.blocks[0](value)

		compile_count = 0

		def counting_backend(graph_module, _example_inputs):
			nonlocal compile_count
			compile_count += 1
			return graph_module.forward

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([Block(1.0)])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0"],
			{"backend": counting_backend, "fullgraph": True, "dynamic": False},
			False,
		)
		value = torch.ones((1, 2))

		first_output = wrapper(executor, value)
		root.diffusion_model.blocks[0].linear = LinearLeaf(2.0)
		second_output = wrapper(executor, value)
		root.diffusion_model.blocks[0] = Block(3.0)
		third_output = wrapper(executor, value)

		self.assertTrue(torch.equal(first_output, torch.full((1, 2), 2.0)))
		self.assertTrue(torch.equal(second_output, torch.full((1, 2), 4.0)))
		self.assertTrue(torch.equal(third_output, torch.full((1, 2), 6.0)))
		self.assertEqual(compile_count, 1)

	def test_transformer_blocks_share_one_compiled_dispatcher(self):
		class LinearLeaf(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.weight = nn.Parameter(torch.full((2, 2), value), requires_grad=False)

			def forward(self, value):
				return value @ self.weight.T

		class Block(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.linear = LinearLeaf(value)

			def forward(self, value):
				return self.linear(value)

		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return tuple(block(value) for block in self.class_obj.diffusion_model.blocks)

		compile_count = 0

		def counting_backend(graph_module, _example_inputs):
			nonlocal compile_count
			compile_count += 1
			return graph_module.forward

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([Block(1.0), Block(2.0)])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0", "diffusion_model.blocks.1"],
			{"backend": counting_backend, "fullgraph": True, "dynamic": False},
			False,
		)
		value = torch.ones((1, 2))

		first_output = wrapper(executor, value)
		root.diffusion_model.blocks[0].linear = LinearLeaf(3.0)
		root.diffusion_model.blocks[1].linear = LinearLeaf(4.0)
		second_output = wrapper(executor, value)

		self.assertTrue(torch.equal(first_output[0], torch.full((1, 2), 2.0)))
		self.assertTrue(torch.equal(first_output[1], torch.full((1, 2), 4.0)))
		self.assertTrue(torch.equal(second_output[0], torch.full((1, 2), 6.0)))
		self.assertTrue(torch.equal(second_output[1], torch.full((1, 2), 8.0)))
		self.assertEqual(compile_count, 1)

	def test_self_bound_partial_forwards_share_one_compiled_dispatcher(self):
		def patched_attention_forward(attention, value):
			return value @ attention.weight.T

		class Attention(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.weight = nn.Parameter(torch.full((2, 2), value), requires_grad=False)

			def forward(self, value):
				raise AssertionError("The instance partial should replace this forward")

		class Block(nn.Module):
			def __init__(self, value):
				super().__init__()
				self.attention = Attention(value)
				self.attention.forward = functools.partial(patched_attention_forward, self.attention)

			def forward(self, value):
				return self.attention(value)

		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return tuple(block(value) for block in self.class_obj.diffusion_model.blocks)

		compile_count = 0

		def counting_backend(graph_module, _example_inputs):
			nonlocal compile_count
			compile_count += 1
			return graph_module.forward

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([Block(1.0), Block(2.0)])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0", "diffusion_model.blocks.1"],
			{"backend": counting_backend, "fullgraph": True, "dynamic": False},
			False,
		)

		output = wrapper(executor, torch.ones((1, 2)))

		self.assertTrue(torch.equal(output[0], torch.full((1, 2), 2.0)))
		self.assertTrue(torch.equal(output[1], torch.full((1, 2), 4.0)))
		self.assertEqual(compile_count, 1)
		for block in root.diffusion_model.blocks:
			self.assertIs(block.attention.__class__, Attention)
			self.assertIsInstance(block.attention.forward, functools.partial)

	def test_compile_wrapper_reports_each_guard_failure_once(self):
		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return self.class_obj.diffusion_model.blocks[0](value)

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0"],
			{"backend": lambda graph_module, _inputs: graph_module.forward, "fullgraph": True},
			True,
		)
		failure_map = {lazy_compile._dispatch_compiled_module.__code__: ["0/0: test guard failure"]}

		with mock.patch("torch._dynamo.utils.guard_failures", failure_map):
			with mock.patch.object(lazy_compile.logging, "info") as log_message:
				wrapper(executor, torch.ones(1))
				wrapper(executor, torch.ones(1))

		matching_calls = [
			call
			for call in log_message.call_args_list
			if "Dynamo cache miss" in str(call)
		]
		self.assertEqual(len(matching_calls), 1)

	def test_compile_wrapper_suppresses_diagnostics_when_not_verbose(self):
		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return self.class_obj.diffusion_model.blocks[0](value)

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0"],
			{"backend": lambda graph_module, _inputs: graph_module.forward, "fullgraph": True},
			False,
		)

		with mock.patch.object(
			torch._dynamo.eval_frame,
			"_debug_get_cache_entry_list",
		) as get_cache_entries:
			with mock.patch.object(lazy_compile.time, "perf_counter") as perf_counter:
				with mock.patch.object(lazy_compile.logging, "info") as log_message:
					output = wrapper(executor, torch.ones(1))

		self.assertTrue(torch.equal(output, torch.ones(1)))
		get_cache_entries.assert_not_called()
		perf_counter.assert_not_called()
		log_message.assert_not_called()

	def test_compile_wrapper_reports_pending_guard_failure_before_dispatch(self):
		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return self.class_obj.diffusion_model.blocks[0](value)

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0"],
			{"backend": lambda graph_module, _inputs: graph_module.forward, "fullgraph": True},
			True,
		)
		guard_info = SimpleNamespace(result=0, verbose_code_parts=["test pending guard failure"])
		cache_entry = SimpleNamespace(
			guard_manager=SimpleNamespace(check_verbose=mock.Mock(return_value=guard_info))
		)

		with mock.patch.object(
			torch._dynamo.eval_frame,
			"_debug_get_cache_entry_list",
			return_value=[cache_entry],
		):
			with mock.patch.object(lazy_compile.logging, "info") as log_message:
				wrapper(executor, torch.ones(1))

		self.assertTrue(any(
			"Pending Dynamo cache miss: test pending guard failure" in str(call)
			for call in log_message.call_args_list
		))

	def test_compile_wrapper_reports_graph_cache_growth(self):
		class Executor:
			def __init__(self, class_obj):
				self.class_obj = class_obj

			def __call__(self, value):
				return self.class_obj.diffusion_model.blocks[0](value)

		root = nn.Module()
		root.diffusion_model = nn.Module()
		root.diffusion_model.blocks = nn.ModuleList([nn.Identity()])
		executor = Executor(root)
		wrapper = lazy_compile._make_lazy_compile_wrapper(
			["diffusion_model.blocks.0"],
			{"backend": lambda graph_module, _inputs: graph_module.forward, "fullgraph": True},
			True,
		)

		with mock.patch.object(
			torch._dynamo.eval_frame,
			"_debug_get_cache_entry_list",
			side_effect=[[], [SimpleNamespace()]],
		):
			with mock.patch.object(lazy_compile.logging, "info") as log_message:
				wrapper(executor, torch.ones(1))

		self.assertTrue(any(
			"Dynamo graph cache grew to 1 entry after diffusion_model.blocks.0" in str(call)
			for call in log_message.call_args_list
		))


if __name__ == "__main__":
	unittest.main()
