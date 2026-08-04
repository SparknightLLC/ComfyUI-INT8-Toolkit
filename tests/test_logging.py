import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LoggingTests(unittest.TestCase):
	def test_runtime_code_does_not_call_print(self):
		print_locations = []
		for path in PROJECT_ROOT.glob("*.py"):
			tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
			for node in ast.walk(tree):
				if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
					print_locations.append(f"{path.name}:{node.lineno}")

		self.assertEqual(print_locations, [])

	def test_runtime_code_does_not_use_legacy_product_name(self):
		legacy_locations = []
		for path in PROJECT_ROOT.glob("*.py"):
			for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
				if "INT8 Toolkit" in line or "ComfyUI-INT8-Toolkit" in line:
					legacy_locations.append(f"{path.name}:{line_number}")

		self.assertEqual(legacy_locations, [])


if __name__ == "__main__":
	unittest.main()
