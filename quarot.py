import torch

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

try:
	_disable_torch_compile = torch.compiler.disable
except Exception:
	try:
		import torch._dynamo as _torch_dynamo
		_disable_torch_compile = _torch_dynamo.disable
	except Exception:
		def _disable_torch_compile(fn):
			return fn


def _is_power_of_two(value: int) -> bool:
	return value > 0 and (value & (value - 1)) == 0


@_disable_torch_compile
def build_hadamard(
	size: int,
	device: str | torch.device = "cpu",
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""
	Build a normalized Hadamard matrix H where H @ H.T == I.
	Size must be a power of two.
	"""
	if not _is_power_of_two(size):
		raise ValueError(f"Hadamard size must be a power of two, got {size}")

	device_obj = torch.device(device)
	cache_key = (size, str(device_obj), dtype)
	cached = _HADAMARD_CACHE.get(cache_key)
	if cached is not None:
		return cached

	# Build in float32 for numerical stability, then cast once.
	h_matrix = torch.ones((1, 1), dtype=torch.float32)
	while h_matrix.shape[0] < size:
		top = torch.cat((h_matrix, h_matrix), dim=1)
		bottom = torch.cat((h_matrix, -h_matrix), dim=1)
		h_matrix = torch.cat((top, bottom), dim=0)

	h_matrix = h_matrix / (size ** 0.5)
	h_matrix = h_matrix.to(device=device_obj, dtype=dtype)
	_HADAMARD_CACHE[cache_key] = h_matrix
	return h_matrix


@_disable_torch_compile
def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
	"""
	Rotate weight matrix offline: W_rot = W @ H_block.T.
	Weight shape is expected to be [out_features, in_features].
	"""
	out_features, in_features = weight.shape
	if in_features % group_size != 0:
		raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")

	group_count = in_features // group_size
	grouped_weight = weight.view(out_features, group_count, group_size)
	h_transposed = h_matrix.T.to(device=weight.device, dtype=weight.dtype)
	rotated_weight = torch.matmul(grouped_weight, h_transposed)
	return rotated_weight.reshape(out_features, in_features)


@_disable_torch_compile
def rotate_activation(x: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
	"""
	Rotate activations online: x_rot = x @ H_block.
	Applies to the last feature dimension.
	"""
	original_shape = x.shape
	feature_count = original_shape[-1]
	if feature_count % group_size != 0:
		raise ValueError(f"features {feature_count} not divisible by group_size {group_size}")

	group_count = feature_count // group_size
	grouped_x = x.view(*original_shape[:-1], group_count, group_size)
	h_device = h_matrix.to(device=x.device, dtype=x.dtype)
	rotated_x = torch.matmul(grouped_x, h_device)
	return rotated_x.view(original_shape)
