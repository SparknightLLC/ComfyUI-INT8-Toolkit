import math


DEFAULT_INT4_MIXED_RATIO = 0.2
INT4_CONVROT_GROUP_SIZE = 256


def normalize_int4_mixed_ratio(value):
	try:
		ratio = float(value)
	except (TypeError, ValueError):
		return DEFAULT_INT4_MIXED_RATIO

	if not math.isfinite(ratio):
		return DEFAULT_INT4_MIXED_RATIO
	return min(1.0, max(0.0, ratio))


def normalize_layer_name(name):
	return str(name).rstrip(".")


def is_int4_compatible_linear(weight):
	shape = getattr(weight, "shape", ())
	return bool(
		len(shape) == 2
		and int(shape[0]) > 1
		and int(shape[1]) > 1
		and int(shape[1]) % INT4_CONVROT_GROUP_SIZE == 0
	)


def _select_evenly(names, count):
	if count <= 0:
		return []
	if count >= len(names):
		return list(names)

	return [
		names[min(len(names) - 1, int(((index + 0.5) * len(names)) / count))]
		for index in range(count)
	]


def _van_der_corput(index):
	value = 0.0
	factor = 0.5
	while index > 0:
		value += factor * (index % 2)
		index //= 2
		factor *= 0.5
	return value


def _nested_even_order(names):
	return [
		name
		for _index, name in sorted(
			enumerate(names),
			key=lambda entry: (_van_der_corput(entry[0] + 1), entry[0]),
		)
	]


def _build_mixed_layer_order(ordered_names, priority_patterns):
	priority_names = [
		name
		for name in ordered_names
		if any(pattern in name for pattern in priority_patterns)
	]
	priority_set = set(priority_names)
	remaining_names = [name for name in ordered_names if name not in priority_set]
	anchor_count = min(
		len(ordered_names),
		int(math.ceil(len(ordered_names) * DEFAULT_INT4_MIXED_RATIO)),
	)

	if len(priority_names) >= anchor_count:
		anchor_names = _select_evenly(priority_names, anchor_count)
	else:
		anchor_names = priority_names + _select_evenly(
			remaining_names,
			anchor_count - len(priority_names),
		)

	anchor_set = set(anchor_names)
	priority_anchor = [name for name in priority_names if name in anchor_set]
	fallback_anchor = [name for name in anchor_names if name not in priority_set]
	priority_remainder = [name for name in priority_names if name not in anchor_set]
	fallback_remainder = [
		name
		for name in remaining_names
		if name not in anchor_set
	]
	return (
		_nested_even_order(priority_anchor)
		+ _nested_even_order(fallback_anchor)
		+ _nested_even_order(priority_remainder)
		+ _nested_even_order(fallback_remainder)
	)


def select_int4_mixed_layers(layer_names, ratio, priority_patterns=()):
	ordered_names = list(dict.fromkeys(normalize_layer_name(name) for name in layer_names))
	target_count = int(math.ceil(len(ordered_names) * normalize_int4_mixed_ratio(ratio)))
	target_count = min(len(ordered_names), max(0, target_count))
	if target_count <= 0:
		return ()

	selected_set = set(_build_mixed_layer_order(ordered_names, priority_patterns)[:target_count])
	return tuple(name for name in ordered_names if name in selected_set)
