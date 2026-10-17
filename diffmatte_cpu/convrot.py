"""Apply the ConvRot paper's regular Hadamard rotation to ONNX linear layers.

This is an independent NumPy/ONNX implementation of the rotation described in
arXiv:2512.03673, equations (9)--(13), not the authors' CUDA implementation.
It performs no quantization.  For ``Y = X @ B`` it inserts ``X @ R`` and stores
``R.T @ B``, where R is block diagonal with normalized regular Hadamard blocks.

Only top-level, initializer-backed, float32 ``MatMul`` layers are transformed.
Biases, attention products, graph inputs/outputs, and the opset remain intact.
New rotation MatMul nodes start with ``ConvRot/rotation/`` and must be excluded
from subsequent quantization.  Use ``report["quantize_node_names"]`` to target
the transformed original linear layers.

Example validation command (onnxruntime is needed only for this self-test)::

    python convrot.py --self-test
"""

from __future__ import annotations

import argparse
import json
import operator
from collections import Counter
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PAPER_URL = "https://arxiv.org/abs/2512.03673"
ROTATION_PREFIX = "ConvRot/rotation/"
_METADATA_VERSION_KEY = "convrot.rotation_implementation_version"
_REGULAR_H4 = np.asarray(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
    dtype=np.float64,
)


def _validate_group_size(group_size: int) -> int:
    """Require 4**k, k >= 1; an arbitrary power of two is insufficient."""
    if isinstance(group_size, (bool, np.bool_)):
        raise ValueError("group_size must be an integer power of four >= 4")
    try:
        value = operator.index(group_size)
    except TypeError as exc:
        raise ValueError("group_size must be an integer power of four >= 4") from exc
    reduced = value
    while reduced >= 4 and reduced % 4 == 0:
        reduced //= 4
    if value < 4 or reduced != 1:
        raise ValueError("group_size must be an integer power of four >= 4")
    return value


def regular_hadamard(group_size: int, dtype: Any = np.float32) -> np.ndarray:
    """Return H4's normalized Kronecker power, an orthogonal symmetric matrix.

    The rows and columns sum to +1 after normalization.  In particular, a
    constant row remains constant rather than concentrating in one channel.
    """
    group_size = _validate_group_size(group_size)
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("The normalized rotation requires a floating-point dtype")
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < group_size:
        matrix = np.kron(matrix, _REGULAR_H4)
    return (matrix / np.sqrt(group_size)).astype(dtype)


def _all_identifiers(graph: onnx.GraphProto) -> set[str]:
    """Reserve both tensor and node identifiers in the modified graph scope."""
    names = {item.name for item in graph.initializer}
    names.update(item.name for item in graph.input)
    names.update(item.name for item in graph.output)
    names.update(item.name for item in graph.value_info)
    for item in graph.sparse_initializer:
        names.add(item.values.name)
        names.add(item.indices.name)
    for node in graph.node:
        names.add(node.name)
        names.update(node.input)
        names.update(node.output)
    names.discard("")
    return names


class _Names:
    """Allocate stable names without changing any existing tensor identifier."""

    def __init__(self, graph: onnx.GraphProto) -> None:
        self.used = _all_identifiers(graph)

    def new(self, base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in self.used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.used.add(candidate)
        return candidate


def _known_shapes(graph: onnx.GraphProto) -> dict[str, tuple[int, ...]]:
    """Read fully specified, nonempty shapes; leave symbolic shapes dynamic."""
    result: dict[str, tuple[int, ...]] = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            result[value.name] = tuple(dim.dim_value for dim in dims)
    for value in graph.initializer:
        if value.dims and all(dim > 0 for dim in value.dims):
            result[value.name] = tuple(value.dims)
    return result


def _referenced_names(graph: onnx.GraphProto) -> set[str]:
    """Include nested graph captures when deciding whether a weight is unused."""
    result = {value.name for value in (*graph.input, *graph.output)}
    for node in graph.node:
        result.update(node.input)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                result.update(_referenced_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    result.update(_referenced_names(nested))
    return result


def apply_convrot(
    model: onnx.ModelProto,
    group_size: int = 64,
    node_names: Iterable[str] | None = None,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Copy a model and rotate eligible constant-weight MatMul layers.

    Args:
        model: A loaded ONNX model, including any external tensor data.
        group_size: A power of four >= 4 that must divide each selected K.
        node_names: Original node names to consider, or None for all eligible
            top-level MatMul nodes.  An empty collection transforms no nodes.

    Returns:
        A new model and a JSON-serializable transformation report.  The input
        model is never mutated.  Ineligible or unmatched selections are reported.
        The report's quantize_node_names are the effective names of the original
        linears, excluding the newly inserted rotation matrix multiplications.

    Notes:
        Initializers that are also graph inputs are overridable and are skipped.
        Rotated weights are separate initializers, so an unselected consumer of a
        shared original weight retains its exact semantics.  Multiple selected
        consumers share one rotated copy.  Only unused transformed originals are
        pruned; nested graph captures and the original graph interface are kept.

        This pass must start from an unrotated model.  It requires opset >= 5 and
        does not upgrade the opset.  For opset < 14, input dimensions must be
        nonzero; opset >= 14 uses allowzero=1 when restoring the original shape.
    """
    group_size = _validate_group_size(group_size)
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    if any(item.key == _METADATA_VERSION_KEY for item in model.metadata_props):
        raise ValueError("Model already has this ConvRot pass; start from the unrotated model")
    if isinstance(node_names, str):
        raise TypeError("node_names must be a collection of node names, not one string")
    requested = None if node_names is None else set(node_names)
    if requested is not None and not all(isinstance(name, str) for name in requested):
        raise TypeError("Every node_names entry must be a string")

    opset = next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), None)
    if opset is None or opset < 5:
        raise ValueError("ConvRot insertion requires the original standard ONNX opset >= 5")

    converted = onnx.ModelProto()
    converted.CopyFrom(model)
    graph = converted.graph
    original_nodes = list(graph.node)
    initializers = {value.name: value for value in graph.initializer}
    if len(initializers) != len(graph.initializer):
        raise ValueError("Duplicate initializer names make this graph ambiguous")
    overridable = {value.name for value in graph.input}
    shapes = _known_shapes(graph)
    names = _Names(graph)
    original_name_counts = Counter(node.name for node in original_nodes if node.name)
    original_node_names = {node.name for node in original_nodes}
    matrix = regular_hadamard(group_size)
    added_initializers: list[onnx.TensorProto] = []
    output_nodes: list[onnx.NodeProto] = []
    rotated_weights: dict[str, str] = {}
    static_shape_names: dict[tuple[int, ...], str] = {}
    matrix_name: str | None = None
    flat_shape_name: str | None = None
    report: dict[str, Any] = {
        "method": "ConvRot regular Hadamard rotation; independent ONNX implementation",
        "paper": PAPER_URL,
        "group_size": group_size,
        "quantize_node_names": [],
        "original_matmul_names": [],
        "rotation_matmul_nodes": [],
        "transformed": [],
        "skipped": [],
        "unmatched_requested_node_names": sorted((requested or set()) - original_node_names),
        "removed_initializers": [],
    }

    def skip(node: onnx.NodeProto, reason: str) -> None:
        report["skipped"].append({"node_name": node.name, "reason": reason})

    for index, original in enumerate(original_nodes):
        if requested is not None and original.name not in requested:
            output_nodes.append(original)
            continue
        if original.op_type != "MatMul" or original.domain not in ("", "ai.onnx"):
            if requested is not None:
                skip(original, "not_a_standard_matmul")
            output_nodes.append(original)
            continue
        if original.name.startswith(ROTATION_PREFIX):
            skip(original, "rotation_matmul_is_not_a_linear_quantization_target")
            output_nodes.append(original)
            continue
        if len(original.input) != 2 or original.input[1] not in initializers:
            skip(original, "right_operand_is_not_an_original_initializer")
            output_nodes.append(original)
            continue
        weight_name = original.input[1]
        weight_tensor = initializers[weight_name]
        if weight_name in overridable:
            skip(original, "weight_initializer_is_an_overridable_graph_input")
            output_nodes.append(original)
            continue
        if weight_tensor.data_type != TensorProto.FLOAT:
            skip(original, "weight_dtype_is_not_float32")
            output_nodes.append(original)
            continue
        if len(weight_tensor.dims) != 2 or min(weight_tensor.dims) <= 0:
            skip(original, "weight_is_not_a_nonempty_2d_matrix")
            output_nodes.append(original)
            continue
        in_features, out_features = map(int, weight_tensor.dims)
        if in_features % group_size:
            skip(original, f"in_features_{in_features}_not_divisible_by_group_size_{group_size}")
            output_nodes.append(original)
            continue
        input_name = original.input[0]
        input_shape = shapes.get(input_name)
        if input_shape is not None and input_shape[-1] != in_features:
            raise ValueError(
                f"MatMul {original.name!r}: input last dimension {input_shape[-1]} "
                f"differs from weight K={in_features}"
            )

        if matrix_name is None:
            matrix_name = names.new(f"ConvRot/regular_H{group_size}")
            added_initializers.append(numpy_helper.from_array(matrix, matrix_name))
            flat_shape_name = names.new(f"ConvRot/flat_shape_{group_size}")
            added_initializers.append(
                numpy_helper.from_array(np.asarray([-1, group_size], dtype=np.int64), flat_shape_name)
            )
        if weight_name not in rotated_weights:
            try:
                weight = numpy_helper.to_array(weight_tensor)
            except Exception as exc:
                raise ValueError(
                    f"Cannot read weight {weight_name!r}; load the model including external tensor data"
                ) from exc
            # B[K,N] is transformed on its K axis: R.T @ B.  Reordering into
            # [N, K/group, group] avoids materializing the large block diagonal R.
            rows = weight.T.reshape(out_features, in_features // group_size, group_size)
            rotated = (rows @ matrix).reshape(out_features, in_features).T
            rotated_name = names.new(f"ConvRot/weight/{weight_name}")
            added_initializers.append(
                numpy_helper.from_array(np.ascontiguousarray(rotated, dtype=np.float32), rotated_name)
            )
            rotated_weights[weight_name] = rotated_name
        rotated_weight_name = rotated_weights[weight_name]

        flat_name = names.new(f"ConvRot/flat/{index}")
        rotated_flat_name = names.new(f"ConvRot/rotated_flat/{index}")
        rotated_input_name = names.new(f"ConvRot/input/{index}")
        rotation_node_name = names.new(f"{ROTATION_PREFIX}{index}")
        inserted: list[onnx.NodeProto] = []
        if input_shape is not None:
            if input_shape not in static_shape_names:
                restore_shape_name = names.new(f"ConvRot/restore_shape/{index}")
                added_initializers.append(
                    numpy_helper.from_array(np.asarray(input_shape, dtype=np.int64), restore_shape_name)
                )
                static_shape_names[input_shape] = restore_shape_name
            restore_shape_name = static_shape_names[input_shape]
            shape_mode = "static"
        else:
            restore_shape_name = names.new(f"ConvRot/original_shape/{index}")
            inserted.append(
                helper.make_node("Shape", [input_name], [restore_shape_name], name=names.new(f"ConvRot/shape/{index}"))
            )
            shape_mode = "dynamic"
        inserted.extend(
            [
                helper.make_node(
                    "Reshape", [input_name, flat_shape_name], [flat_name],
                    name=names.new(f"ConvRot/reshape_in/{index}"),
                ),
                helper.make_node("MatMul", [flat_name, matrix_name], [rotated_flat_name], name=rotation_node_name),
                helper.make_node(
                    "Reshape", [rotated_flat_name, restore_shape_name], [rotated_input_name],
                    name=names.new(f"ConvRot/reshape_out/{index}"),
                    **({"allowzero": 1} if opset >= 14 else {}),
                ),
            ]
        )
        linear = onnx.NodeProto()
        linear.CopyFrom(original)
        if not linear.name or original_name_counts[linear.name] > 1:
            linear.name = names.new(f"ConvRot/linear/{index}")
        linear.input[0] = rotated_input_name
        linear.input[1] = rotated_weight_name
        output_nodes.extend(inserted)
        output_nodes.append(linear)
        report["quantize_node_names"].append(linear.name)
        report["original_matmul_names"].append(original.name)
        report["rotation_matmul_nodes"].append(rotation_node_name)
        report["transformed"].append(
            {
                "original_name": original.name,
                "effective_name": linear.name,
                "input": input_name,
                "original_weight": weight_name,
                "rotated_weight": rotated_weight_name,
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "shape_mode": shape_mode,
                "rotation_node": rotation_node_name,
            }
        )

    del graph.node[:]
    graph.node.extend(output_nodes)
    graph.initializer.extend(added_initializers)
    used = _referenced_names(graph)
    pruned = {name for name in rotated_weights if name not in used}
    retained = [value for value in graph.initializer if value.name not in pruned]
    del graph.initializer[:]
    graph.initializer.extend(retained)
    report["removed_initializers"] = sorted(pruned)
    report["counts"] = {
        "rotated_matmul": len(report["transformed"]),
        "distinct_rotated_weights": len(rotated_weights),
        "rotation_matmul": len(report["rotation_matmul_nodes"]),
        "skipped": len(report["skipped"]),
        "removed_original_weights": len(pruned),
    }
    if report["transformed"]:
        metadata = {item.key: item.value for item in converted.metadata_props}
        metadata.update(
            {
                _METADATA_VERSION_KEY: "1",
                "convrot.paper": PAPER_URL,
                "convrot.group_size": str(group_size),
                "convrot.quantize_node_names": json.dumps(report["quantize_node_names"], ensure_ascii=False),
                "convrot.original_matmul_names": json.dumps(report["original_matmul_names"], ensure_ascii=False),
                "convrot.rotation_matmul_nodes": json.dumps(report["rotation_matmul_nodes"], ensure_ascii=False),
            }
        )
        helper.set_model_props(converted, metadata)
    return converted, report


def self_test() -> dict[str, Any]:
    """Check the mathematical identity and execute tiny ONNX models on CPU.

    No user model or full image inference is used.  ONNX Runtime is imported
    here only, keeping the transformation's dependencies to NumPy and ONNX.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(251203673)
    mathematical_error = 0.0
    for size in (4, 16, 64, 256):
        rotation = regular_hadamard(size, np.float64)
        np.testing.assert_array_equal(rotation.T @ rotation, np.eye(size))
        np.testing.assert_array_equal(rotation.sum(axis=0), np.ones(size))
        np.testing.assert_array_equal(rotation.sum(axis=1), np.ones(size))
        x = rng.standard_normal((2, 3, 3 * size))
        b = rng.standard_normal((3 * size, 7))
        xr = (x.reshape(-1, size) @ rotation).reshape(x.shape)
        br = (b.T.reshape(7, 3, size) @ rotation).reshape(7, 3 * size).T
        difference = float(np.max(np.abs(x @ b - xr @ br)))
        mathematical_error = max(mathematical_error, difference)
        np.testing.assert_allclose(xr @ br, x @ b, rtol=1e-11, atol=1e-11)

    for invalid in (0, 1, 2, 8, 32, -4, True, 4.0):
        try:
            regular_hadamard(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid group_size was accepted: {invalid!r}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    maximum_onnx_error = 0.0
    cases: list[dict[str, Any]] = []
    for dynamic in (False, True):
        shape = ["batch", 3, 8] if dynamic else [2, 3, 8]
        b = rng.standard_normal((8, 5)).astype(np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "shared_weight"], ["y"], name="selected_linear"),
                helper.make_node("MatMul", ["x", "shared_weight"], ["unselected_y"], name="unselected_linear"),
            ],
            "convrot_validation",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [shape[0], 3, 5]),
                helper.make_tensor_value_info("unselected_y", TensorProto.FLOAT, [shape[0], 3, 5]),
            ],
            [numpy_helper.from_array(b, "shared_weight")],
        )
        original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        original.ir_version = 9
        source_bytes = original.SerializeToString()
        input_data = rng.standard_normal((2, 3, 8)).astype(np.float32)
        original_session = ort.InferenceSession(source_bytes, sess_options=options, providers=["CPUExecutionProvider"])
        expected = original_session.run(None, {"x": input_data})
        for selected in ({"selected_linear"}, None):
            converted, report = apply_convrot(original, group_size=4, node_names=selected)
            assert original.SerializeToString() == source_bytes, "Input model was mutated"
            onnx.checker.check_model(converted, full_check=True)
            assert list(converted.graph.input) == list(original.graph.input)
            assert list(converted.graph.output) == list(original.graph.output)
            assert list(converted.opset_import) == list(original.opset_import)
            initializer_names = {value.name for value in converted.graph.initializer}
            assert ("shared_weight" in initializer_names) == (selected is not None)
            assert report["counts"]["distinct_rotated_weights"] == 1
            assert report["quantize_node_names"] == (
                ["selected_linear"] if selected is not None else ["selected_linear", "unselected_linear"]
            )
            assert all(name.startswith(ROTATION_PREFIX) for name in report["rotation_matmul_nodes"])
            session = ort.InferenceSession(converted.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {"x": input_data})
            error = max(float(np.max(np.abs(a - e))) for a, e in zip(actual, expected))
            maximum_onnx_error = max(maximum_onnx_error, error)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, rtol=3e-6, atol=3e-6)
            if selected is not None:
                np.testing.assert_array_equal(actual[1], expected[1])
            cases.append({"dynamic_shape": dynamic, "all_consumers_rotated": selected is None, "max_abs_error": error})
    return {
        "status": "passed",
        "mathematical_max_abs_error": mathematical_error,
        "onnx_max_abs_error": maximum_onnx_error,
        "onnx_cases": cases,
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run mathematical and tiny CPU ONNX validation")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("Import apply_convrot in a conversion tool, or use --self-test")
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
