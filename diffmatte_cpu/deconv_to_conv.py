"""Rewrite a specific ConvTranspose pattern as four quantizable Conv layers.

Supported pattern: float32 constant W[Cin,Cout,4,4], group=1, stride=(2,2),
padding=(1,1,1,1), dilation=(1,1), output_padding=(0,0), auto_pad=NOTSET,
without an explicit output_shape attribute.  No quantization is performed.

For parity (a,b), the 2x2 Conv weight is obtained by selecting kernel offsets
((a+1)%2, (b+1)%2) with step 2, reversing both spatial axes, and exchanging
the input/output channel axes.  Its pads are (1-a, 1-b, a, b).  Concatenating
the outputs in order (0,0),(0,1),(1,0),(1,1), followed by DepthToSpace(DCR,2),
reconstructs the original output.  Arithmetic summation order may change, so
float32 output agreement is subject to ordinary floating-point rounding.

The inference semantics used here are specified by:
https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PHASE_PREFIX = "PhaseConv/"
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _reserved_names(graph: onnx.GraphProto) -> set[str]:
    names = {item.name for item in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
    for item in graph.sparse_initializer:
        names.update((item.values.name, item.indices.name))
    for node in graph.node:
        names.add(node.name)
        names.update(node.input)
        names.update(node.output)
    return names


def _references(graph: onnx.GraphProto) -> set[str]:
    """Conservatively retain weights captured by nested control-flow graphs."""
    names = {value.name for value in (*graph.input, *graph.output)}
    for node in graph.node:
        names.update(node.input)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                names.update(_references(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    names.update(_references(child))
    return names


def _tensor_information(graph: onnx.GraphProto) -> dict[str, tuple[int, list[Any] | None]]:
    info: dict[str, tuple[int, list[Any] | None]] = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        tensor = value.type.tensor_type
        shape = None
        if tensor.HasField("shape"):
            shape = [
                dim.dim_value if dim.HasField("dim_value") else (dim.dim_param or None)
                for dim in tensor.shape.dim
            ]
        info[value.name] = (tensor.elem_type, shape)
    for value in graph.initializer:
        info[value.name] = (value.data_type, list(value.dims))
    return info


def _match(
    node: onnx.NodeProto,
    weights: dict[str, onnx.TensorProto],
    graph_inputs: set[str],
    info: dict[str, tuple[int, list[Any] | None]],
    opset: int,
) -> str | None:
    """Return an exclusion reason, or None for the proven supported pattern."""
    if node.domain != "":
        return "nonstandard_operator_domain"
    if opset < 11:
        return "opset_below_11_does_not_support_explicit_DCR_mode"
    if len(node.input) not in (2, 3) or len(node.output) != 1:
        return "unsupported_input_or_output_arity"
    if node.input[1] not in weights:
        return "weight_is_not_an_initializer"
    if node.input[1] in graph_inputs:
        return "weight_is_an_overridable_graph_input"
    weight = weights[node.input[1]]
    if weight.data_type != TensorProto.FLOAT:
        return "weight_is_not_float32"
    if len(weight.dims) != 4 or list(weight.dims[2:]) != [4, 4] or min(weight.dims) <= 0:
        return "weight_is_not_nonempty_Cin_Cout_4_4"
    attributes = {value.name: helper.get_attribute_value(value) for value in node.attribute}
    supported = {"auto_pad", "dilations", "group", "kernel_shape", "output_padding", "output_shape", "pads", "strides"}
    if len(attributes) != len(node.attribute) or set(attributes) - supported:
        return "unrecognized_or_duplicate_attributes"
    if "output_shape" in attributes:
        return "explicit_output_shape_is_not_supported"
    auto_pad = attributes.get("auto_pad", b"NOTSET")
    if auto_pad not in ("NOTSET", b"NOTSET"):
        return "auto_pad_is_not_NOTSET"
    expected = {
        "group": (1, 1),
        "kernel_shape": ([4, 4], [4, 4]),
        "strides": ([1, 1], [2, 2]),
        "pads": ([0, 0, 0, 0], [1, 1, 1, 1]),
        "dilations": ([1, 1], [1, 1]),
        "output_padding": ([0, 0], [0, 0]),
    }
    for key, (default, required) in expected.items():
        if attributes.get(key, default) != required:
            return f"unsupported_{key}"
    cin, cout = map(int, weight.dims[:2])
    for tensor_name, channels in ((node.input[0], cin), (node.output[0], cout)):
        if tensor_name not in info:
            continue
        dtype, shape = info[tensor_name]
        if dtype not in (TensorProto.UNDEFINED, TensorProto.FLOAT):
            return "input_or_output_is_not_float32"
        if shape is not None:
            if len(shape) != 4:
                return "input_or_output_is_not_NCHW"
            if isinstance(shape[1], int) and shape[1] != channels:
                return "input_or_output_channel_count_mismatch"
            if any(isinstance(dimension, int) and dimension <= 0 for dimension in shape):
                return "nonpositive_input_or_output_dimension"
    if len(node.input) == 3 and node.input[2] and node.input[2] in info:
        dtype, shape = info[node.input[2]]
        if dtype not in (TensorProto.UNDEFINED, TensorProto.FLOAT):
            return "bias_is_not_float32"
        if shape is not None and (len(shape) != 1 or (isinstance(shape[0], int) and shape[0] != cout)):
            return "bias_shape_does_not_match_Cout"
    return None


def rewrite_convtranspose(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Return a copied model with supported ConvTranspose nodes decomposed.

    All original tensor names, graph inputs/outputs, and opset imports remain
    unchanged.  Each replacement DepthToSpace also retains the original node
    name and documentation.  New Conv and Concat nodes use PhaseConv/ names.
    Repeated consumers of one weight reuse the four phase-weight initializers.
    Unsupported consumers retain their original weight, including subgraph
    captures.  Unsupported ConvTranspose configurations are reported and kept.

    The report's quantize_node_names contains only the added Conv node names.
    Run calibration after this transformation, not before it.
    """
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an onnx.ModelProto")
    converted = onnx.ModelProto()
    converted.CopyFrom(model)
    graph = converted.graph
    original_nodes = list(graph.node)
    weights = {value.name: value for value in graph.initializer}
    if len(weights) != len(graph.initializer):
        raise ValueError("Duplicate initializer names make the graph ambiguous")
    graph_inputs = {value.name for value in graph.input}
    info = _tensor_information(graph)
    opset = next((value.version for value in converted.opset_import if value.domain == ""), 0)
    reserved = _reserved_names(graph)

    def new_name(base: str) -> str:
        value = base
        suffix = 1
        while value in reserved:
            value = f"{base}_{suffix}"
            suffix += 1
        reserved.add(value)
        return value

    report: dict[str, Any] = {
        "method": "ConvTranspose_4x4_s2_p1_to_four_phase_Conv_and_DepthToSpace_DCR",
        "quantize_node_names": [],
        "converted": [],
        "skipped": [],
        "removed_initializers": [],
    }
    phase_weights: dict[str, list[str]] = {}
    added_weights: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    for index, original in enumerate(original_nodes):
        if original.op_type != "ConvTranspose":
            nodes.append(original)
            continue
        reason = _match(original, weights, graph_inputs, info, opset)
        if reason is not None:
            nodes.append(original)
            report["skipped"].append({"node_name": original.name, "reason": reason})
            continue
        weight_name = original.input[1]
        cin, cout = map(int, weights[weight_name].dims[:2])
        if weight_name not in phase_weights:
            try:
                weight = numpy_helper.to_array(weights[weight_name])
            except Exception as exc:
                raise ValueError(f"Cannot load {weight_name!r}; load all external tensor data first") from exc
            phase_weights[weight_name] = []
            for a, b in PHASES:
                selected = weight[:, :, (a + 1) % 2 :: 2, (b + 1) % 2 :: 2]
                phase = np.ascontiguousarray(selected[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))
                name = new_name(f"{PHASE_PREFIX}weight/{weight_name}/p{a}{b}")
                added_weights.append(numpy_helper.from_array(phase, name))
                phase_weights[weight_name].append(name)
        outputs: list[str] = []
        phase_node_names: list[str] = []
        input_shape = info.get(original.input[0], (0, None))[1]
        bias = original.input[2] if len(original.input) == 3 else ""
        for (a, b), phase_weight_name in zip(PHASES, phase_weights[weight_name]):
            output = new_name(f"{PHASE_PREFIX}output/{index}/p{a}{b}")
            name = new_name(f"{PHASE_PREFIX}conv/{index}/p{a}{b}")
            inputs = [original.input[0], phase_weight_name]
            if bias:
                inputs.append(bias)
            nodes.append(
                helper.make_node(
                    "Conv", inputs, [output], name=name, group=1, kernel_shape=[2, 2],
                    pads=[1 - a, 1 - b, a, b], strides=[1, 1], dilations=[1, 1],
                )
            )
            if input_shape is not None:
                graph.value_info.append(
                    helper.make_tensor_value_info(output, TensorProto.FLOAT, [input_shape[0], cout, *input_shape[2:]])
                )
            outputs.append(output)
            phase_node_names.append(name)
        concatenated = new_name(f"{PHASE_PREFIX}concatenated/{index}")
        nodes.append(helper.make_node("Concat", outputs, [concatenated], name=new_name(f"{PHASE_PREFIX}concat/{index}"), axis=1))
        if input_shape is not None:
            graph.value_info.append(
                helper.make_tensor_value_info(concatenated, TensorProto.FLOAT, [input_shape[0], 4 * cout, *input_shape[2:]])
            )
        replacement = onnx.NodeProto()
        replacement.CopyFrom(original)
        replacement.op_type = "DepthToSpace"
        if not replacement.name:
            replacement.name = new_name(f"{PHASE_PREFIX}depth_to_space/{index}")
        del replacement.input[:]
        replacement.input.append(concatenated)
        del replacement.attribute[:]
        replacement.attribute.extend([helper.make_attribute("blocksize", 2), helper.make_attribute("mode", "DCR")])
        nodes.append(replacement)
        report["quantize_node_names"].extend(phase_node_names)
        report["converted"].append(
            {
                "original_node_name": original.name,
                "depth_to_space_node_name": replacement.name,
                "input": original.input[0],
                "output": original.output[0],
                "weight": weight_name,
                "bias": bias or None,
                "in_channels": cin,
                "out_channels": cout,
                "phase_conv_node_names": phase_node_names,
                "phase_weight_names": phase_weights[weight_name],
            }
        )
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(added_weights)
    used = _references(graph)
    removed = set(phase_weights) - used
    retained = [value for value in graph.initializer if value.name not in removed]
    del graph.initializer[:]
    graph.initializer.extend(retained)
    report["removed_initializers"] = sorted(removed)
    report["counts"] = {
        "converted_convtranspose": len(report["converted"]),
        "new_conv": len(report["quantize_node_names"]),
        "new_depth_to_space": len(report["converted"]),
        "distinct_original_weights": len(phase_weights),
        "new_phase_weights": len(added_weights),
        "removed_original_weights": len(removed),
        "skipped_convtranspose": len(report["skipped"]),
    }
    if report["converted"]:
        properties = {value.key: value.value for value in converted.metadata_props}
        properties.update(
            {
                "phaseconv.implementation_version": "1",
                "phaseconv.method": report["method"],
                "phaseconv.quantize_node_names": json.dumps(report["quantize_node_names"], ensure_ascii=False),
            }
        )
        helper.set_model_props(converted, properties)
    return converted, report


def _scatter_reference(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Independent direct transposed-convolution definition in float64."""
    n, cin, height, width = x.shape
    cout = weight.shape[1]
    result = np.zeros((n, cout, 2 * height, 2 * width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            for kh in range(4):
                for kw in range(4):
                    oh, ow = 2 * i + kh - 1, 2 * j + kw - 1
                    if 0 <= oh < 2 * height and 0 <= ow < 2 * width:
                        result[:, :, oh, ow] += x[:, :, i, j].astype(np.float64) @ weight[:, :, kh, kw].astype(np.float64)
    if bias is not None:
        result += bias[None, :, None, None]
    return result


def self_test() -> dict[str, Any]:
    """Check tiny CPU models, edge spatial sizes, channel ordering and sharing."""
    import onnxruntime as ort

    rng = np.random.default_rng(20261017)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cases: list[dict[str, Any]] = []
    configurations = [
        (1, 1, 1, 1, 1, False, False),
        (1, 1, 3, 1, 4, True, False),
        (1, 3, 1, 4, 1, True, False),
        (2, 2, 3, 2, 3, True, False),
        (1, 5, 4, 3, 5, False, True),
        (2, 3, 5, 1, 1, True, True),
    ]
    maximum_difference = 0.0
    reference_difference = 0.0
    for n, cin, cout, height, width, use_bias, dynamic in configurations:
        x = rng.standard_normal((n, cin, height, width)).astype(np.float32)
        weight = rng.standard_normal((cin, cout, 4, 4)).astype(np.float32)
        bias = (rng.standard_normal(cout) + 0.75).astype(np.float32) if use_bias else None
        inputs = ["x", "W"] + (["bias"] if use_bias else [])
        initializers = [numpy_helper.from_array(weight, "W")]
        if use_bias:
            initializers.append(numpy_helper.from_array(bias, "bias"))
        shape = ["N", cin, "H", "Wdim"] if dynamic else list(x.shape)
        output_shape = ["N", cout, "OH", "OW"] if dynamic else [n, cout, 2 * height, 2 * width]
        nodes = [
            helper.make_node("ConvTranspose", inputs, ["y"], name="PhaseConv/conv/0/p00", strides=[2, 2], pads=[1, 1, 1, 1]),
            helper.make_node("ConvTranspose", inputs, ["again"], name="shared_supported", strides=[2, 2], pads=[1, 1, 1, 1]),
            helper.make_node("ConvTranspose", inputs, ["unsupported"], name="shared_unsupported", strides=[1, 1], pads=[1, 1, 1, 1]),
        ]
        graph = helper.make_graph(
            nodes, "phase_conv_validation", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape),
             helper.make_tensor_value_info("again", TensorProto.FLOAT, output_shape),
             helper.make_tensor_value_info("unsupported", TensorProto.FLOAT, [None, cout, None, None])],
            initializers,
        )
        original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        original.ir_version = 9
        source = original.SerializeToString()
        converted, report = rewrite_convtranspose(original)
        assert original.SerializeToString() == source, "Input model was mutated"
        assert report["counts"]["converted_convtranspose"] == 2
        assert report["counts"]["new_phase_weights"] == 4
        assert report["skipped"] == [{"node_name": "shared_unsupported", "reason": "unsupported_strides"}]
        assert "W" in {value.name for value in converted.graph.initializer}
        assert list(converted.graph.input) == list(original.graph.input)
        assert list(converted.graph.output) == list(original.graph.output)
        assert list(converted.opset_import) == list(original.opset_import)
        assert len({node.name for node in converted.graph.node}) == len(converted.graph.node)
        onnx.checker.check_model(converted, full_check=True)
        baseline = ort.InferenceSession(source, sess_options=options, providers=["CPUExecutionProvider"]).run(None, {"x": x})
        actual = ort.InferenceSession(converted.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]).run(None, {"x": x})
        reference = _scatter_reference(x, weight, bias)
        difference = max(float(np.max(np.abs(a - b))) for a, b in zip(actual, baseline))
        maximum_difference = max(maximum_difference, difference)
        reference_difference = max(reference_difference, float(np.max(np.abs(actual[0] - reference))))
        for a, b in zip(actual, baseline):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(actual[0], reference, rtol=1e-5, atol=2e-5)
        np.testing.assert_array_equal(actual[2], baseline[2])
        cases.append({"input_shape": list(x.shape), "out_channels": cout, "bias": use_bias, "dynamic_shape": dynamic, "max_abs_error": difference})

        # With the unsupported consumer removed, the old weight must be pruned.
        no_remaining_consumer = onnx.ModelProto()
        no_remaining_consumer.CopyFrom(original)
        del no_remaining_consumer.graph.node[-1]
        del no_remaining_consumer.graph.output[-1]
        pruned, pruned_report = rewrite_convtranspose(no_remaining_consumer)
        assert pruned_report["removed_initializers"] == ["W"]
        assert "W" not in {value.name for value in pruned.graph.initializer}
        onnx.checker.check_model(pruned, full_check=True)
    return {
        "status": "passed", "max_abs_error_vs_ConvTranspose": maximum_difference,
        "max_abs_error_vs_float64_scatter": reference_difference,
        "cases": cases, "numpy_version": np.__version__, "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Validate with tiny ONNX Runtime CPU models")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("Import rewrite_convtranspose in a conversion tool, or use --self-test")
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
