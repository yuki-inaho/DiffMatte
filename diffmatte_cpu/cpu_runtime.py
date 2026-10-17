"""圧縮された定数の畳み込み重みを、CPU セッション作成時だけ float32 に戻します。

ONNX Runtime が Conv / ConvTranspose の重みを定数として事前処理できるように
する補助処理です。保存した ONNX は書き換えず、新しい量子化も行いません。
量子化された入力直後の Conv、および MatMul の入力は変更しません。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class ConstantEvaluationError(ValueError):
    """値を安全に定数として展開できない場合の例外です。"""


@dataclass(frozen=True)
class ConstantValue:
    array: np.ndarray
    nodes: frozenset[int] = frozenset()
    initializers: frozenset[str] = frozenset()
    contains_dequantization: bool = False


class ConstantEvaluator:
    """対応する定数演算だけを評価します。既存の配列は変更しません。"""

    def __init__(self, graph: onnx.GraphProto) -> None:
        self.initializers = {item.name: item for item in graph.initializer}
        self.graph_inputs = {item.name for item in graph.input}
        self.producers = {name: (index, node) for index, node in enumerate(graph.node) for name in node.output}
        self.cache: dict[str, ConstantValue] = {}
        self.active: set[str] = set()

    def evaluate(self, name: str) -> ConstantValue:
        if name in self.graph_inputs:
            raise ConstantEvaluationError("グラフ入力は実行時に変更できるため定数扱いしません。")
        if name in self.cache:
            return self.cache[name]
        if name in self.active:
            raise ConstantEvaluationError("定数の依存関係に循環があります。")
        self.active.add(name)
        try:
            if name in self.initializers:
                result = ConstantValue(
                    numpy_helper.to_array(self.initializers[name]),
                    initializers=frozenset([name]),
                )
            else:
                result = self._evaluate_node(name)
            self.cache[name] = result
            return result
        finally:
            self.active.remove(name)

    def _evaluate_node(self, name: str) -> ConstantValue:
        if name not in self.producers:
            raise ConstantEvaluationError("初期化済み定数または対応する定数演算ではありません。")
        index, node = self.producers[name]
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1:
            raise ConstantEvaluationError("独自演算または複数出力の演算は展開しません。")
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        if node.op_type == "Constant":
            array = self._constant(attributes)
            return ConstantValue(array, nodes=frozenset([index]))
        if node.op_type not in ("Identity", "DequantizeLinear", "Reshape", "Transpose"):
            raise ConstantEvaluationError(f"未対応の定数演算です: {node.op_type}")
        values = [self.evaluate(item) for item in node.input if item]
        if not values:
            raise ConstantEvaluationError("必要な入力がありません。")
        if node.op_type == "Identity":
            if len(values) != 1:
                raise ConstantEvaluationError("Identity の入力数が正しくありません。")
            array = values[0].array
        elif node.op_type == "DequantizeLinear":
            array = self._dequantize(values, attributes)
        elif node.op_type == "Reshape":
            array = self._reshape(values, attributes)
        else:
            if len(values) != 1:
                raise ConstantEvaluationError("Transpose の入力数が正しくありません。")
            source = values[0].array
            perm = list(attributes.get("perm", reversed(range(source.ndim))))
            if sorted(perm) != list(range(source.ndim)):
                raise ConstantEvaluationError("Transpose の軸指定が正しくありません。")
            array = np.transpose(source, perm)
        return ConstantValue(
            array=array,
            nodes=frozenset([index]).union(*(item.nodes for item in values)),
            initializers=frozenset().union(*(item.initializers for item in values)),
            contains_dequantization=(
                node.op_type == "DequantizeLinear" or any(item.contains_dequantization for item in values)
            ),
        )

    @staticmethod
    def _constant(attributes: dict[str, Any]) -> np.ndarray:
        if "value" in attributes:
            return numpy_helper.to_array(attributes["value"])
        conversions = {
            "value_int": np.int64, "value_ints": np.int64,
            "value_float": np.float32, "value_floats": np.float32,
        }
        for key, dtype in conversions.items():
            if key in attributes:
                return np.asarray(attributes[key], dtype=dtype)
        raise ConstantEvaluationError("数値以外の Constant は展開しません。")

    @staticmethod
    def _dequantize(values: list[ConstantValue], attributes: dict[str, Any]) -> np.ndarray:
        if len(values) not in (2, 3):
            raise ConstantEvaluationError("DequantizeLinear の入力数が正しくありません。")
        quantized, scale = values[0].array, values[1].array
        if quantized.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            raise ConstantEvaluationError("畳み込み重みの uint8 / int8 だけを展開します。")
        if scale.dtype != np.float32 or attributes.get("output_dtype", TensorProto.FLOAT) != TensorProto.FLOAT:
            raise ConstantEvaluationError("復号結果が float32 の重みだけを展開します。")
        if attributes.get("block_size", 0) != 0 or scale.ndim > 1:
            raise ConstantEvaluationError("対応する尺度はスカラーまたは 1 次元の軸別尺度です。")
        if not scale.size or not np.isfinite(scale).all():
            raise ConstantEvaluationError("尺度が空、または有限値ではありません。")
        zero = values[2].array if len(values) == 3 else np.asarray(0, dtype=quantized.dtype)
        if zero.dtype != quantized.dtype:
            raise ConstantEvaluationError("ゼロ点と量子化された重みの型が一致しません。")
        if scale.size == 1:
            if zero.size != 1:
                raise ConstantEvaluationError("スカラー尺度のゼロ点は 1 要素が必要です。")
            multiplier = scale.reshape(())
            offset = zero.astype(np.float32).reshape(())
        else:
            axis = int(attributes.get("axis", 1))
            if not -quantized.ndim <= axis < quantized.ndim:
                raise ConstantEvaluationError("復号する軸が重みの次元の範囲外です。")
            axis %= quantized.ndim
            if scale.shape[0] != quantized.shape[axis]:
                raise ConstantEvaluationError("軸別尺度の長さと重みの軸長が一致しません。")
            broadcast_shape = [1] * quantized.ndim
            broadcast_shape[axis] = scale.shape[0]
            multiplier = scale.reshape(broadcast_shape)
            if len(values) == 3:
                if zero.shape != scale.shape:
                    raise ConstantEvaluationError("軸別の尺度とゼロ点の形状が一致しません。")
                offset = zero.astype(np.float32).reshape(broadcast_shape)
            else:
                offset = np.asarray(0, dtype=np.float32)
        # 中間演算も float32 に固定し、元の DequantizeLinear と同じ数式を使います。
        difference = np.subtract(quantized.astype(np.float32), offset, dtype=np.float32)
        return np.multiply(difference, multiplier, dtype=np.float32)

    @staticmethod
    def _reshape(values: list[ConstantValue], attributes: dict[str, Any]) -> np.ndarray:
        if len(values) != 2:
            raise ConstantEvaluationError("Reshape の入力数が正しくありません。")
        source, shape_array = values[0].array, values[1].array
        if shape_array.ndim != 1 or shape_array.dtype != np.int64:
            raise ConstantEvaluationError("Reshape の形状は 1 次元の int64 定数が必要です。")
        shape = [int(value) for value in shape_array]
        allowzero = int(attributes.get("allowzero", 0))
        if allowzero not in (0, 1) or shape.count(-1) > 1 or any(value < -1 for value in shape):
            raise ConstantEvaluationError("Reshape の形状指定が正しくありません。")
        if allowzero and 0 in shape and -1 in shape:
            raise ConstantEvaluationError("allowzero=1 では 0 と -1 を同時に指定できません。")
        if not allowzero:
            for index, value in enumerate(shape):
                if value == 0:
                    if index >= source.ndim:
                        raise ConstantEvaluationError("元の次元がない軸を Reshape の 0 で参照しています。")
                    shape[index] = source.shape[index]
        try:
            return np.reshape(source, shape)
        except ValueError as exc:
            raise ConstantEvaluationError(f"Reshape を計算できません: {exc}") from exc


def _subgraph_references(graph: onnx.GraphProto) -> set[str]:
    """子グラフから参照される名前を保護します。過剰な削除を避けるため保守的です。"""
    names: set[str] = set()
    for node in graph.node:
        for attribute in node.attribute:
            children = []
            if attribute.type == onnx.AttributeProto.GRAPH:
                children = [attribute.g]
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                children = list(attribute.graphs)
            for child in children:
                names.update(item for subnode in child.node for item in subnode.input if item)
                names.update(item.name for item in child.output)
                names.update(_subgraph_references(child))
    return names


def _prune_replaced_chains(
    graph: onnx.GraphProto, candidate_nodes: set[int], candidate_initializers: set[str]
) -> dict[str, int]:
    """置換した定数経路のうち、参照がなくなった部分だけを取り除きます。"""
    protected = {item.name for item in (*graph.input, *graph.output)} | _subgraph_references(graph)
    references = Counter(name for node in graph.node for name in node.input if name)
    removed: set[int] = set()
    removed_values: set[str] = set()
    # ONNX のノードは依存順に並ぶため、逆順なら利用側から元の定数まで確認できます。
    for index in sorted(candidate_nodes, reverse=True):
        node = graph.node[index]
        if any(name in protected or references[name] > 0 for name in node.output):
            continue
        removed.add(index)
        removed_values.update(node.output)
        for name in node.input:
            if name:
                references[name] -= 1
    retained_nodes = [node for index, node in enumerate(graph.node) if index not in removed]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    unused_initializers = {
        name for name in candidate_initializers if references[name] <= 0 and name not in protected
    }
    retained_initializers = [item for item in graph.initializer if item.name not in unused_initializers]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    removed_values.update(unused_initializers)
    retained_info = [item for item in graph.value_info if item.name not in removed_values or item.name in protected]
    del graph.value_info[:]
    graph.value_info.extend(retained_info)
    return {"removed_constant_nodes": len(removed), "removed_initializers": len(unused_initializers)}


def prepare_cpu_model(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """ONNX の保存内容を変更せず、CPU に渡すモデルのバイト列と処理記録を返します。

    対象は最上位グラフの Conv / ConvTranspose の定数重みです。入力が
    DequantizeLinear から直接渡される畳み込みは、整数演算への融合を保つため
    変更しません。実行時に上書き可能なグラフ入力も定数として扱いません。
    """
    started = time.perf_counter()
    source_path = Path(path).expanduser().resolve(strict=True)
    source_file_bytes = source_path.stat().st_size
    model = onnx.load_model(str(source_path), load_external_data=True)
    # 外部データがある場合も、返すバイト列だけでセッションを作成できるようにします。
    onnx.external_data_helper.convert_model_from_external_data(model)
    graph = model.graph
    evaluator = ConstantEvaluator(graph)
    graph_inputs = {item.name for item in graph.input}
    all_names = (
        {item.name for item in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
        | {name for node in graph.node for name in (*node.input, *node.output) if name}
    )
    replacements: dict[str, str] = {}
    restored: list[dict[str, Any]] = []
    candidate_nodes: set[int] = set()
    candidate_initializers: set[str] = set()
    skipped = Counter()
    unsupported = []
    rewritten_node_count = 0
    restored_float_bytes = 0
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type not in ("Conv", "ConvTranspose"):
            continue
        if len(node.input) < 2:
            skipped["missing_weight"] += 1
            continue
        activation_producer = evaluator.producers.get(node.input[0])
        if activation_producer is not None and activation_producer[1].op_type == "DequantizeLinear":
            skipped["activation_dequantization_preserved"] += 1
            continue
        weight_name = node.input[1]
        initializer = evaluator.initializers.get(weight_name)
        if initializer is not None and initializer.data_type == TensorProto.FLOAT and weight_name not in graph_inputs:
            skipped["already_float32_initializer"] += 1
            continue
        if weight_name in replacements:
            node.input[1] = replacements[weight_name]
            rewritten_node_count += 1
            continue
        try:
            value = evaluator.evaluate(weight_name)
            if not value.contains_dequantization:
                skipped["no_dequantization_in_weight_chain"] += 1
                continue
            if value.array.dtype != np.float32 or value.array.ndim < 3 or any(size < 1 for size in value.array.shape):
                raise ConstantEvaluationError("復号した重みは空でない float32 の畳み込みテンソルが必要です。")
            if not np.isfinite(value.array).all():
                raise ConstantEvaluationError("復号した重みに NaN または無限大が含まれています。")
        except (ConstantEvaluationError, ValueError, TypeError) as exc:
            skipped["unsupported_or_nonconstant_weight"] += 1
            unsupported.append({"node": node.name, "weight": weight_name, "reason": str(exc)})
            continue
        new_name = f"{weight_name}__cpu_fp32"
        suffix = 1
        while new_name in all_names:
            new_name = f"{weight_name}__cpu_fp32_{suffix}"
            suffix += 1
        all_names.add(new_name)
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value.array), name=new_name))
        node.input[1] = new_name
        replacements[weight_name] = new_name
        rewritten_node_count += 1
        restored_float_bytes += value.array.nbytes
        candidate_nodes.update(value.nodes)
        candidate_initializers.update(value.initializers)
        restored.append({
            "original_weight_value": weight_name,
            "new_initializer": new_name,
            "shape": list(value.array.shape),
            "dtype": "float32",
            "bytes": value.array.nbytes,
        })

    pruning = _prune_replaced_chains(graph, candidate_nodes, candidate_initializers)
    prepared_bytes = model.SerializeToString()
    report = {
        "enabled": True,
        "source_file": str(source_path),
        "on_disk_model_unchanged": True,
        "disk_model_file_bytes": source_file_bytes,
        "in_memory_model_bytes": len(prepared_bytes),
        "restored_weight_count": len(restored),
        "rewritten_conv_node_count": rewritten_node_count,
        "restored_float_weight_bytes": restored_float_bytes,
        "preparation_seconds": time.perf_counter() - started,
        "skipped_counts": dict(skipped),
        "unsupported_weights": unsupported,
        "restored_weights": restored,
        **pruning,
        "scope": (
            "保存済みの重みを同じ尺度・ゼロ点で float32 に戻し、セッション作成時だけ定数化します。"
            "新しい量子化は行いません。MatMul、入力復号直後の畳み込み、グラフ入出力は保持します。"
            "in_memory_model_bytes は渡す ONNX バイト列の容量で、実行時の常駐メモリとは別です。"
        ),
    }
    return prepared_bytes, report
