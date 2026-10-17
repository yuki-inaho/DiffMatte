"""ConvRot の正則回転と UINT8 量子化を固定形状 DiffMatte ONNX に適用します。

公式 CUDA カーネルの移植ではありません。ONNX 標準演算と ONNX Runtime の
CPU 量子化を使用します。元モデルとの出力一致度は benchmark_models.py で
検証してください。正解に対する画像精度を測るには、別途正解アルファが必要です。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def deduplicate_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """同一の定数を共有し、未参照の定数を削除します。入出力は変更しません。"""
    protected = {x.name for x in list(model.graph.input) + list(model.graph.output)}
    canonical, rename, unique = {}, {}, []
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(array.tobytes()).digest())
        if key in canonical and tensor.name not in protected:
            rename[tensor.name] = canonical[key]
        else:
            canonical.setdefault(key, tensor.name)
            unique.append(tensor)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = rename.get(name, name)
    used = protected | {name for node in model.graph.node for name in node.input}
    del model.graph.initializer[:]
    model.graph.initializer.extend(x for x in unique if x.name in used)
    # Initializer aliases can leave stale value_info entries after graph rewrites.
    produced = used | {name for node in model.graph.node for name in node.output}
    retained = [x for x in model.graph.value_info if x.name in produced and x.name not in rename]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    return model


def split_gemm(model: onnx.ModelProto) -> onnx.ModelProto:
    """定数重みの Gemm を意味を保って MatMul と Add に展開します。"""
    weights = {x.name: x for x in model.graph.initializer}
    nodes = []
    for node in model.graph.node:
        if node.op_type != 'Gemm':
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.input[1] not in weights:
            raise ValueError(f'動的 Gemm 重みは対応外です: {node.name}')
        if attrs.get('alpha', 1.) != 1. or attrs.get('beta', 1.) != 1.:
            raise ValueError(f'alpha/beta が 1 でない Gemm は対応外です: {node.name}')
        a, b = node.input[:2]
        if attrs.get('transA', 0):
            transposed = node.name + '/A_transposed'
            nodes.append(helper.make_node('Transpose', [a], [transposed], perm=[1, 0], name=transposed))
            a = transposed
        if attrs.get('transB', 0):
            b = node.name + '/B_transposed'
            model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(weights[node.input[1]]).T.copy(), b))
        has_bias = len(node.input) > 2 and bool(node.input[2])
        product = node.name + '/product' if has_bias else node.output[0]
        nodes.append(helper.make_node('MatMul', [a, b], [product], name=node.name + '/MatMul'))
        if has_bias:
            nodes.append(helper.make_node('Add', [product, node.input[2]], list(node.output), name=node.name + '/Bias'))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return deduplicate_initializers(model)


def inventory(model: onnx.ModelProto) -> dict:
    sizes = Counter()
    for tensor in model.graph.initializer:
        sizes[TensorProto.DataType.Name(tensor.data_type)] += numpy_helper.to_array(tensor).nbytes
    return {'nodes': len(model.graph.node), 'initializer_count': len(model.graph.initializer),
            'initializer_bytes_by_type': dict(sizes),
            'operators': dict(Counter(node.op_type for node in model.graph.node))}


def compact_uniform_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """簡略化時に展開された大きな一様定数を、値を変えず小さく保存します。"""
    nodes, retained = [], []
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        if tensor.data_type == TensorProto.FLOAT and array.size >= 65536 and np.all(array == array.flat[0]):
            shape_name = tensor.name + '/compact_shape'
            retained.append(numpy_helper.from_array(np.asarray(array.shape, dtype=np.int64), shape_name))
            value = numpy_helper.from_array(np.asarray([array.flat[0]], dtype=np.float32))
            nodes.append(helper.make_node('ConstantOfShape', [shape_name], [tensor.name],
                                          value=value, name=tensor.name + '/ConstantOfShape'))
        else:
            retained.append(tensor)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    existing = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes + existing)
    return model


def prepare_model(input_path: Path, output_path: Path, threads: int = 4) -> dict:
    """onnxsim、ORT BASIC、定数共有を実施します。数値検証は別に実行します。"""
    import onnxsim
    original = onnx.load(str(input_path))
    simplified, structural_ok = onnxsim.simplify(original, check_n=0)
    if not structural_ok:
        raise RuntimeError('onnxsim の構造検証に失敗しました。')
    with tempfile.TemporaryDirectory(prefix='diffmatte_prepare_', dir=output_path.parent) as directory:
        temporary = Path(directory) / 'basic.onnx'
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(temporary)
        # Keep initializer-backed linear layers available to ConvRot.
        session = ort.InferenceSession(simplified.SerializeToString(), options,
                                       providers=['CPUExecutionProvider'],
                                       disabled_optimizers={'MatMulAddFusion'})
        del session
        prepared = split_gemm(onnx.load(str(temporary)))
    onnx.checker.check_model(prepared)
    onnx.save(prepared, str(output_path))
    return {'onnxsim_version': onnxsim.__version__, 'onnxruntime_version': ort.__version__,
            'onnxsim_random_comparisons': 0, 'requires_separate_numerical_validation': True,
            'original': inventory(original), 'simplified': inventory(simplified),
            'prepared': inventory(prepared)}


def quantize_array_uint8(array: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """出力チャンネルごとにゼロを表現できる非対称 UINT8 へ丸めます。"""
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('量子化対象は有限な float32 配列である必要があります。')
    axis %= array.ndim
    reduction = tuple(index for index in range(array.ndim) if index != axis)
    low = np.minimum(array.min(axis=reduction), np.float32(0))
    high = np.maximum(array.max(axis=reduction), np.float32(0))
    scale = ((high - low) / np.float32(255)).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1)).astype(np.float32)
    zero = np.clip(np.rint(-low / scale), 0, 255).astype(np.uint8)
    shape = [1] * array.ndim
    shape[axis] = array.shape[axis]
    quantized = np.clip(np.rint(array / scale.reshape(shape)) + zero.reshape(shape), 0, 255).astype(np.uint8)
    return quantized, scale, zero


def protected_head_weights(model: onnx.ModelProto) -> set[str]:
    """最終アルファ出力頭の畳み込みを、融合後の重み名でも識別します。"""
    producers = {name: node for node in model.graph.node for name in node.output}
    protected = set()
    for node in model.graph.node:
        if node.op_type != 'Conv' or 'matting_head' not in node.input[1]:
            continue
        protected.add(node.input[1])
        previous = producers.get(node.input[0])
        while previous is not None and previous.op_type in ('Relu', 'Clip', 'Identity'):
            previous = producers.get(previous.input[0])
        if previous is not None and previous.op_type == 'Conv':
            protected.add(previous.input[1])
    return protected


def compress_remaining_weights(model: onnx.ModelProto, include_linear: bool = True,
                               include_relative_position: bool = True,
                               weight_group_size: int = 0) -> tuple[onnx.ModelProto, list[dict]]:
    """整数演算に置換しない重みを UINT8 保存し、浮動小数演算用に復元します。"""
    model = copy.deepcopy(model)
    initializers = {x.name: x for x in model.graph.initializer}
    candidates: dict[str, int] = {}
    convolution_types = {}
    protected = protected_head_weights(model)
    for node in model.graph.node:
        if node.name.startswith('ConvRot/rotation/'):
            continue
        if node.op_type in ('Conv', 'ConvTranspose', 'MatMul') and len(node.input) > 1:
            name = node.input[1]
            if name not in initializers or name in protected:
                continue
            if node.op_type == 'MatMul' and not include_linear:
                continue
            candidates[name] = 0 if node.op_type == 'Conv' else 1
            if node.op_type in ('Conv', 'ConvTranspose'):
                convolution_types[name] = node.op_type
        elif include_relative_position and node.op_type == 'Einsum':
            for name in node.input:
                if name in initializers and len(initializers[name].dims) >= 2:
                    candidates[name] = len(initializers[name].dims) - 1
    dq_nodes, new_initializers, reports = [], [], []
    quantized_names = set()
    for name, axis in candidates.items():
        tensor = initializers[name]
        if tensor.data_type != TensorProto.FLOAT or np.prod(tensor.dims) < 64:
            continue
        array = numpy_helper.to_array(tensor)
        if weight_group_size and name in convolution_types:
            # Small blocks improve fidelity for the repeatedly used decoder weights.
            is_transposed = convolution_types[name] == 'ConvTranspose'
            canonical = array.transpose(1, 0, 2, 3).copy() if is_transposed else array
            features = int(np.prod(canonical.shape[1:]))
            if features % weight_group_size == 0:
                blocked = canonical.reshape(-1, weight_group_size)
                quantized, scale, zero = quantize_array_uint8(blocked, 0)
                qname, sname, zname = name + '/uint8_groups', name + '/group_scale', name + '/group_zero_point'
                shape_name = name + '/canonical_shape'
                new_initializers.extend([numpy_helper.from_array(quantized, qname),
                                         numpy_helper.from_array(scale, sname),
                                         numpy_helper.from_array(zero, zname),
                                         numpy_helper.from_array(np.asarray(canonical.shape, dtype=np.int64), shape_name)])
                dequantized_name = name + '/dequantized_groups'
                canonical_name = name + '/canonical_restored' if is_transposed else name
                dq_nodes.append(helper.make_node('DequantizeLinear', [qname, sname, zname],
                                                [dequantized_name], axis=0, name=name + '/GroupDequantizeLinear'))
                dq_nodes.append(helper.make_node('Reshape', [dequantized_name, shape_name],
                                                [canonical_name], name=name + '/RestoreWeightShape'))
                if is_transposed:
                    dq_nodes.append(helper.make_node('Transpose', [canonical_name], [name],
                                                    perm=[1, 0, 2, 3], name=name + '/RestoreWeightLayout'))
                quantized_names.add(name)
                restored = (quantized.astype(np.float32) - zero[:, None]) * scale[:, None]
                reports.append({'name': name, 'shape': list(array.shape), 'weight_group_size': weight_group_size,
                                'original_bytes': array.nbytes,
                                'stored_bytes': quantized.nbytes + scale.nbytes + zero.nbytes,
                                'weight_mae': float(np.mean(np.abs(restored - blocked), dtype=np.float64))})
                continue
        quantized, scale, zero = quantize_array_uint8(array, axis)
        qname, sname, zname = name + '/uint8', name + '/scale', name + '/zero_point'
        new_initializers.extend([numpy_helper.from_array(quantized, qname),
                                 numpy_helper.from_array(scale, sname),
                                 numpy_helper.from_array(zero, zname)])
        dq_nodes.append(helper.make_node('DequantizeLinear', [qname, sname, zname], [name],
                                        axis=axis, name=name + '/DequantizeLinear'))
        quantized_names.add(name)
        broadcast = [1] * array.ndim
        broadcast[axis] = array.shape[axis]
        restored = (quantized.astype(np.float32) - zero.reshape(broadcast)) * scale.reshape(broadcast)
        reports.append({'name': name, 'shape': list(array.shape), 'axis': axis,
                        'original_bytes': array.nbytes,
                        'stored_bytes': quantized.nbytes + scale.nbytes + zero.nbytes,
                        'weight_mae': float(np.mean(np.abs(restored - array), dtype=np.float64))})
    retained = [x for x in model.graph.initializer if x.name not in quantized_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + new_initializers)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dq_nodes + original_nodes)
    return deduplicate_initializers(model), reports


def linear_node_names(model: onnx.ModelProto) -> list[str]:
    weights = {x.name for x in model.graph.initializer}
    return [node.name for node in model.graph.node if node.op_type == 'MatMul'
            and node.input[1] in weights and not node.name.startswith('ConvRot/rotation/')]


def dynamic_linears(model: onnx.ModelProto, destination: Path) -> onnx.ModelProto:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(model, str(destination), op_types_to_quantize=['MatMul'],
                     nodes_to_quantize=linear_node_names(model), per_channel=True,
                     reduce_range=False, weight_type=QuantType.QUInt8,
                     extra_options={'MatMulConstBOnly': True, 'WeightSymmetric': False})
    return onnx.load(str(destination))


class NPZCalibrationReader:
    """明示的な image/trimap/initial_noise を収めた NPZ を逐次供給します。"""
    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError('静的量子化には較正用 NPZ が必要です。')
        self.paths = paths
        self.rewind()

    def rewind(self):
        self.iterator = iter(self.paths)

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key], dtype=np.float32) for key in ('image', 'trimap', 'initial_noise')}


def static_convolutions(model: onnx.ModelProto, destination: Path, calibration: list[Path],
                        threads: int, cache_path: Path | None = None,
                        scope: str = 'all') -> onnx.ModelProto:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    from .calibrate_cpu import calibrate_minmax
    names = []
    protected = protected_head_weights(model)
    producers = {name: node for node in model.graph.node for name in node.output}
    for node in model.graph.node:
        if node.op_type == 'Conv' and node.input[1] not in protected and 'patch_embed' not in node.input[1]:
            is_phase = node.name.startswith('PhaseConv/')
            previous = producers.get(node.input[0])
            is_upsample = previous is not None and previous.op_type == 'Resize'
            selected = scope == 'all' or (scope == 'phase' and is_phase) or (scope == 'upsample' and is_upsample)
            selected |= scope == 'upsample-phase' and (is_phase or is_upsample)
            if selected:
                names.append(node.name)
    if not names:
        raise ValueError('指定した範囲に静的量子化できる畳み込みがありません。')
    if cache_path is None:
        # The helper runs an isolated CPU process with bounded memory usage.
        input_path = destination.with_suffix('.calibration_input.onnx')
        cache_path = destination.with_suffix('.calibration.json')
        onnx.save(model, str(input_path))
        calibrate_minmax(input_path, cache_path, calibration, op_types=['Conv'], threads=threads)
    elif not cache_path.is_file():
        raise ValueError(f'較正結果が存在しません: {cache_path}')
    quantize_static(model, str(destination), None,
                    quant_format=QuantFormat.QDQ, op_types_to_quantize=['Conv'],
                    nodes_to_quantize=names, per_channel=True, reduce_range=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    calibration_cache_path=str(cache_path),
                    extra_options={'ActivationSymmetric': False, 'WeightSymmetric': False})
    return onnx.load(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['prepare', 'rotation-fp32', 'weights', 'dynamic', 'static'], default='dynamic')
    parser.add_argument('--prepared', action='store_true', help='入力は onnxsim と ORT BASIC の処理済み')
    parser.add_argument('--no-rotation', action='store_true', help='回転なしの比較用候補を作成')
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--weight-group-size', type=int, default=0,
                        help='対応する畳み込み重みを小さな組ごとに量子化します。0 または割り切れない場合はチャンネル別です。')
    parser.add_argument('--rewrite-deconv', action='store_true',
                        help='対応する転置畳み込みを同値な4相の通常畳み込みへ置換します。')
    parser.add_argument('--static-scope', choices=['all', 'upsample-phase', 'upsample', 'phase'], default='all')
    parser.add_argument('--calibration-cache', type=Path,
                        help='同一の浮動小数グラフで取得済みの MinMax 結果を再利用します。')
    parser.add_argument('--linear-activation', choices=['uint8', 'fp32'], default='uint8')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--calibration', type=Path, nargs='*', default=[])
    parser.add_argument('--keep-relative-position-fp32', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads は 1 以上で指定してください。')
    if args.weight_group_size < 0:
        parser.error('--weight-group-size は 0 以上で指定してください。')
    if args.input.resolve() == args.output.resolve():
        parser.error('元モデルの上書きを避けるため、出力先を別に指定してください。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    effective_linear_activation = (
        args.linear_activation if args.variant in ('dynamic', 'static') else 'fp32'
    )
    report = {'input': str(args.input), 'output': str(args.output), 'variant': args.variant,
              'rotation_group_size': None if args.no_rotation else args.group_size,
              'weight_group_size': args.weight_group_size, 'rewrite_deconv': args.rewrite_deconv,
              'static_scope': args.static_scope, 'linear_activation': effective_linear_activation,
              'onnx_version': onnx.__version__, 'onnxruntime_version': ort.__version__}
    if args.variant == 'prepare':
        report['preparation'] = prepare_model(args.input, args.output, args.threads)
    else:
        with tempfile.TemporaryDirectory(prefix='diffmatte_convert_', dir=args.output.parent) as directory:
            directory = Path(directory)
            prepared = args.input
            if not args.prepared:
                prepared = directory / 'prepared.onnx'
                report['preparation'] = prepare_model(args.input, prepared, args.threads)
            model = split_gemm(onnx.load(str(prepared)))
            if not args.no_rotation:
                from .convrot import apply_convrot
                model, report['rotation'] = apply_convrot(model, group_size=args.group_size)
            if args.rewrite_deconv:
                from .deconv_to_conv import rewrite_convtranspose
                model, report['deconvolution_rewrite'] = rewrite_convtranspose(model)
            if args.variant == 'static':
                model = static_convolutions(model, directory / 'static.onnx', args.calibration, args.threads,
                                            args.calibration_cache, args.static_scope)
            if args.variant in ('dynamic', 'static') and args.linear_activation == 'uint8':
                model = dynamic_linears(model, directory / 'dynamic.onnx')
            if args.variant in ('weights', 'dynamic', 'static'):
                model, report['weight_storage'] = compress_remaining_weights(
                    model, include_linear=args.variant == 'weights' or args.linear_activation == 'fp32',
                    include_relative_position=not args.keep_relative_position_fp32,
                    weight_group_size=args.weight_group_size)
            model = compact_uniform_constants(deduplicate_initializers(model))
            report['inventory'] = inventory(model)
            metadata = {x.key: x.value for x in model.metadata_props}
            metadata.update({'convrot.paper': 'https://arxiv.org/abs/2512.03673',
                             'convrot.implementation': 'ONNX regular-Hadamard adaptation; not the official CUDA kernel',
                             'convrot.enabled': str(not args.no_rotation).lower(),
                             'convrot.variant': args.variant,
                             'convrot.group_size': str(0 if args.no_rotation else args.group_size),
                             'quantization.weight_storage': (
                                 'UINT8 with float32 scales; sensitive arithmetic remains float32'
                                 if args.variant in ('weights', 'dynamic', 'static')
                                 else 'FLOAT32; no weight quantization'),
                             'quantization.linear_activation': effective_linear_activation})
            helper.set_model_props(model, metadata)
            onnx.checker.check_model(model)
            onnx.save(model, str(args.output))
    report['output_bytes'] = args.output.stat().st_size
    args.output.with_suffix('.conversion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'bytes': report['output_bytes'], 'variant': args.variant}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
