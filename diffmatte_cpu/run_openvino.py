#!/usr/bin/env python3
"""同じ圧縮 ONNX を OpenVINO の CPU デバイスで実行します。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .run_diffmatte import RSSMonitor, make_feeds, process_peak_rss, read_source_inputs, write_json

HEAD_PRECISION_SOURCE = {
    'tested_openvino_release': '2026.4.1',
    'source_commit': 'e213a14725769a6a2bf86fc86650aa78a9c05d3e',
    'mechanism': 'version-pinned internal runtime attribute; normalized by serialize/read',
    'sources': [
        'https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/bindings/python/tests/test_runtime/test_model.py',
        'https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/core/src/pass/serialize.cpp',
        'https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/common/transformations/src/transformations/fp16_compression/convert_legacy_precision_attribute.cpp',
        'https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_cpu/src/node.cpp',
        'https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_cpu/src/graph.cpp',
    ],
}


def select_matting_head_nodes(model) -> tuple[list, list[dict]]:
    """重み名で head の最後の Conv を特定し、2 Conv と間の処理を辿ります。

    ONNX の bias 付き Conv は OpenVINO で Convolution と Add に分かれます。
    DDIM で展開された同じ head の全呼び出しを扱い、復号器本体には遡りません。
    """
    ordered = model.get_ordered_ops()
    names = [node.get_friendly_name() for node in ordered]
    if len(names) != len(set(names)):
        raise ValueError('head 精度保持には一意な OpenVINO friendly_name が必要です。')
    convolutions = {'Convolution', 'GroupConvolution'}
    backward_ops = {'Add', 'Relu', 'Convert'}
    forward_ops = {'Add', 'Relu', 'Convert', 'Sigmoid'}
    selected_names: set[str] = set()
    heads = []
    for last in ordered:
        if last.get_type_name() not in convolutions or last.get_input_size() < 2:
            continue
        weight = last.input_value(1)
        weight_names = {weight.get_node().get_friendly_name(), *weight.get_names()}
        if not any('matting_head' in name for name in weight_names):
            continue
        output_shape = last.get_output_partial_shape(0)
        if output_shape.is_dynamic or len(output_shape) != 4 or output_shape[1].get_length() != 1:
            continue
        local_names = {last.get_friendly_name()}
        current = last.input_value(0).get_node()
        previous_conv = None
        for _ in range(12):
            current_name, current_type = current.get_friendly_name(), current.get_type_name()
            if current_name in local_names:
                raise ValueError(f'head の探索で循環を検出しました: {current_name}')
            local_names.add(current_name)
            if current_type in convolutions:
                previous_conv = current_name
                break
            if current_type not in backward_ops:
                raise ValueError(f'head の手前の Conv を特定できません: {current_name} ({current_type})')
            candidates = [value.get_node() for value in current.input_values()
                          if value.get_node().get_type_name() in convolutions | backward_ops]
            if len(candidates) != 1:
                raise ValueError(f'head の入力経路を一意に特定できません: {current_name}')
            current = candidates[0]
        if previous_conv is None:
            raise ValueError(f'head の手前の Conv が見つかりません: {last.get_friendly_name()}')

        current = last
        sigmoid_name = None
        for _ in range(12):
            candidates = {port.get_node().get_friendly_name(): port.get_node()
                          for value in current.outputs() for port in value.get_target_inputs()
                          if port.get_node().get_type_name() in forward_ops}
            if len(candidates) != 1:
                raise ValueError(f'head の Sigmoid までの経路を一意に特定できません: {current.get_friendly_name()}')
            current = next(iter(candidates.values()))
            current_name = current.get_friendly_name()
            if current_name in local_names:
                raise ValueError(f'head の探索で循環を検出しました: {current_name}')
            local_names.add(current_name)
            if current.get_type_name() == 'Sigmoid':
                sigmoid_name = current_name
                break
        if sigmoid_name is None:
            raise ValueError(f'head の Sigmoid が見つかりません: {last.get_friendly_name()}')
        if selected_names & local_names:
            raise ValueError('異なる matting head の対象ノードが重複しています。')
        selected_names.update(local_names)
        heads.append({'first_convolution': previous_conv, 'last_convolution': last.get_friendly_name(),
                      'sigmoid': sigmoid_name, 'weight_names': sorted(weight_names),
                      'node_names': sorted(local_names)})
    if not heads:
        raise ValueError('matting_head の対象が 0 件です。未確認のモデルには精度保持を適用できません。')
    selected = [node for node in ordered if node.get_friendly_name() in selected_names]
    return selected, heads


def prepare_head_fp32(core, model) -> tuple[object, dict]:
    """元の ONNX を変更せず、一時 IR 経由で head の精度保持属性を正規化します。"""
    import openvino as ov

    if ov.__version__.split('-')[0] != HEAD_PRECISION_SOURCE['tested_openvino_release']:
        raise ValueError('--keep-head-fp32 は内部属性を使うため、確認済み OpenVINO 2026.4.1 が必要です。')
    started = time.perf_counter()
    selected, heads = select_matting_head_nodes(model)
    target_nodes = [{'name': node.get_friendly_name(), 'type': node.get_type_name()}
                    for node in selected]
    for node in selected:
        if any(output.element_type != ov.Type.f32 for output in node.outputs()):
            raise ValueError(f'head の元の型が FP32 ではありません: {node.get_friendly_name()}')
        node.get_rt_info()['precise_0'] = ''
    with tempfile.TemporaryDirectory(prefix='diffmatte_head_precision_') as directory:
        xml, weights = Path(directory) / 'model.xml', Path(directory) / 'model.bin'
        step_start = time.perf_counter()
        # save_model の既定 FP16 重み圧縮を避け、元の型を保つ serialize を使います。
        ov.serialize(model, str(xml), str(weights))
        serialize_seconds = time.perf_counter() - step_start
        temporary_ir_bytes = xml.stat().st_size + weights.stat().st_size
        step_start = time.perf_counter()
        # bytes 入力なら一時ファイルの寿命や OS のファイルロックに依存しません。
        prepared = core.read_model(model=xml.read_bytes(), weights=weights.read_bytes())
        readback_seconds = time.perf_counter() - step_start
    prepared_names = {node.get_friendly_name(): node for node in prepared.get_ordered_ops()}
    missing_markers = [item['name'] for item in target_nodes
                       if item['name'] not in prepared_names
                       or 'DisablePrecisionConversion_0' not in prepared_names[item['name']].get_rt_info()]
    if missing_markers:
        raise RuntimeError(f'head 精度保持属性の正規化に失敗しました: {missing_markers}')
    report = {
        'enabled': True, 'on_disk_onnx_unchanged': True,
        'scope': 'all unrolled calls of the two-convolution matting head, including intervening Add/Relu and Sigmoid',
        'head_call_count': len(heads), 'target_node_count': len(target_nodes),
        'target_type_counts': dict(Counter(item['type'] for item in target_nodes)),
        'target_nodes': target_nodes, 'heads': heads,
        'legacy_marker': 'precise_0', 'normalized_marker': 'DisablePrecisionConversion_0',
        'serialize_seconds': serialize_seconds, 'readback_seconds': readback_seconds,
        'preparation_seconds': time.perf_counter() - started,
        'temporary_ir_bytes': temporary_ir_bytes, 'source_evidence': HEAD_PRECISION_SOURCE,
    }
    return prepared, report


def verify_head_runtime_precision(compiled, head_report: dict) -> None:
    """融合後の元ノード名まで照合し、精度保持が無視された場合は推論前に止めます。"""
    target_names = {item['name'] for item in head_report['target_nodes']}
    mapping = {name: [] for name in target_names}
    rows = []
    for node in compiled.get_runtime_model().get_ordered_ops():
        info = node.get_rt_info()
        row = {'name': node.get_friendly_name()}
        for key in ('layerType', 'originalLayersNames', 'runtimePrecision', 'primitiveType', 'outputPrecisions'):
            if key in info:
                row[key] = str(info[key].value)
        original_names = {name.strip() for name in row.get('originalLayersNames', '').split(',') if name.strip()}
        matched = (original_names | {row['name']}) & target_names
        if not matched:
            continue
        if row.get('layerType') in {'Input', 'Output', 'Const', 'Reorder'}:
            continue
        row['matched_target_nodes'] = sorted(matched)
        rows.append(row)
        for name in matched:
            mapping[name].append(row['name'])
    missing = sorted(name for name, matches in mapping.items() if not matches)
    non_fp32 = [row for row in rows if row.get('runtimePrecision', '').lower() not in {'f32', 'fp32'}
                or any(value.strip().lower() not in {'f32', 'fp32'}
                       for value in row.get('outputPrecisions', '').split(','))]
    head_report['runtime_nodes'] = rows
    head_report['target_to_runtime_nodes'] = mapping
    head_report['missing_runtime_targets'] = missing
    head_report['non_fp32_runtime_nodes'] = non_fp32
    head_report['runtime_verification_passed'] = not missing and not non_fp32 and bool(rows)
    if not head_report['runtime_verification_passed']:
        raise RuntimeError('head の FP32 実行を確認できません。内部属性が保持されなかった可能性があります。'
                           f' 未対応ノード={missing}; FP32 以外={json.dumps(non_fp32, ensure_ascii=False)}')


def run(args: argparse.Namespace) -> dict:
    import openvino as ov

    if args.threads < 1 or args.runs < 1 or args.warmup < 0 or not 0 <= args.seed < 2**64:
        raise ValueError('threads/runs は正、warmup は非負、seed は uint64 の範囲で指定してください。')
    model_path = args.model.expanduser().resolve(strict=True)
    image_path = args.image.expanduser().resolve(strict=True)
    trimap_path = args.trimap.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    for filename in ('alpha.npy', 'alpha16.png', 'timing.json'):
        if output_dir / filename in (model_path, image_path, trimap_path):
            raise ValueError('出力先が入力ファイルと重なっています。')
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = RSSMonitor()
    monitor.start()
    pipeline_start = time.perf_counter()
    try:
        started = time.perf_counter()
        core = ov.Core()
        model = core.read_model(str(model_path))
        model_read_seconds = time.perf_counter() - started
        head_report = {'enabled': False}
        if getattr(args, 'keep_head_fp32', False):
            model, head_report = prepare_head_fp32(core, model)
        inputs = {item.get_any_name(): item for item in model.inputs}
        expected_channels = {'image': 3, 'trimap': 1, 'initial_noise': 1}
        if set(inputs) != set(expected_channels):
            raise ValueError(f'入力名が一致しません: {list(inputs)}')
        if any(item.partial_shape.is_dynamic for item in inputs.values()):
            raise ValueError('固定寸法の ONNX が必要です。')
        image_shape = list(inputs['image'].shape)
        if len(image_shape) != 4:
            raise ValueError('NCHW 画像が必要です。')
        height, width = image_shape[-2:]
        for name, channels in expected_channels.items():
            item = inputs[name]
            if list(item.shape) != [1, channels, height, width] or item.element_type != ov.Type.f32:
                raise ValueError(f'入力 {name} の寸法または型が異なります。')
        if len(model.outputs) != 1 or 'alpha' not in model.output(0).get_names():
            raise ValueError('単一の alpha 出力が必要です。')
        if model.output(0).element_type != ov.Type.f32:
            raise ValueError('alpha 出力は float32 が必要です。')

        properties = {
            'PERFORMANCE_HINT': 'LATENCY',
            'INFERENCE_NUM_THREADS': args.threads,
            'NUM_STREAMS': 1,
            'ENABLE_CPU_PINNING': False,
            'ENABLE_HYPER_THREADING': True,
            'INFERENCE_PRECISION_HINT': ov.Type.f32 if args.precision == 'f32' else ov.Type.bf16,
            'EXECUTION_MODE_HINT': 'ACCURACY' if args.precision == 'f32' else 'PERFORMANCE',
        }
        started = time.perf_counter()
        compiled = core.compile_model(model, 'CPU', properties)
        compile_seconds = time.perf_counter() - started
        if head_report['enabled']:
            verify_head_runtime_precision(compiled, head_report)
        request = compiled.create_infer_request()
        started = time.perf_counter()
        rgb, tri = read_source_inputs(image_path, trimap_path)
        feeds, padding = make_feeds(rgb, tri, (height, width), args.seed, args.pad_to_model)
        if any(not np.isfinite(value).all() for value in feeds.values()):
            raise ValueError('入力に NaN または無限大があります。')
        tensors = {name: ov.Tensor(value, shared_memory=True) for name, value in feeds.items()}
        for name, tensor in tensors.items():
            request.set_tensor(name, tensor)
        input_seconds = time.perf_counter() - started
        warmup, measured = [], []
        for destination, count in ((warmup, args.warmup), (measured, args.runs)):
            for _ in range(count):
                started = time.perf_counter()
                request.infer()
                destination.append(time.perf_counter() - started)
                monitor.sample()
        raw = request.get_tensor('alpha').data
        if raw.shape != (1, 1, height, width) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError('alpha の寸法、型、または数値が不正です。')
        alpha = raw[0, 0, :rgb.shape[-2], :rgb.shape[-1]].copy()
        np.save(output_dir / 'alpha.npy', alpha, allow_pickle=False)
        Image.fromarray(np.rint(np.clip(alpha, 0, 1) * 65535).astype(np.uint16)).save(output_dir / 'alpha16.png')

        execution_counts = {key: Counter() for key in ('layerType', 'runtimePrecision', 'primitiveType')}
        precision_by_layer = Counter()
        for node in compiled.get_runtime_model().get_ordered_ops():
            info = node.get_rt_info()
            for key, counts in execution_counts.items():
                if key in info:
                    counts[str(info[key].value)] += 1
            if 'layerType' in info and 'runtimePrecision' in info:
                precision_by_layer[str(info['layerType'].value) + '/' + str(info['runtimePrecision'].value)] += 1
        # CPU が BF16 命令を持たない場合、OpenVINO は BF16 指定を黙って FP32 に戻します。
        bf16_runtime_nodes = execution_counts['runtimePrecision'].get('bf16', 0)
        if args.precision == 'bf16' and not bf16_runtime_nodes:
            print('警告: BF16 を指定しましたが、BF16 で実行されたノードは 0 件です（FP32 で実行）。'
                  f" CPU の OPTIMIZATION_CAPABILITIES={core.get_property('CPU', 'OPTIMIZATION_CAPABILITIES')}",
                  file=sys.stderr, flush=True)
        effective = {}
        for key in ('EXECUTION_DEVICES', 'INFERENCE_NUM_THREADS', 'NUM_STREAMS', 'INFERENCE_PRECISION_HINT', 'EXECUTION_MODE_HINT'):
            effective[key] = str(compiled.get_property(key))
        report = {
            'schema_version': 1,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'model': str(model_path),
            'model_file_bytes': model_path.stat().st_size,
            'runtime': {
                'engine': 'OpenVINO', 'openvino_version': ov.__version__,
                'python_version': platform.python_version(), 'numpy_version': np.__version__,
                'requested_device': 'CPU', 'device_name': core.get_property('CPU', 'FULL_DEVICE_NAME'),
                'requested_precision': args.precision, 'requested_threads': args.threads,
                'hyper_threading_enabled': True, 'cpu_pinning_enabled': False,
                'effective_properties': effective,
                'runtime_graph_counts': {key: dict(value) for key, value in execution_counts.items()},
                'runtime_precision_by_layer': dict(precision_by_layer),
                'optimization_capabilities': list(core.get_property('CPU', 'OPTIMIZATION_CAPABILITIES')),
                'bf16_runtime_node_count': bf16_runtime_nodes,
                'head_precision': head_report,
                'input_binding': 'shared float32 host tensors, bound once before timing',
            },
            'input': {
                'image': str(image_path), 'trimap': str(trimap_path),
                'seed': args.seed, 'same_noise_for_all_runs': True,
                'feeds_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in feeds.items()},
                'shapes': {name: list(value.shape) for name, value in feeds.items()}, 'padding': padding,
            },
            'timing_seconds': {
                'model_read': model_read_seconds, 'compile': compile_seconds,
                'input_preparation_and_binding': input_seconds,
                'warmup': warmup, 'measured': measured,
                'mean': statistics.mean(measured), 'median': statistics.median(measured),
                'minimum': min(measured), 'maximum': max(measured),
                'population_std': statistics.pstdev(measured),
                'pipeline_elapsed_before_report_write': time.perf_counter() - pipeline_start,
            },
            'memory': {'sampled_rss_peak_bytes': monitor.peak_bytes,
                       'process_lifetime_peak_rss': process_peak_rss(),
                       'scope': 'Python process RSS; includes model, compiler, tensors, and runtime'},
            'ground_truth_used': False,
        }
        if args.reference_alpha:
            from .benchmark_models import difference_metrics
            reference = np.load(args.reference_alpha, allow_pickle=False)
            if reference.ndim == 4:
                reference = reference[0, 0, :alpha.shape[0], :alpha.shape[1]]
            report['consistency_against_reference'] = {
                'reference_alpha': str(args.reference_alpha.resolve()),
                'full_image': difference_metrics(reference, alpha),
                'unknown_region': difference_metrics(reference, alpha, tri[0, 0] == 0.5),
            }
        write_json(output_dir / 'timing.json', report)
        print(json.dumps({'output_dir': str(output_dir), 'precision': args.precision,
                          'bf16_runtime_nodes': bf16_runtime_nodes,
                          'median_seconds': report['timing_seconds']['median'],
                          'unknown_mae': report.get('consistency_against_reference', {}).get('unknown_region', {}).get('mae')}, ensure_ascii=False), flush=True)
        return report
    finally:
        monitor.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--trimap', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--precision', choices=['f32', 'bf16'], default='f32')
    parser.add_argument('--keep-head-fp32', action='store_true',
                        help='OpenVINO 2026.4.1 の内部属性で matting head を FP32 に保ち、実行精度を検証します。')
    parser.add_argument('--seed', type=int, default=8282)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--pad-to-model', action='store_true')
    parser.add_argument('--reference-alpha', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
