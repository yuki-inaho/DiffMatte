"""CPU メモリを抑えて ONNX の MinMax 較正結果を保存します。

推論用出力の代わりに各対象テンソルの最小値・最大値だけを公開し、
それらの演算を生成元の直後へ配置します。さらに、最小値・最大値から
作るゼロを後続処理の Reshape の形状へ加える依存関係を作り、両方の
縮約が終わるまで元の利用先を待たせます。有限値ではデータ自体を一切
変更せず、中間値の同時保持を避けられます。NaN/Inf は較正対象外です。
較正は独立した子プロセスで実行するため、終了後に ONNX Runtime の
メモリも確実に解放されます。
保存した JSON は ORT 1.31 の quantize_static(calibration_cache_path=...)
で利用できます。元モデル、入力画像、量子化対象の値は変更しません。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes(pid: int | None = None) -> int | None:
    """Linux ではプロセスの現在の常駐メモリ量を取得します。"""
    try:
        status_path = Path('/proc/self/status' if pid is None else f'/proc/{pid}/status')
        for line in status_path.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return None


def _proc_pid() -> int | None:
    """PID namespace と /proc の参照先が異なる環境にも対応します。"""
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('Pid:'):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return None


def _cgroup_memory() -> tuple[int | None, int | None]:
    try:
        root = Path('/sys/fs/cgroup')
        current = int((root / 'memory.current').read_text().strip())
        maximum = (root / 'memory.max').read_text().strip()
        return current, int(maximum) if maximum != 'max' else None
    except (FileNotFoundError, PermissionError, ValueError):
        return None, None


def _topological_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """元の順序を優先しつつ、依存する演算を先へ配置します。"""
    nodes = list(graph.node)
    producers = {name: i for i, node in enumerate(nodes) for name in node.output if name}
    consumers: dict[int, set[int]] = {}
    dependencies = []
    for index, node in enumerate(nodes):
        upstream = {producers[name] for name in node.input if name in producers}
        dependencies.append(len(upstream))
        for producer in upstream:
            consumers.setdefault(producer, set()).add(index)
    ready = [index for index, count in enumerate(dependencies) if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(nodes[index])
        for consumer in consumers.get(index, ()):
            dependencies[consumer] -= 1
            if dependencies[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(ordered) != len(nodes):
        raise ValueError('較正対象グラフに循環した依存関係があります。')
    return ordered


def build_calibration_graph(model_path: Path, destination: Path,
                            op_types: Iterable[str],
                            node_names: Iterable[str] | None = None,
                            tensor_names: Iterable[str] | None = None,
                            retain_original_outputs: bool = False) -> list[dict]:
    """対象の浮動小数テンソルを全要素で縮約する較正グラフを作成します。"""
    model = onnx.load(str(model_path))
    graph = model.graph
    initializer_names = {item.name for item in graph.initializer}
    value_types = {item.name: item.type.tensor_type.elem_type
                   for item in (*graph.input, *graph.value_info, *graph.output)
                   if item.type.HasField('tensor_type')}
    value_types.update({item.name: item.data_type for item in graph.initializer})
    operations = set(op_types)
    selected_names = None if node_names is None else set(node_names)
    if not operations:
        raise ValueError('較正する演算を少なくとも一つ指定してください。')
    actual_names = {node.name for node in graph.node}
    if selected_names is not None and selected_names - actual_names:
        raise ValueError(f'指定したノードがモデルにありません: {sorted(selected_names - actual_names)}')
    tensors = []
    seen = set()
    for node in graph.node:
        if node.op_type not in operations or (selected_names is not None and node.name not in selected_names):
            continue
        for name in (*node.input, *node.output):
            if not name or name in seen or name in initializer_names:
                continue
            dtype = value_types.get(name)
            if dtype is None:
                raise ValueError(f'型情報がありません。形状推論後のモデルを使用してください: {name}')
            if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16):
                continue
            seen.add(name)
            tensors.append(name)
    if not tensors:
        raise ValueError('較正する浮動小数テンソルが見つかりません。')
    if tensor_names is not None:
        requested_tensors = set(tensor_names)
        if requested_tensors - set(tensors):
            raise ValueError('指定された較正テンソルの一部が対象グラフにありません。')
        tensors = [name for name in tensors if name in requested_tensors]
    value_infos = {item.name: item for item in (*graph.input, *graph.value_info, *graph.output)}

    reserved = {name for node in graph.node for name in (*node.input, *node.output)}
    reserved.update(node.name for node in graph.node)
    reserved.update(initializer_names)

    def new_name(base: str) -> str:
        name, suffix = base, 0
        while name in reserved:
            suffix += 1
            name = f'{base}_{suffix}'
        reserved.add(name)
        return name

    added: dict[str, list[onnx.NodeProto]] = {}
    ranges = []
    outputs = []
    guarded_tensors = {}
    zero_names = {}
    for index, tensor in enumerate(tensors):
        dimensions = value_infos[tensor].type.tensor_type.shape.dim
        fixed = all(d.HasField('dim_value') and d.dim_value > 0 for d in dimensions)
        item_size = 2 if value_types[tensor] == TensorProto.FLOAT16 else 4
        tensor_bytes = int(np.prod([d.dim_value for d in dimensions])) * item_size if fixed else None
        entry = {'tensor': tensor, 'tensor_bytes': tensor_bytes}
        for kind, operator in (('min', 'ReduceMin'), ('max', 'ReduceMax')):
            name = new_name(f'CalibrationCPU/range_{index}/{kind}')
            # keepdims=0 gives one scalar and needs no extra Reshape node.
            node = helper.make_node(operator, [tensor], [name], name=name, keepdims=0)
            added.setdefault(tensor, []).append(node)
            outputs.append(helper.make_tensor_value_info(name, value_types[tensor], []))
            entry[kind] = name
        # A graph output alone does not force ORT to execute its reduction early.
        # These data dependencies require both reductions before every original
        # consumer. Reshape changes no values or storage; only its shape depends
        # on the zero token. Calibration requires finite intermediate values.
        dtype = value_types[tensor]
        if dtype not in zero_names:
            zero_names[dtype] = new_name(f'CalibrationCPU/zero_{dtype}')
            graph.initializer.append(numpy_helper.from_array(
                np.asarray(0, dtype=np.float16 if dtype == TensorProto.FLOAT16 else np.float32),
                zero_names[dtype]))
        gate = {kind: new_name(f'CalibrationCPU/guard_{index}/{kind}')
                for kind in ('min_zero', 'max_zero', 'zero', 'zero_int64', 'shape', 'guard_shape', 'value')}
        gate_nodes = [
            helper.make_node('Mul', [entry['min'], zero_names[dtype]], [gate['min_zero']], name=gate['min_zero']),
            helper.make_node('Mul', [entry['max'], zero_names[dtype]], [gate['max_zero']], name=gate['max_zero']),
            helper.make_node('Add', [gate['min_zero'], gate['max_zero']], [gate['zero']], name=gate['zero']),
            helper.make_node('Cast', [gate['zero']], [gate['zero_int64']], name=gate['zero_int64'], to=TensorProto.INT64),
        ]
        if fixed:
            graph.initializer.append(numpy_helper.from_array(
                np.asarray([d.dim_value for d in dimensions], dtype=np.int64), gate['shape']))
        else:
            gate_nodes.append(helper.make_node('Shape', [tensor], [gate['shape']], name=gate['shape']))
        gate_nodes.extend([
            helper.make_node('Add', [gate['shape'], gate['zero_int64']], [gate['guard_shape']], name=gate['guard_shape']),
            helper.make_node('Reshape', [tensor, gate['guard_shape']], [gate['value']], name=gate['value']),
        ])
        added[tensor].extend(gate_nodes)
        guarded_tensors[tensor] = gate['value']
        ranges.append(entry)

    ordered = []
    produced = initializer_names | {item.name for item in graph.input}
    for name in tensors:
        if name in produced:
            ordered.extend(added[name])
    for node in _topological_nodes(graph):
        for index, name in enumerate(node.input):
            node.input[index] = guarded_tensors.get(name, name)
        ordered.append(node)
        for name in node.output:
            ordered.extend(added.get(name, ()))
    del graph.node[:]
    graph.node.extend(ordered)
    if not retain_original_outputs:
        del graph.output[:]
    graph.output.extend(outputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return ranges


def _run_worker(configuration: dict) -> dict:
    from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData, save_tensors_data

    start = time.perf_counter()
    Path(configuration['worker_pid_path']).write_text(json.dumps({'proc_pid': _proc_pid()}), encoding='utf-8')
    model_path = Path(configuration['model_path'])
    cache_path = Path(configuration['cache_path'])
    paths = [Path(path) for path in configuration['calibration_paths']]
    augmented = cache_path.parent / 'calibration_augmented.onnx'
    ranges = build_calibration_graph(model_path, augmented, configuration['op_types'],
                                     configuration.get('node_names'))
    gc.collect()
    print(json.dumps({'event': 'calibration_graph_ready', 'tensors': len(ranges),
                      'seconds': time.perf_counter() - start}), flush=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = configuration['threads']
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # MEMORY_EFFICIENT exists in the Python enum but requires a training build.
    # The ordinary CPU wheel must use DEFAULT; early reductions are in the graph.
    options.execution_order = ort.ExecutionOrder.DEFAULT
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = True
    options.log_severity_level = 3
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    session = ort.InferenceSession(str(augmented), options, providers=['CPUExecutionProvider'])
    init_seconds = time.perf_counter() - start
    print(json.dumps({'event': 'calibration_session_ready', 'seconds': init_seconds,
                      'rss_mib': (_rss_bytes() or 0) / 2**20}), flush=True)

    input_specs = session.get_inputs()
    output_names = [item[kind] for item in ranges for kind in ('min', 'max')]
    lows = np.full(len(ranges), np.inf, dtype=np.float32)
    highs = np.full(len(ranges), -np.inf, dtype=np.float32)
    samples = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feeds = {}
            for specification in input_specs:
                if specification.name not in archive:
                    raise ValueError(f'{path} に入力 {specification.name} がありません。')
                dtype = {'tensor(float)': np.float32, 'tensor(float16)': np.float16,
                         'tensor(int64)': np.int64, 'tensor(int32)': np.int32,
                         'tensor(bool)': np.bool_}.get(specification.type)
                if dtype is None:
                    raise ValueError(f'較正入力の型に対応していません: {specification.type}')
                value = np.ascontiguousarray(archive[specification.name], dtype=dtype)
                if not np.isfinite(value).all():
                    raise ValueError(f'{path} の {specification.name} に非有限値があります。')
                feeds[specification.name] = value
        sample_start = time.perf_counter()
        values = np.asarray([float(np.asarray(value).reshape(()))
                             for value in session.run(output_names, feeds)], dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(values).all() or np.any(values[:, 0] > values[:, 1]):
            raise ValueError(f'{path} の較正結果に無効な最小値・最大値があります。')
        lows = np.minimum(lows, values[:, 0])
        highs = np.maximum(highs, values[:, 1])
        sample = {'path': str(path), 'sha256': _sha256(path),
                  'seconds': time.perf_counter() - sample_start,
                  'rss_after_mib': (_rss_bytes() or 0) / 2**20}
        samples.append(sample)
        print(json.dumps({'event': 'calibration_sample_finished', **sample}), flush=True)
        del feeds, values
    tensors_data = TensorsData(CalibrationMethod.MinMax,
                              {item['tensor']: (np.asarray(lows[index]), np.asarray(highs[index]))
                               for index, item in enumerate(ranges)})
    temporary_cache = cache_path.with_suffix('.partial.json')
    save_tensors_data(tensors_data, temporary_cache)
    temporary_cache.replace(cache_path)
    report = {'model_path': str(model_path), 'model_sha256': _sha256(model_path),
              'cache_path': str(cache_path), 'onnxruntime_version': ort.__version__,
              'method': 'MinMax', 'range_tensor_count': len(ranges),
              'range_tensor_storage_mib': sum(item['tensor_bytes'] or 0 for item in ranges) / 2**20,
              'reduction_dependency': 'finite min/max zero tokens feed identity Reshape shapes',
              'finite_intermediate_values_required': True,
              'original_outputs_in_calibration': False,
              'selected_node_count': None if configuration.get('node_names') is None else len(configuration['node_names']),
              'op_types': configuration['op_types'], 'threads': configuration['threads'],
              'execution_order': 'DEFAULT; early scalar reductions in graph',
              'graph_optimization': 'ORT_ENABLE_ALL',
              'cpu_memory_arena': False, 'memory_pattern': False,
              'preparation_and_session_initialization_seconds': init_seconds,
              'samples': samples, 'total_seconds': time.perf_counter() - start}
    try:
        import resource
        factor = 1 if sys.platform == 'darwin' else 1024
        report['worker_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor / 2**20
    except ImportError:
        report['worker_peak_rss_mib'] = None
    del session
    augmented.unlink(missing_ok=True)
    Path(configuration['report_path']).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'calibration_complete', 'seconds': report['total_seconds'],
                      'tensor_count': len(ranges)}), flush=True)
    return report


def calibrate_minmax(model_path: str | Path, cache_path: str | Path,
                     calibration_paths: Iterable[str | Path],
                     op_types: Iterable[str] = ('Conv',), threads: int = 8,
                     node_names: Iterable[str] | None = None,
                     max_worker_rss_mib: int = 6144) -> dict:
    """独立プロセスで較正し、使用した入力・処理時間・メモリの記録を返します。

    Linux では 0.1 秒ごとに常駐メモリと cgroup 全体の使用量を監視し、
    指定上限または cgroup 上限の 87.5% を超えた場合は子プロセスを停止します。
    これは他の大規模な推論を同時に行わないことを前提とした保護です。
    """
    model_path, cache_path = Path(model_path).resolve(), Path(cache_path).resolve()
    paths = [Path(path).resolve() for path in calibration_paths]
    if not model_path.is_file() or not paths or any(not path.is_file() for path in paths):
        raise ValueError('モデルと一つ以上の既存の較正 NPZ を指定してください。')
    if threads < 1 or max_worker_rss_mib < 256:
        raise ValueError('スレッド数は1以上、メモリ上限は256MiB以上にしてください。')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = cache_path.with_suffix('.report.json')
    log_path = cache_path.with_suffix('.log')
    start = time.perf_counter()
    peak_rss, peak_cgroup = 0, 0
    with tempfile.TemporaryDirectory(prefix='calibrate_cpu_', dir=cache_path.parent) as directory:
        directory = Path(directory)
        staged_cache = directory / 'ranges.json'
        worker_pid_path = directory / 'worker_pid.json'
        configuration = {'model_path': str(model_path), 'cache_path': str(staged_cache),
                         'calibration_paths': [str(path) for path in paths],
                         'op_types': list(op_types), 'threads': threads,
                         'node_names': None if node_names is None else list(node_names),
                         'report_path': str(directory / 'report.json'),
                         'worker_pid_path': str(worker_pid_path)}
        configuration_path = directory / 'configuration.json'
        configuration_path.write_text(json.dumps(configuration), encoding='utf-8')
        last_progress = start
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-m', 'diffmatte_cpu.calibrate_cpu', '--worker',
                                        str(configuration_path)], stdout=log, stderr=subprocess.STDOUT)
            try:
                observed_proc_pid = None
                while process.poll() is None:
                    if observed_proc_pid is None and worker_pid_path.is_file():
                        try:
                            observed_proc_pid = json.loads(worker_pid_path.read_text())['proc_pid']
                        except (OSError, ValueError):
                            pass
                    rss = (_rss_bytes(observed_proc_pid) or 0) if observed_proc_pid else 0
                    current, maximum = _cgroup_memory()
                    peak_rss, peak_cgroup = max(peak_rss, rss), max(peak_cgroup, current or 0)
                    if rss > max_worker_rss_mib * 2**20 or (maximum and current and current > maximum * .875):
                        raise MemoryError(f'較正のメモリ上限に到達しました。子プロセスRSS={rss / 2**20:.0f}MiB, '
                                          f'cgroup={(current or 0) / 2**20:.0f}MiB。ログ: {log_path}')
                    now = time.perf_counter()
                    if now - last_progress > 15:
                        print(json.dumps({'event': 'calibration_running', 'seconds': now - start,
                                          'rss_mib': rss / 2**20,
                                          'cgroup_mib': (current or 0) / 2**20}), flush=True)
                        last_progress = now
                    time.sleep(.1)
                if process.returncode != 0:
                    raise RuntimeError(f'較正が終了コード {process.returncode} で停止しました。ログ: {log_path}\n'
                                       + log_path.read_text(encoding='utf-8')[-4000:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        staged_cache.replace(cache_path)
        report.update({'cache_path': str(cache_path), 'log_path': str(log_path),
                       'worker_peak_rss_mib_sampled': peak_rss / 2**20,
                       'cgroup_peak_mib_sampled': peak_cgroup / 2**20,
                       'watchdog_interval_seconds': .1,
                       'wall_seconds_including_process': time.perf_counter() - start})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--calibration', type=Path, nargs='+')
    parser.add_argument('--op-types', nargs='+', default=['Conv'])
    parser.add_argument('--node-names', type=Path, help='対象ノード名のJSON配列')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-worker-rss-mib', type=int, default=6144)
    args = parser.parse_args()
    if args.worker:
        _run_worker(json.loads(args.worker.read_text(encoding='utf-8')))
        return
    if not args.model or not args.cache or not args.calibration:
        parser.error('--model、--cache、--calibration を指定してください。')
    names = None if args.node_names is None else json.loads(args.node_names.read_text(encoding='utf-8'))
    report = calibrate_minmax(args.model, args.cache, args.calibration,
                             op_types=args.op_types, threads=args.threads,
                             node_names=names, max_worker_rss_mib=args.max_worker_rss_mib)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
