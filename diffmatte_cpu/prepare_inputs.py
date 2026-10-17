#!/usr/bin/env python3
"""DiffMatte の画像・トライマップ・初期乱数を較正用 NPZ に保存します。

ONNX の入力定義だけを読み、推論用と同じ前処理・乱数生成を使います。
モデルの読み込みによる ONNX Runtime セッションの作成や推論は行いません。
元画像より大きい固定モデルには --pad-to-model で右・下へ余白を追加できます。
保存先と同名の JSON にハッシュ、乱数の種、寸法、余白追加の記録を残します。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from .run_diffmatte import (
    REQUIRED_INPUT_CHANNELS,
    inspect_input_shape,
    make_feeds,
    read_source_inputs,
    write_json,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fixed_model_shape(model_path: Path) -> tuple[int, int]:
    """ONNX の入出力定義を既存の入力契約検証へ渡します。推論は不要です。"""
    model = onnx.load(str(model_path), load_external_data=False)

    def argument(value: onnx.ValueInfoProto) -> SimpleNamespace:
        if not value.type.HasField('tensor_type'):
            raise ValueError(f'テンソル以外の入出力には対応していません: {value.name}')
        tensor = value.type.tensor_type
        shape = [dimension.dim_value if dimension.HasField('dim_value')
                 else dimension.dim_param or None for dimension in tensor.shape.dim]
        dtype = 'tensor(float)' if tensor.elem_type == TensorProto.FLOAT else TensorProto.DataType.Name(tensor.elem_type)
        return SimpleNamespace(name=value.name, shape=shape, type=dtype)

    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [argument(value) for value in model.graph.input if value.name not in initializers]
    outputs = [argument(value) for value in model.graph.output]
    interface = SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs)
    return inspect_input_shape(interface)


def prepare_inputs(model_path: Path, image_path: Path, trimap_path: Path,
                   output_path: Path, seed: int = 8282,
                   allow_padding: bool = False) -> dict[str, Any]:
    """元ファイルを保護し、入力配列と再現用の情報を保存します。"""
    if not 0 <= seed < 2**64:
        raise ValueError('--seed は 0 以上 2**64 未満の整数にしてください。')
    model_path = model_path.expanduser().resolve(strict=True)
    image_path = image_path.expanduser().resolve(strict=True)
    trimap_path = trimap_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != '.npz':
        raise ValueError('--output は拡張子 .npz のファイルにしてください。')
    metadata_path = output_path.with_suffix('.json')
    source_paths = {model_path, image_path, trimap_path}
    if output_path in source_paths or metadata_path.resolve() in source_paths:
        raise ValueError('NPZ と JSON の保存先には、元モデル・画像・トライマップとは別のパスを指定してください。')

    model_shape = fixed_model_shape(model_path)
    image, trimap = read_source_inputs(image_path, trimap_path)
    feeds, padding = make_feeds(image, trimap, model_shape, seed, allow_padding)
    for name, channels in REQUIRED_INPUT_CHANNELS.items():
        array = feeds[name]
        expected = (1, channels, *model_shape)
        if array.dtype != np.float32 or array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f'入力 {name} は有限な float32 {expected} である必要があります。')

    padding = dict(padding)
    padding['inference_output_requires_crop'] = padding.pop('output_cropped_to_source')
    report: dict[str, Any] = {
        'schema_version': 1,
        'output_npz': str(output_path),
        'metadata_json': str(metadata_path),
        'inference_performed': False,
        'model': {'path': str(model_path), 'sha256': file_sha256(model_path),
                  'height': model_shape[0], 'width': model_shape[1]},
        'source_files': {
            'image': {'path': str(image_path), 'sha256': file_sha256(image_path)},
            'trimap': {'path': str(trimap_path), 'sha256': file_sha256(trimap_path)},
        },
        'seed': seed,
        'numpy_version': np.__version__,
        'onnx_version': onnx.__version__,
        'noise_generator': 'numpy.default_rng(seed).standard_normal(dtype=float32)',
        'preprocessing': {
            'image': 'RGB 0..1; 平均・標準偏差による正規化はモデル内部',
            'trimap': '画素値 / 255 が >0.9 なら1、<0.1なら0、それ以外は0.5',
            'padding': padding,
        },
        'inputs': {name: {'shape': list(array.shape), 'dtype': str(array.dtype),
                          'finite': True, 'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
                   for name, array in feeds.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz: Path | None = None
    temporary_json: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{output_path.stem}_', suffix='.npz',
                                         dir=output_path.parent, delete=False) as stream:
            temporary_npz = Path(stream.name)
            np.savez_compressed(stream, **feeds)
        report['npz_sha256'] = file_sha256(temporary_npz)
        report['npz_bytes'] = temporary_npz.stat().st_size
        temporary_json = temporary_npz.with_suffix('.json')
        write_json(temporary_json, report)
        temporary_npz.replace(output_path)
        temporary_json.replace(metadata_path)
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='固定入力寸法を持つ ONNX')
    parser.add_argument('--image', type=Path, required=True, help='RGB 入力画像')
    parser.add_argument('--trimap', type=Path, required=True, help='対応するトライマップ')
    parser.add_argument('--seed', type=int, default=8282, help='初期乱数の種（既定: 8282）')
    parser.add_argument('--pad-to-model', action='store_true', help='右・下へ端の画素を複製して固定寸法に合わせる')
    parser.add_argument('--output', type=Path, required=True, help='保存する .npz のパス')
    args = parser.parse_args()
    try:
        report = prepare_inputs(args.model, args.image, args.trimap, args.output,
                                seed=args.seed, allow_padding=args.pad_to_model)
    except KeyboardInterrupt:
        print('中断しました。', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'入力を保存できませんでした: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'output': report['output_npz'], 'metadata': report['metadata_json'],
                      'seed': report['seed'], 'shapes': {name: item['shape'] for name, item in report['inputs'].items()}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
