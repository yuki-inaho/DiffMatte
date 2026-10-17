#!/usr/bin/env python3
"""モデルごとに別プロセスで CPU 推論し、速度・容量・出力一致度を比較します。

基準には元の float32 ONNX を指定してください。同じ画像、トライマップ、
初期乱数を使用し、整数画像へ丸める前のアルファを比較します。
ここで求める指標は基準モデルとの一致度であり、正解に対する精度ではありません。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .run_diffmatte import (
    add_execution_arguments,
    read_source_inputs,
    validate_execution_arguments,
    write_json,
)


def difference_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """0..1 のアルファ差を float64 で集計します。PSNR の最大値は 1 です。"""
    if reference.shape != candidate.shape:
        raise ValueError(f"アルファ寸法が異なります: {reference.shape} / {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("比較対象に NaN または無限大が含まれています。")
    if mask is not None and mask.shape != reference.shape:
        raise ValueError(f"領域マスクの寸法が異なります: {mask.shape} / {reference.shape}")
    if mask is None:
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
    else:
        difference = candidate[mask].astype(np.float64) - reference[mask].astype(np.float64)
    count = int(difference.size)
    if not count:
        return {
            "pixel_count": 0, "mae": None, "mse": None, "rmse": None,
            "absolute_error_p99": None, "absolute_error_max": None,
            "psnr_db": None, "psnr_is_infinite": False, "empty_region": True,
        }
    absolute = np.abs(difference)
    mse = float(np.mean(difference * difference))
    return {
        "pixel_count": count,
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "absolute_error_p99": float(np.percentile(absolute, 99)),
        "absolute_error_max": float(np.max(absolute)),
        # JSON の Infinity は標準外のため、完全一致は別の真偽値で表します。
        "psnr_db": None if mse == 0 else -10.0 * math.log10(mse),
        "psnr_is_infinite": mse == 0,
        "empty_region": False,
    }


def run_worker(model: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """各モデルを新規 Python プロセスで実行し、終了後に記録を読みます。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "diffmatte_cpu.run_diffmatte",
        "--model", str(model),
        "--image", str(args.image),
        "--trimap", str(args.trimap),
        "--output-dir", str(output_dir),
        "--save-npy",
        "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--inter-op-threads", str(args.inter_op_threads),
        "--warmup", str(args.warmup),
        "--runs", str(args.runs),
    ]
    for flag in ("pad_to_model", "no_spinning", "save_alpha8", "prepack_float_weights"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        try:
            completed = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=args.timeout if args.timeout > 0 else None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = None
            timed_out = True
    record: dict[str, Any] = {
        "model": str(model),
        "model_file_bytes": model.stat().st_size,
        "status": "failed",
        "return_code": return_code,
        "timed_out": timed_out,
        "worker_wall_seconds": time.perf_counter() - started,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "timing_json": str(output_dir / "timing.json"),
        "alpha_npy": str(output_dir / "alpha.npy"),
        "alpha16_png": str(output_dir / "alpha16.png"),
    }
    if timed_out:
        record["error"] = f"指定した {args.timeout} 秒の制限時間を超えました。"
        return record
    if return_code != 0:
        record["error"] = f"推論処理が終了コード {return_code} で失敗しました。stderr.log を参照してください。"
        return record
    try:
        details = json.loads((output_dir / "timing.json").read_text(encoding="utf-8"))
        if not (output_dir / "alpha.npy").is_file():
            raise ValueError("alpha.npy が保存されていません。")
        record.update({
            "status": "succeeded",
            "runtime": details["runtime"],
            "input": details["input"],
            "timing_seconds": details["timing_seconds"],
            "memory": details["memory"],
            "cpu_weight_preparation": details["cpu_weight_preparation"],
        })
    except (OSError, ValueError, KeyError) as exc:
        record["error"] = f"推論の記録を読み取れませんでした: {exc}"
    return record


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validate_execution_arguments(args)
    if args.timeout < 0:
        raise ValueError("--timeout は 0 以上にしてください。")
    args.reference = args.reference.expanduser().resolve(strict=True)
    args.models = [path.expanduser().resolve(strict=True) for path in args.models]
    args.image = args.image.expanduser().resolve(strict=True)
    args.trimap = args.trimap.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    comparison_path = output_dir / "comparison.json"
    _, source_trimap = read_source_inputs(args.image, args.trimap)
    unknown = source_trimap[0, 0] == 0.5

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "output_consistency_against_user_supplied_float32_reference",
        "ground_truth_used": False,
        "interpretation": (
            "指標は元モデルの出力との一致度です。正解アルファに対する精度ではありません。"
            "全画像とトライマップの未知領域を、整数化前の 0..1 アルファで比較します。"
        ),
        "image": str(args.image),
        "trimap": str(args.trimap),
        "reference_model": str(args.reference),
        "unknown_pixel_count": int(np.count_nonzero(unknown)),
        "total_pixel_count": int(unknown.size),
        "metric_definitions": {
            "mae": "平均絶対差",
            "mse": "平均二乗差",
            "rmse": "平均二乗差の平方根",
            "absolute_error_p99": "絶対差の 99 パーセンタイル",
            "absolute_error_max": "絶対差の最大値",
            "psnr_db": "-10*log10(MSE)、アルファの最大値は 1",
            "psnr_is_infinite": "完全一致のとき true。この場合 psnr_db は null",
            "empty_region": "対象画素が 0 のとき true。指標は null",
        },
        "execution": {
            "isolated_process_per_model": True,
            "models_run_sequentially": True,
            "same_feed_hashes_required": True,
            "seed": args.seed,
            "threads": args.threads,
            "inter_op_threads": args.inter_op_threads,
            "warmup_count": args.warmup,
            "measured_count": args.runs,
            "pad_to_model": args.pad_to_model,
            "spinning_disabled": args.no_spinning,
            "prepack_float_weights_for_all_models": args.prepack_float_weights,
        },
        "models": [],
    }

    print(f"基準モデルを計測します: {args.reference.name}", flush=True)
    reference_record = run_worker(args.reference, output_dir / "00_reference", args)
    reference_record["role"] = "reference"
    report["models"].append(reference_record)
    if reference_record["status"] != "succeeded":
        write_json(comparison_path, report)
        print(f"基準モデルが実行できませんでした: {reference_record['error']}", file=sys.stderr)
        return report, 1

    reference = np.load(reference_record["alpha_npy"], mmap_mode="r", allow_pickle=False)
    if reference.shape != unknown.shape or not np.isfinite(reference).all():
        reference_record["status"] = "failed"
        reference_record["error"] = "基準アルファの寸法または数値が正しくありません。"
        write_json(comparison_path, report)
        return report, 1
    reference_record["consistency"] = {
        "full_image": difference_metrics(reference, reference),
        "unknown_region": difference_metrics(reference, reference, unknown),
    }
    reference_record["file_size_ratio_to_reference"] = 1.0
    reference_record["speedup_from_median"] = 1.0
    reference_seconds = reference_record["timing_seconds"]["median"]
    reference_bytes = reference_record["model_file_bytes"]
    failed = False
    write_json(comparison_path, report)

    for index, model in enumerate(args.models, start=1):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", model.stem)[:100] or "model"
        model_output = output_dir / f"{index:02d}_{name}"
        print(f"比較モデル {index}/{len(args.models)} を計測します: {model.name}", flush=True)
        record = run_worker(model, model_output, args)
        record["role"] = "candidate"
        if record["status"] == "succeeded":
            try:
                if record["input"]["feeds_sha256"] != reference_record["input"]["feeds_sha256"]:
                    raise ValueError("基準と入力のハッシュが一致しません。異なる入力・乱数の比較は行いません。")
                candidate = np.load(record["alpha_npy"], mmap_mode="r", allow_pickle=False)
                record["consistency"] = {
                    "full_image": difference_metrics(reference, candidate),
                    "unknown_region": difference_metrics(reference, candidate, unknown),
                }
                record["file_size_ratio_to_reference"] = record["model_file_bytes"] / reference_bytes
                record["file_size_reduction_percent"] = 100.0 * (1.0 - record["file_size_ratio_to_reference"])
                median = record["timing_seconds"]["median"]
                record["speedup_from_median"] = reference_seconds / median
                record["median_latency_reduction_percent"] = 100.0 * (1.0 - median / reference_seconds)
                del candidate
            except (ValueError, OSError, KeyError, ZeroDivisionError) as exc:
                record["status"] = "failed"
                record["error"] = f"出力の比較を完了できませんでした: {exc}"
        if record["status"] != "succeeded":
            failed = True
            print(f"比較できませんでした: {model.name}: {record.get('error', '不明なエラー')}", file=sys.stderr)
        else:
            metrics = record["consistency"]["unknown_region"]
            mae = "対象画素なし" if metrics["mae"] is None else f"{metrics['mae']:.8f}"
            print(
                f"{model.name}: 中央値 {record['timing_seconds']['median']:.6f} 秒、"
                f"速度比 {record['speedup_from_median']:.3f} 倍、未知領域 MAE {mae}",
                flush=True,
            )
        report["models"].append(record)
        write_json(comparison_path, report)

    del reference
    return report, 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="基準とする元の float32 ONNX")
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="比較する ONNX（複数指定可）")
    parser.add_argument("--image", type=Path, required=True, help="全モデルで共通の RGB 入力画像")
    parser.add_argument("--trimap", type=Path, required=True, help="対応するトライマップ")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"), help="比較結果の保存先")
    parser.add_argument("--save-alpha8", action="store_true", help="各モデルの 8 ビット PNG も保存")
    parser.add_argument("--timeout", type=float, default=0.0, help="モデル 1 個の制限秒数。0 は制限なし")
    add_execution_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _, status = run(args)
    except KeyboardInterrupt:
        print("中断しました。完了分の記録は保存先をご確認ください。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"比較できませんでした: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"比較記録: {args.output_dir.expanduser().resolve() / 'comparison.json'}")
    print("これらの指標は基準モデルとの出力一致度です。正解に対する精度は測定していません。")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
