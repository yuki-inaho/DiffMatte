#!/usr/bin/env python3
"""DiffMatte ONNX を CPU で実行し、アルファ画像と計測記録を保存します。

入力画像は RGB の 0..1、トライマップは 0 / 0.5 / 1 で渡します。
平均・標準偏差による正規化はモデル内にあるため、ここでは行いません。
PNG への保存時だけ 0..1 に制限し、任意保存の NPY には元の浮動小数点値を
残します。元画像より小さいモデルへの縮小は行いません。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
REQUIRED_INPUT_CHANNELS = {"image": 3, "trimap": 1, "initial_noise": 1}


def default_thread_count() -> int:
    """通常の CPU で過剰な並列化を避ける初期値です。CLI で調整できます。"""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """推論と比較の両コマンドで同じ計測条件を受け付けます。"""
    parser.add_argument("--seed", type=int, default=8282, help="初期乱数の種（既定: 8282）")
    parser.add_argument(
        "--threads", type=int, default=default_thread_count(),
        help="演算内の CPU スレッド数。0 は ONNX Runtime に委任（既定: 最大 4）",
    )
    parser.add_argument(
        "--inter-op-threads", type=int, default=1,
        help="演算間スレッド数（既定: 1。演算順序は ORT_SEQUENTIAL）",
    )
    parser.add_argument("--warmup", type=int, default=1, help="計測前の推論回数（既定: 1）")
    parser.add_argument("--runs", type=int, default=3, help="計測する推論回数（既定: 3）")
    parser.add_argument(
        "--pad-to-model", action="store_true",
        help="入力が小さい場合だけ右・下へ端の画素を複製し、出力を元寸法へ戻す",
    )
    parser.add_argument(
        "--no-spinning", action="store_true",
        help="待機中の CPU スレッドの能動的な待機を無効にする",
    )
    parser.add_argument(
        "--prepack-float-weights", action="store_true",
        help="圧縮された畳み込み重みを起動時だけ float32 定数に戻す（保存したモデルは変更しない）",
    )


def validate_execution_arguments(args: argparse.Namespace) -> None:
    if not 0 <= args.seed < 2**64:
        raise ValueError("--seed は 0 以上 2**64 未満の整数にしてください。")
    if args.threads < 0 or args.inter_op_threads < 1:
        raise ValueError("--threads は 0 以上、--inter-op-threads は 1 以上が必要です。")
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup は 0 以上、--runs は 1 以上が必要です。")


def read_source_inputs(image_path: Path, trimap_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """実演用ソースと同じ方法で NCHW の RGB と三値トライマップを読みます。"""
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    with Image.open(trimap_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32) / np.float32(255.0)
    if rgb.shape[:2] != gray.shape:
        raise ValueError(
            f"画像とトライマップの寸法が異なります: {rgb.shape[:2]} / {gray.shape}。"
        )
    if min(gray.shape) < 1:
        raise ValueError("空の画像は使用できません。")
    # run_one_image.py の区切り方に合わせます。中間階調を含む画像にも対応します。
    tri = np.where(gray > 0.9, 1.0, np.where(gray < 0.1, 0.0, 0.5)).astype(np.float32)
    return (
        np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]),
        np.ascontiguousarray(tri[None, None]),
    )


def inspect_input_shape(session: Any) -> tuple[int, int]:
    """この配布モデルの入力契約を確認し、固定された高さ・幅を返します。"""
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != set(REQUIRED_INPUT_CHANNELS):
        raise ValueError(
            "モデル入力は image / trimap / initial_noise の 3 つが必要です。"
            f"実際: {list(inputs)}"
        )
    image_shape = inputs["image"].shape
    if len(image_shape) != 4 or not all(isinstance(v, int) and v > 0 for v in image_shape):
        raise ValueError(f"固定寸法 NCHW のモデルが必要です。実際: {image_shape}")
    height, width = image_shape[-2:]
    for name, channels in REQUIRED_INPUT_CHANNELS.items():
        item = inputs[name]
        expected = [1, channels, height, width]
        if item.type != "tensor(float)" or list(item.shape) != expected:
            raise ValueError(
                f"入力 {name} は float32 {expected} が必要です。"
                f"実際: {item.type} {item.shape}"
            )
    outputs = {item.name: item for item in session.get_outputs()}
    if "alpha" not in outputs or outputs["alpha"].type != "tensor(float)":
        raise ValueError("モデルには float32 の alpha 出力が必要です。")
    return height, width


def make_feeds(
    image: np.ndarray,
    trimap: np.ndarray,
    model_shape: tuple[int, int],
    seed: int,
    allow_padding: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """暗黙の縮小を避け、必要時のみ余白を追加して同じ乱数を生成します。"""
    source_h, source_w = image.shape[-2:]
    target_h, target_w = model_shape
    if source_h > target_h or source_w > target_w:
        raise ValueError(
            f"入力 {source_h}×{source_w} はモデル {target_h}×{target_w} より大きいです。"
            "寸法の自動変更は行いません。対応するモデルまたは入力をご用意ください。"
        )
    pad_h, pad_w = target_h - source_h, target_w - source_w
    if (pad_h or pad_w) and not allow_padding:
        raise ValueError(
            f"入力 {source_h}×{source_w} とモデル {target_h}×{target_w} が一致しません。"
            "右・下への余白追加を許可する場合は --pad-to-model を指定してください。"
        )
    if pad_h or pad_w:
        pads = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
        image = np.pad(image, pads, mode="edge")
        trimap = np.pad(trimap, pads, mode="edge")
    noise = np.random.default_rng(seed).standard_normal(
        (1, 1, target_h, target_w), dtype=np.float32
    )
    feeds = {
        "image": np.ascontiguousarray(image),
        "trimap": np.ascontiguousarray(trimap),
        "initial_noise": noise,
    }
    padding = {
        "requested": allow_padding,
        "performed": bool(pad_h or pad_w),
        "mode": "edge",
        "top": 0,
        "left": 0,
        "bottom": pad_h,
        "right": pad_w,
        "source_height": source_h,
        "source_width": source_w,
        "model_height": target_h,
        "model_width": target_w,
        "output_cropped_to_source": bool(pad_h or pad_w),
    }
    return feeds, padding


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """書き込み途中の JSON が残らないよう、一時ファイルから置換します。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_peak_rss() -> dict[str, Any]:
    """OS が記録した、このプロセス全体の常駐メモリ最大値です。"""
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return {"bytes": int(raw * 1024), "source": "Linux ru_maxrss (KiB × 1024)"}
        if sys.platform == "darwin":
            return {"bytes": int(raw), "source": "macOS ru_maxrss (bytes)"}
        return {"bytes": None, "source": "ru_maxrss: この OS の単位は未確認", "raw": raw}
    except (ImportError, OSError, AttributeError):
        return {"bytes": None, "source": "利用できません"}


class RSSMonitor:
    """任意の psutil による簡易測定。OS のプロセス最大値とは別に記録します。"""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._read = None
        self.source = "利用できません（psutil を任意で導入できます）"
        try:
            import psutil

            process = psutil.Process(os.getpid())
            self._read = lambda: process.memory_info().rss
            self.source = "psutil.Process.memory_info().rss"
        except Exception:
            # 一部のコンテナでは psutil が別の PID 名前空間を参照する場合があります。
            # このときも推論を妨げず、Linux の自己プロセス情報があれば利用します。
            try:
                statm = Path("/proc/self/statm")
                page_bytes = int(os.sysconf("SC_PAGE_SIZE"))
                if statm.is_file():
                    self._read = lambda: int(statm.read_text().split()[1]) * page_bytes
                    self.source = "/proc/self/statm resident pages × page size"
            except (AttributeError, OSError, ValueError, IndexError):
                pass

    def sample(self) -> int | None:
        if self._read is None:
            return None
        try:
            value = int(self._read())
            self.peak_bytes = max(self.peak_bytes or 0, value)
            return value
        except Exception:
            # メモリ計測が取得できなくても、推論そのものは継続します。
            return None

    def start(self) -> None:
        self.sample()
        if self._read is not None:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sample()


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_execution_arguments(args)
    model_path = args.model.expanduser().resolve(strict=True)
    image_path = args.image.expanduser().resolve(strict=True)
    trimap_path = args.trimap.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    timing_path = (
        args.timing_json.expanduser().resolve()
        if args.timing_json is not None else output_dir / "timing.json"
    )
    pipeline_start = time.perf_counter()
    import onnxruntime as ort

    monitor = RSSMonitor()
    monitor.start()
    rss_before_preparation = monitor.sample()
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = args.inter_op_threads
        if args.no_spinning:
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        model_source: str | bytes = str(model_path)
        preparation_report = {
            "enabled": False,
            "on_disk_model_unchanged": True,
            "disk_model_file_bytes": model_path.stat().st_size,
            "in_memory_model_bytes": None,
            "restored_weight_count": 0,
            "rewritten_conv_node_count": 0,
            "preparation_seconds": 0.0,
        }
        if args.prepack_float_weights:
            from .cpu_runtime import prepare_cpu_model

            model_source, preparation_report = prepare_cpu_model(model_path)
        rss_before_session = monitor.sample()
        started = time.perf_counter()
        session = ort.InferenceSession(
            model_source, sess_options=options, providers=["CPUExecutionProvider"]
        )
        initialization_seconds = time.perf_counter() - started
        del model_source
        rss_after_session = monitor.sample()
        model_shape = inspect_input_shape(session)

        started = time.perf_counter()
        image, trimap = read_source_inputs(image_path, trimap_path)
        feeds, padding = make_feeds(image, trimap, model_shape, args.seed, args.pad_to_model)
        feed_hashes = {
            name: hashlib.sha256(array.tobytes(order="C")).hexdigest()
            for name, array in feeds.items()
        }
        input_preparation_seconds = time.perf_counter() - started

        warmup_seconds = []
        for _ in range(args.warmup):
            started = time.perf_counter()
            session.run(["alpha"], feeds)
            warmup_seconds.append(time.perf_counter() - started)

        measured_seconds = []
        result = None
        for _ in range(args.runs):
            started = time.perf_counter()
            result = session.run(["alpha"], feeds)[0]
            measured_seconds.append(time.perf_counter() - started)
        rss_after_runs = monitor.sample()

        expected_shape = (1, 1, *model_shape)
        if result is None or result.shape != expected_shape or result.dtype != np.float32:
            actual = None if result is None else (result.shape, str(result.dtype))
            raise ValueError(f"alpha 出力は float32 {expected_shape} が必要です。実際: {actual}")
        source_h, source_w = padding["source_height"], padding["source_width"]
        alpha = np.ascontiguousarray(result[0, 0, :source_h, :source_w])
        if not np.isfinite(alpha).all():
            raise ValueError("alpha に NaN または無限大が含まれています。出力を保存しません。")

        output_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {"alpha16_png": str(output_dir / "alpha16.png")}
        clipped = np.clip(alpha, 0.0, 1.0)
        Image.fromarray(np.rint(clipped * 65535.0).astype(np.uint16)).save(files["alpha16_png"])
        if args.save_alpha8:
            files["alpha8_png"] = str(output_dir / "alpha8.png")
            Image.fromarray(np.rint(clipped * 255.0).astype(np.uint8)).save(files["alpha8_png"])
        if args.save_npy:
            files["alpha_npy"] = str(output_dir / "alpha.npy")
            np.save(files["alpha_npy"], alpha, allow_pickle=False)
        files["timing_json"] = str(timing_path)

        report = {
            "schema_version": SCHEMA_VERSION,
            "model": str(model_path),
            "model_file_bytes": model_path.stat().st_size,
            "image": str(image_path),
            "trimap": str(trimap_path),
            "runtime": {
                "onnxruntime_version": ort.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "requested_providers": ["CPUExecutionProvider"],
                "observed_providers": session.get_providers(),
                "graph_optimization_level": "ORT_ENABLE_ALL",
                "execution_mode": "ORT_SEQUENTIAL",
                "intra_op_num_threads": args.threads,
                "inter_op_num_threads": args.inter_op_threads,
                "spinning_disabled": args.no_spinning,
                "prepack_float_weights": args.prepack_float_weights,
            },
            "cpu_weight_preparation": preparation_report,
            "input": {
                "shapes": {name: list(array.shape) for name, array in feeds.items()},
                "dtype": "float32",
                "image_range": "RGB 0..1; 平均・標準偏差の正規化はモデル内部",
                "trimap_rule": ">0.9: 1; <0.1: 0; それ以外: 0.5（画素値 / 255）",
                "seed": args.seed,
                "noise_generator": "numpy.default_rng(seed).standard_normal(dtype=float32)",
                "same_noise_for_all_runs": True,
                "feeds_sha256": feed_hashes,
                "padding": padding,
            },
            "timing_seconds": {
                "session_initialization": initialization_seconds,
                "cpu_weight_preparation": preparation_report["preparation_seconds"],
                "model_preparation_and_session_initialization": (
                    preparation_report["preparation_seconds"] + initialization_seconds
                ),
                "input_preparation": input_preparation_seconds,
                "warmup": warmup_seconds,
                "measured": measured_seconds,
                "mean": float(np.mean(measured_seconds)),
                "median": float(np.median(measured_seconds)),
                "minimum": min(measured_seconds),
                "maximum": max(measured_seconds),
                "population_std": float(np.std(measured_seconds)),
            },
            "timing_scope": (
                "推論時間は同期 session.run のみ。画像読み込み、初期化、PNG/NPY 保存を含みません。"
                "各回で同じ画像・トライマップ・初期乱数を使用します。"
            ),
            "output": {
                "shape": list(alpha.shape),
                "minimum": float(alpha.min()),
                "maximum": float(alpha.max()),
                "pixels_clipped_for_png": int(np.count_nonzero((alpha < 0.0) | (alpha > 1.0))),
                "postprocessing": (
                    "モデルの出力条件を保持。元寸法に切り出し、PNG のみ [0,1] に制限して丸めます。"
                    "NPY は浮動小数点出力を保持し、前景の強制設定は追加しません。"
                ),
                "files": files,
            },
        }
    finally:
        monitor.stop()

    report["timing_seconds"]["pipeline_elapsed_before_report_write"] = time.perf_counter() - pipeline_start
    report["memory"] = {
        "rss_before_preparation_bytes": rss_before_preparation,
        "rss_before_session_bytes": rss_before_session,
        "rss_after_session_bytes": rss_after_session,
        "rss_after_runs_bytes": rss_after_runs,
        "sampled_rss_peak_bytes": monitor.peak_bytes,
        "sampled_rss_source": monitor.source,
        "sample_interval_seconds": monitor.interval_seconds,
        "process_lifetime_peak_rss": process_peak_rss(),
        "scope": (
            "いずれも Python プロセス全体の常駐メモリです。モデル単体の使用量ではありません。"
            "標本測定は短時間の最大値を取り逃す場合があります。"
        ),
    }
    write_json(timing_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="使用する ONNX ファイル")
    parser.add_argument("--image", type=Path, required=True, help="RGB 入力画像")
    parser.add_argument("--trimap", type=Path, required=True, help="対応するトライマップ")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="保存先（既定: output）")
    parser.add_argument("--save-alpha8", action="store_true", help="16 ビット PNG に加えて 8 ビット PNG も保存")
    parser.add_argument("--save-npy", action="store_true", help="元寸法の float32 アルファを alpha.npy に保存")
    parser.add_argument("--timing-json", type=Path, help="計測 JSON の保存先（既定: 保存先/timing.json）")
    add_execution_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except KeyboardInterrupt:
        print("中断しました。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"実行できませんでした: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    timing = report["timing_seconds"]
    padding = report["input"]["padding"]
    if padding["performed"]:
        print(
            f"余白追加: 下 {padding['bottom']} 画素、右 {padding['right']} 画素（端を複製）。"
            "出力は元の寸法へ戻しました。"
        )
    print(f"保存先: {report['output']['files']['alpha16_png']}")
    preparation = report["cpu_weight_preparation"]
    if preparation["enabled"]:
        print(
            f"CPU 用の定数重み準備: 重み {preparation['restored_weight_count']} 個、"
            f"畳み込み {preparation['rewritten_conv_node_count']} 箇所、"
            f"{preparation['preparation_seconds']:.6f} 秒。"
        )
    print(
        f"CPU 推論 {len(timing['measured'])} 回: 中央値 {timing['median']:.6f} 秒、"
        f"平均 {timing['mean']:.6f} 秒。初期化 {timing['session_initialization']:.6f} 秒。"
    )
    print(f"計測記録: {report['output']['files']['timing_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
