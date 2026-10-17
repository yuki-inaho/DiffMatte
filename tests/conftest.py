from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
ORIGINAL_NAME = "diffmatte_vits_1024_ddim10_643x960.onnx"
ORIGINAL_SHA256 = "d69827e512803e1fd352e73aff55f327f766564d47de77908bac84d5bbc6acd6"
ORIGINAL_CANDIDATES = (
    REPO_ROOT / "artifacts" / "models" / ORIGINAL_NAME,
    REPO_ROOT.parent / "diffmatte_onnx" / "models" / ORIGINAL_NAME,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_module(module: str, *arguments: object, timeout: float = 1800) -> subprocess.CompletedProcess[str]:
    """CLI と同じく別プロセスで実行し、失敗時は標準出力・標準エラーを示します。"""
    command = [sys.executable, "-m", f"diffmatte_cpu.{module}", *map(str, arguments)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        pytest.fail(
            f"{' '.join(command)} が終了コード {completed.returncode} で失敗しました。\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


@pytest.fixture(scope="session")
def e2e_threads() -> int:
    value = int(os.environ.get("DIFFMATTE_E2E_THREADS", "0"))
    return value if value > 0 else max(1, min(8, (os.cpu_count() or 2) // 2))


@pytest.fixture(scope="session")
def e2e_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("DIFFMATTE_E2E_OUTPUT")
    if configured:
        path = Path(configured).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("cpu_convrot_e2e")


@pytest.fixture(scope="session")
def original_onnx() -> Path:
    configured = os.environ.get("DIFFMATTE_ORIGINAL_ONNX")
    candidates = (Path(configured).expanduser(),) if configured else ORIGINAL_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            path = candidate.resolve()
            assert sha256(path) == ORIGINAL_SHA256, f"元 ONNX の SHA-256 が一致しません: {path}"
            return path
    pytest.skip(
        "元の FP32 ONNX がありません。`just fetch-original-onnx` で取得するか、"
        "DIFFMATTE_ORIGINAL_ONNX で場所を指定してください: " + ", ".join(map(str, candidates))
    )
