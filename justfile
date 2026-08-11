set shell := ["bash", "-cu"]

sync:
    uv sync --extra export --extra dev

export-onnx checkpoint output="../diffmatte_onnx/models/diffmatte_vits_1024_ddim10_643x960.onnx":
    uv run diffmatte-export-onnx --config ViTS_1024 --checkpoint "{{checkpoint}}" --output "{{output}}" --height 643 --width 960 --steps 10 --device cpu --verify

validate-onnx checkpoint model image="demo/retriever_rgb.png" trimap="demo/retriever_trimap.png" output_dir="artifacts/validation":
    uv run diffmatte-validate-onnx --config ViTS_1024 --checkpoint "{{checkpoint}}" --model "{{model}}" --image "{{image}}" --trimap "{{trimap}}" --output-dir "{{output_dir}}"

test:
    uv run pytest

lint:
    uv run ruff check diffmatte_onnx tests
