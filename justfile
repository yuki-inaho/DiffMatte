set shell := ["bash", "-cu"]

original_onnx := "artifacts/models/diffmatte_vits_1024_ddim10_643x960.onnx"
original_onnx_sha256 := "d69827e512803e1fd352e73aff55f327f766564d47de77908bac84d5bbc6acd6"
cpu_model := "artifacts/models/diffmatte_vits_1024_ddim10_643x960_convrot_uint8.onnx"

sync:
    uv sync --extra export --extra dev

# CPU 推論 (ConvRot + UINT8 保存、OpenVINO / ONNX Runtime) を含めて同期します。
sync-cpu:
    uv sync --extra export --extra cpu --extra dev

export-onnx checkpoint output="../diffmatte_onnx/models/diffmatte_vits_1024_ddim10_643x960.onnx":
    uv run diffmatte-export-onnx --config ViTS_1024 --checkpoint "{{checkpoint}}" --output "{{output}}" --height 643 --width 960 --steps 10 --device cpu --verify

validate-onnx checkpoint model image="demo/retriever_rgb.png" trimap="demo/retriever_trimap.png" output_dir="artifacts/validation":
    uv run diffmatte-validate-onnx --config ViTS_1024 --checkpoint "{{checkpoint}}" --model "{{model}}" --image "{{image}}" --trimap "{{trimap}}" --output-dir "{{output_dir}}"

# 元の FP32 ONNX を Release gtx1070-onnx-v0.1.0 から取得し、SHA-256 を照合します。
fetch-original-onnx:
    mkdir -p artifacts/models
    gh release download gtx1070-onnx-v0.1.0 -R yuki-inaho/DiffMatte -p diffmatte_vits_1024_ddim10_643x960.onnx -D artifacts/models --skip-existing
    echo "{{original_onnx_sha256}}  {{original_onnx}}" | sha256sum -c -

# 元 ONNX → onnxsim/ORT BASIC → ConvRot (64) + UINT8 重み保存 (畳み込みは 16 要素組)
convert-cpu input=original_onnx output=cpu_model threads="4":
    uv run --extra cpu diffmatte-cpu-convert --input "{{input}}" --output artifacts/cpu_work/prepared.onnx --variant prepare --threads {{threads}}
    uv run --extra cpu diffmatte-cpu-convert --input artifacts/cpu_work/prepared.onnx --output "{{output}}" --variant weights --prepared --group-size 64 --weight-group-size 16 --linear-activation fp32 --threads {{threads}}

# 推奨: UINT8 保存 / OpenVINO BF16 + matting head FP32。BF16 命令のない CPU では FP32 で実行されます。
infer-cpu image="demo/retriever_rgb.png" trimap="demo/retriever_trimap.png" output_dir="artifacts/cpu_output/retriever" threads="4" *flags="":
    uv run --extra cpu diffmatte-cpu-run-openvino --model "{{cpu_model}}" --image "{{image}}" --trimap "{{trimap}}" --output-dir "{{output_dir}}" --threads {{threads}} --precision bf16 --keep-head-fp32 --seed 42 --warmup 0 --runs 1 {{flags}}

test:
    uv run pytest

# 元 ONNX から CPU 用モデルへ変換し、retriever / bulb を ORT・OpenVINO で推論して元モデルと比較します。
e2e-cpu:
    uv run --extra cpu pytest -m e2e -rs

lint:
    uv run ruff check diffmatte_onnx diffmatte_cpu tests
