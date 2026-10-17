# DiffMatte CPU: ConvRot + UINT8 重み保存

exporter が出力する固定形状 FP32 ONNX（`diffmatte_vits_1024_ddim10_643x960.onnx`、124.72 MB）を、CPU 推論用の 37.40 MB モデルへ変換し、OpenVINO / ONNX Runtime の CPU で実行します。パッケージ `diffmatte_cpu/` と optional extra `cpu` として同じ uv プロジェクトに含めています（Python 3.12、numpy 2.3.5）。

- 変換: onnxsim → ORT BASIC 最適化 → Gemm 分解 → 48 個の線形層へ ConvRot の正則 Hadamard 回転（64 要素組）→ 主な重みを UINT8 保存（畳み込みは 16 要素組の affine、割り切れない重みはチャンネル別）。出力 matting head の重み、正規化・バイアス等は FP32 のまま。
- 推論: 入出力は float32。推奨は **OpenVINO CPU で BF16、matting head（10 ステップ × 2 Conv 等 60 ノード）だけ FP32** の実行。全演算を UINT8 にするモデルではありません。
- 同じネットワーク、DDIM 10 ステップ、入力 `[1,3,643,960]` 固定を維持します。

ConvRot は論文 <https://arxiv.org/abs/2512.03673> の回転を標準 ONNX 演算で適用した独自実装で、公式 CUDA カーネルではありません。コードは ChatGPT で作成した配布物 `DiffMatte_ConvRot_CPU.zip` の `tools/` を、パッケージの相対 import へ書き換えて取り込んだものです（BF16 実行ノード数の記録・警告を追加）。

## セットアップ

```bash
just sync-cpu             # uv sync --extra export --extra cpu --extra dev（cpu extra の版は固定）
just fetch-original-onnx  # Release gtx1070-onnx-v0.1.0 から元 ONNX を取得し SHA-256 を照合
```

元 ONNX は `artifacts/models/` か `../diffmatte_onnx/models/` に置くか、`DIFFMATTE_ORIGINAL_ONNX` で指定します。SHA-256 は `d69827e512803e1fd352e73aff55f327f766564d47de77908bac84d5bbc6acd6` です。

## 変換と推論

```bash
just convert-cpu   # artifacts/models/diffmatte_vits_1024_ddim10_643x960_convrot_uint8.onnx
just infer-cpu     # retriever を OpenVINO BF16 + head FP32 で 1 枚推論
just infer-cpu demo/bulb_rgb.png demo/bulb_trimap.png artifacts/cpu_output/bulb 4 --pad-to-model
```

Release のモデルを使う場合は `gh release download cpu-convrot-uint8-v0.1.0 -R yuki-inaho/DiffMatte -D artifacts/models` で取得し、同梱の `.onnx.json` の SHA-256 と照合してください。

変換済みモデルは Release [`cpu-convrot-uint8-v0.1.0`](https://github.com/yuki-inaho/DiffMatte/releases/tag/cpu-convrot-uint8-v0.1.0) からも取得できます。CLI は次のとおりです。

| コマンド | 内容 |
|---|---|
| `diffmatte-cpu-convert` | `--variant prepare` / `weights`（採用）。実験用に `dynamic` / `static` / `--rewrite-deconv` も残しています（最終モデルには不使用） |
| `diffmatte-cpu-run-openvino` | `--precision bf16 --keep-head-fp32`（推奨）または `--precision f32`（元出力への一致優先） |
| `diffmatte-cpu-run-ort` | ONNX Runtime CPU。圧縮モデルでは `--prepack-float-weights` で起動時に畳み込み重みを FP32 定数へ戻す |
| `diffmatte-cpu-benchmark` | 複数 ONNX を別プロセスで ORT 実行し、速度・一致度を比較 |
| `diffmatte-cpu-prepare-inputs` | 固定入力の NPZ を作成 |

出力は `alpha.npy`（丸め前 float32）、`alpha16.png`、`timing.json`（実行条件・時間・メモリ・実際の演算精度）です。入力より小さい画像は `--pad-to-model` で右・下を端の画素で埋め、出力を元寸法へ戻します。大きい画像はエラーです。

### BF16 は CPU 依存

`--keep-head-fp32` は OpenVINO **2026.4.1** の内部属性を使うため、版を固定しています（版が違えば停止します）。BF16 の高速化には AVX-512 BF16 / AMX が必要です。**BF16 命令のない CPU では、OpenVINO は BF16 指定を黙って FP32 に戻します。** そのため `run_openvino` は `timing.json` に `bf16_runtime_node_count` と `optimization_capabilities` を記録し、BF16 を指定したのに BF16 ノードが 0 件なら標準エラーへ警告を出します（取り込み時に追加）。

## 実測（2026-10-17）

同じ画像・トライマップ・初期乱数（seed 42）、モデルごとに別プロセスで順に実行し、warm-up 1 回後の同期推論 3 回の中央値です。読み込み・コンパイル・前処理・保存は含みません。一致度は**元 FP32 ONNX の出力との差**（トライマップ未知領域、0〜1 のアルファ）で、正解に対する精度ではありません。

### このリポジトリでの追試: Intel Core i7-9750H（6C/12T、AVX2、BF16 非対応）、6 threads

| 実行方法 | retriever | bulb | 元 ORT 比 | RSS 最大 |
|---|---:|---:|---:|---:|
| 元 FP32 / ORT CPU | 16.35 秒 | 12.74 秒 | 1.00 / 1.00 | 3.27 GB |
| UINT8 保存 / ORT（prepack） | 15.18 秒 | 13.57 秒 | 1.08 / 0.94 | 3.56 GB |
| UINT8 保存 / OpenVINO FP32 | 11.47 秒 | 10.58 秒 | 1.43 / 1.20 | 0.92 GB |
| **UINT8 保存 / OpenVINO BF16 ＋ head FP32** | **11.31 秒** | **11.25 秒** | **1.45 / 1.13** | **0.93 GB** |

この CPU では推奨設定も BF16 ノード 0 件（実体は FP32）で、出力は OpenVINO FP32 とビット単位で一致しました。

| 設定 | 画像 | MAE | RMSE | 絶対差 P99 | 最大 |
|---|---|---:|---:|---:|---:|
| OpenVINO（推奨 = 実質 FP32） | retriever | 0.000934 | 0.001815 | 0.006976 | 0.0366 |
| OpenVINO（推奨 = 実質 FP32） | bulb | 0.001104 | 0.002873 | 0.013071 | 0.0891 |
| ORT（prepack） | retriever | 0.000979 | 0.001885 | 0.007213 | 0.0409 |
| ORT（prepack） | bulb | 0.001108 | 0.002884 | 0.013150 | 0.0926 |

### 配布元の実測: AMD EPYC 9V74（AVX-512 BF16）、8 threads

| 実行方法 | retriever | bulb | 元 ORT 比 |
|---|---:|---:|---:|
| 元 FP32 / ORT CPU | 7.242 秒 | 6.493 秒 | 1.00 / 1.00 |
| UINT8 保存 / OpenVINO FP32 | 5.921 秒 | 6.499 秒 | 1.22 / 1.00 |
| **UINT8 保存 / OpenVINO BF16 ＋ head FP32** | **2.948 秒** | **3.053 秒** | **2.46 / 2.13** |

推奨設定（BF16 実行）の未知領域 MAE は 0.001452 / 0.001648、最大差 0.0625 / 0.1155 でした。BF16 による 2 倍超の高速化は BF16 対応 CPU でのみ得られます。

### 再現性

- prepare 後のモデルは配布元の記録とバイト数・演算構成が一致。
- 最終モデルのグラフ（1,846 ノード）は配布モデルと同一。UINT8 重み 30,867,776 要素のうち 75 要素（2.4e-6）だけが 1 LSB 異なり、すべて ConvRot の回転重みでした。回転行列積の浮動小数点の丸め差（AVX-512 と AVX2）によるものです。同じマシンでは変換は決定的です。
- 元 ORT 出力は配布元の元 ORT 出力と最大差 1.8e-6、OpenVINO FP32 の一致度指標も配布元と 1e-7 以内で一致しました。

## テスト

```bash
just test      # 既存テスト + CPU 単体（ConvRot の恒等性、UINT8 量子化の往復誤差）。e2e は除外
just e2e-cpu   # E2E: 元 ONNX → 変換 → retriever/bulb × {OpenVINO 推奨, OpenVINO FP32, ORT prepack} を元 ORT と比較
just lint
```

`cpu` extra が入っていない環境では、CPU の単体テスト・E2E は `importorskip` でスキップされます。E2E（`tests/test_cpu_e2e.py`、マーカー `e2e`、既定の `pytest` では除外）が確認する内容は次のとおりです。

1. 変換: `onnx.checker`、入出力名・固定形状・float32、元の 32% 以下の容量、回転 48 層・DequantizeLinear 106 個、matting head の重みが FP32 のまま、メタデータ。
2. 同じ入力: 各方式の入力テンソルの SHA-256 が元 ORT 実行と一致。bulb は padding と元寸法への切り戻し。
3. 一致度: 未知領域の MAE / RMSE / P99 / 最大差が上限以内（配布時の実測に約 2 倍の余裕）。
4. 推奨設定: head 60 ノードがコンパイル後も FP32。CPU が BF16 対応なら BF16 ノード > 0、非対応なら 0 であること（FP32 フォールバックを記録と照合）。

環境変数: `DIFFMATTE_ORIGINAL_ONNX`（元 ONNX）、`DIFFMATTE_E2E_THREADS`（既定は論理 CPU の半分、最大 8）、`DIFFMATTE_E2E_OUTPUT`（出力を残す場所。既定は pytest の一時ディレクトリ）、`DIFFMATTE_CONVROT_REFERENCE`（配布済みモデルと照合する場合に指定）。i7-9750H・6 threads で約 2 分 10 秒です。速度は環境依存のため判定せず、各結果の `e2e_result.json` に記録します。

## 対象外

任意解像度、蒸留・再学習によるネットワーク小型化、DDIM ステップ削減、配布物にあった C++ ランナー（ORT C API）とレポート生成。C++ ランナーは配布元の比較で Python → ORT より速くならなかったため（6.538 秒 / 7.073 秒）取り込んでいません。
