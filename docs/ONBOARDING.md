# LLMオンボーディングサマリー

> 新任LLMエージェント向けの初期資料。`yuki-inaho/DiffMatte` の `cu128` checkout と 2026-08-11 のローカル検証に基づく。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** DiffMatte ONNX exporter。自然画像 matting の学習済み PyTorch checkpoint を固定 step DDIM ONNX へ変換する。
- **最終成果物:** Detectron2 非依存 model factory、checkpoint loader、DDIM graph、export CLI、PyTorch/ORT 比較 CLI。
- **ビジネス背景・価値:** backbone 特徴を1回だけ計算し、decoder 反復を単一 ONNX graph で実行できる配布モデルを作る。
- **現時点の進捗サマリ:** ViTS_1024 実 checkpoint から公式デモ形状の ONNX を生成し、checker と数値比較を完了。

## 2. クリティカルな要求・制約

- 対象 branch は `cu128` (`4d119197...`)。upstream 互換修正を保ったまま exporter を追加している。
- export 用 uv 環境は CPU PyTorch 2.7.1 を使う。GPU 世代/CUDA toolkit に export 成否を依存させない。
- checkpoint loader は OmegaConf の既知メタデータ型だけを safe allowlist し、`weights_only=False` にしない。
- DDIM は backbone を1回計算し、同じ features を全 step で再利用する。`initial_noise` は明示入力。
- 実モデルは `N=1/H=643/W=960` 固定。試作資料にあった dynamic batch の主張は実 ONNX metadata と一致しないため採用しない。
- legacy `run_one_image.py` / 学習経路は今回の uv 環境で未検証。新 exporter の確認と混同しない。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | upstream 説明と uv/ONNX export 例 |
| 要件定義書 | `diffmatte_onnx/sampler.py` | 固定 step DDIM と入出力規約 |
| WBS / 進捗 | 未確認 | 専用 WBS はない。本書の完了ゲートを使う |
| テスト資産 | `tests/` | factory、upstream sampler parity、torch.export 捕捉 |
| 比較ツール | `diffmatte_onnx/validate.py` | 同一画像・trimap・noise の PyTorch/ORT 比較 |
| 推論先 | `../diffmatte_onnx/docs/ONBOARDING.md` | ONNX-only runtime と成果物 inventory |
| 既知課題リスト | 本書「制約」 | 固定 shape、legacy/CUDA 未検証 |
| CPU 軽量化 | `docs/CPU_CONVROT.md` | ConvRot + UINT8 保存モデルの変換、OpenVINO/ORT CPU 推論、E2E テスト、実測 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- 対応済み config の model factory/checkpoint compatibility 改善。
- export graph と upstream UniformGauss の数値 parity テスト。
- 新しい H/W・step の ONNX export と同一入力検証。

### 任せないタスク（例）

- ユーザーの `../diffmatte_sandbox` submodule や checkpoint の破壊・移動。
- test を通すために strict checkpoint load を弱めること。
- ONNX checker だけを実画像 E2E 検証として報告すること。
- legacy training/inference の互換性を未検証のまま保証すること。

## 5. インタラクション方針

- **回答スタイル:** 日本語を基本とし、model/config/shape/step/noise/provider/誤差を明記する。
- **回答手順:** branch・重み確認 → unit parity → strict load → export/checker → 実画像比較。
- **禁止事項・注意:** checkpoint を任意 pickle として読み込まない。固定 shape を動的と誤記しない。
- **秘匿情報の扱い:** SSH credential、token、ローカル checkpoint を外部へ送信しない。

## 6. 試行タスク（オンボーディング演習）

1. `uv run pytest` を実行し、6 test（ViTS/Res34/SwinT capture を含む）が成功することを確認する。
2. `load_checkpoint(..., strict=True)` で ViTS_1024 が `missing=[]/unexpected=[]`、29,047,041 parameters になることを確認する。
3. `diffmatte-validate-onnx` を公式デモへ実行し、comparison JSON と3画像と `initial_noise.npy` が揃うことを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 実行済みコマンドと結果を `確認済み` として記載し、未実行項目は `未検証` とする。
- **TBDの扱い:** `未確認` と次の確認先を併記する。
- **レビュー/承認フロー:** 未確認。最低限 unit test、Ruff、strict load、checker、実画像比較を提示する。
- **その他の運用ルール:** exporter report の `dynamic_batch` は ONNX metadata から算出する。生成モデル本体は runtime repo 側で管理する。

### 完了ゲート

- [x] uv lock/sync（Python 3.11.14、torch 2.7.1+cpu）。
- [x] test 6/6、Ruff 成功。
- [x] checkpoint SHA-256 `685fdd...6029` を strict load。
- [x] ONNX SHA-256 `d69827...acd6`、118.9 MB、opset 18、checker 合格。
- [x] export 検証 max `1.52587890625e-05` / mean `4.465637175599113e-07`。
- [x] 公式画像 E2E max `4.589557647705078e-06` / mean `3.0285896457371564e-08`。
- [ ] legacy Detectron2 inference、training、CUDA provider は未検証。

### CPU 軽量化（`diffmatte_cpu/` + extra `cpu`、2026-10-17）

- 同じ uv プロジェクトの optional extra `cpu`（numpy 2.3.5、onnx 1.23.2、ORT 1.31.0、onnxsim 0.8.1、OpenVINO 2026.4.1 固定）。これに合わせ `.python-version` を 3.12、base の numpy を `<3` に緩和。Python 3.12.14 / torch 2.7.1+cpu で既存 test 6 件も成功。
- [x] 元 ONNX `d69827e...acd6` → ConvRot + UINT8 保存 37,398,158 bytes（SHA-256 `175c0545...7d25`）。配布元モデルとグラフ同一、UINT8 要素の 2.4e-6 だけ 1 LSB 差。
- [x] `just e2e-cpu` 8/8（i7-9750H、6 threads、約 2 分 10 秒）。extra なしの環境ではスキップ。未知領域 MAE は retriever 0.000934 / bulb 0.001104（元 FP32 ORT 比）。
- [x] Release `cpu-convrot-uint8-v0.1.0` にモデルと manifest を配置。
- [ ] BF16 実行（AVX-512 BF16 / AMX）はこのマシンで未検証。i7-9750H では BF16 指定が FP32 に戻り、元 ORT 比 1.45 / 1.13 倍。配布元の EPYC 9V74 では 2.46 / 2.13 倍。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** exporter=`.`、runtime=`../diffmatte_onnx`、checkpoint=`../diffmatte_sandbox/checkpoints/DiffMatte_ViTS_Com_1024.pth`。
- **代表的なコマンド:** `uv sync --extra export --extra dev`、`uv run pytest`、README/justfile の export・validate 例。
- **依存ライブラリ:** lock 済み Python 3.11、torch CPU、ONNX、ONNX Script、ONNX Runtime、OmegaConf。
- **連絡先/責任者:** 未確認。remote は `git@github.com:yuki-inaho/DiffMatte.git`。

