# Stable Diffusion WebUI Forge - NeoW 日本語README

## まずこれだけ: Krea2 bnb-nf4一括セットアップ

初学者向けに、Krea2 用のモデル、text encoder、VAE を自動でダウンロードして Forge 標準フォルダへ配置するスクリプトを用意しています。

Windowsで次をダブルクリックします。

```text
download_krea2_bnb_nf4_models.bat
```

ダウンロードするもの:

```text
models/Stable-diffusion/Krea2_Center NF4 - Krea2.safetensors
models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors
models/VAE/qwen_image_vae.safetensors
```

合計で約12.27GiBを Hugging Face からダウンロードします。途中で止まった場合は、同じbatをもう一度ダブルクリックすると `.part` から再開します。既に正しいサイズと safetensors ヘッダーのファイルがある場合は再ダウンロードしません。

完了したら次をダブルクリックして起動します。

```text
webui-user.bat
```

Forge UIでは以下を選びます。

```text
Preset: krea
Checkpoint: Krea2_Center NF4 - Krea2.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen3vl_4b_fp8_scaled.safetensors
Diffusion in Low Bits: bnb-nf4
```

選択後はそのまま `txt2img` で生成できます。まずは以下の初期値から試します。

```text
Sampler: Euler
Scheduler: Simple
Steps: 8
CFG scale: 1.0
Distilled CFG: 1.15
Size: 768x1152
```

このリポジトリは、`Stable Diffusion WebUI Forge - Neo` をベースにした作業forkです。

主目的は、Forge Neoの最新系を使いながら、Krea2、低ビット量子化モデル、巨大解像度のimg2img/upscale、モデルマージまわりを実用しやすくすることです。

## このforkの現在地

- ベース: `Haoming02/sd-webui-forge-classic` の `neo` 系
- 作業ブランチ: `neo`
- WebUI: Forge Neo / Gradio 4系
- 目的: Krea2、低ビット量子化、高解像度img2img/upscaleを扱いやすくする

## このforkで追加・反映した主な変更

### Krea2 bnb-nf4モデル検出

Krea2のpre-quantized bitsandbytes safetensorsでは、通常の `tensor.shape` だけを見ると元の重みshapeではなく、量子化後のstorage shapeを拾ってしまうことがあります。

このforkでは、`modules_forge/packages/huggingface_guess/detection.py` に `tensor_shape(...)` を追加し、bitsandbytesの `QuantState` から元shapeを復元してKrea2設定検出に使います。

これにより、Krea2のbnb-nf4形式のpre-quantizedモデルをForgeが正しく検出できます。

対象の修正:

```text
modules_forge/packages/huggingface_guess/detection.py
```

### Krea2 4bit運用

このforkではKrea2を次の構成で使うことを想定しています。

```text
Preset: krea
Diffusion model: Krea2系 bnb-nf4 safetensors
Text Encoder: Krea2対応text encoder
VAE: Krea2対応VAE
Diffusion in Low Bits: bnb-nf4
```

Forge上では、Krea2の追加モジュールとして以下の2つを選びます。

```text
Krea2対応text encoder
Krea2対応VAE
```

生成infotextでは、おおむね次のように記録されます。

```text
Model: Krea2 bnb-nf4 model
Module 1: Krea2 VAE
Module 2: Krea2 text encoder
Diffusion in Low Bits: bnb-nf4
```

### Krea2 8K img2img helper

8K級の画像は、単純な `txt2img` の直接8192px生成ではVRAMを使い切りやすく、Tiled VAEだけでは安定しません。

このforkでは、Krea2向けに次の流れで8K化するhelperを追加しています。

```text
元画像
→ Lanczosで長辺8192へ拡大
→ img2img
→ MultiDiffusion Integratedで拡散処理をタイル化
→ Forge側で保存
```

追加スクリプト:

```text
tools/krea2_8k_img2img.py
```

既定の実行:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>'
```

安全確認だけ行うdry-run:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --dry-run
```

既定値:

```text
Long edge: 8192
Sampler: DPM++ SDE
Scheduler: Simple
Steps: 12
CFG: 1.0
Distilled CFG: 1.15
Denoising strength: 0.28
MultiDiffusion: enabled
Tile: 768x768
Overlap: 96
Tile batch size: 1
send_images: false
save_images: true
```

正方形8Kを明示する場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --width 8192 --height 8192
```

ただし `8192x8192` は約67MPで、縦長の `5440x8192` よりかなり重いです。まずは長辺8192の比率維持を推奨します。

### Tiled VAE / tiled Conv2d

ForgeのVAE decodeは、通常は `Full` で実行され、VAE decode中にOOMした場合にTiled VAEへフォールバックします。

重要な点:

```text
Tiled VAEはVAE encode/decodeのメモリ対策
拡散ステップ本体のメモリ対策ではない
8Kを安定させるにはMultiDiffusionなどで拡散自体をタイル化する必要がある
```

このforkの `webui-user.bat` では、VAE系のConv2dをタイル化するために次を追加しています。

```text
--tiled-conv2d 128
```

これは高解像度VAE処理のメモリを下げる目的です。速度は落ちます。

Tiled VAEのタイル合成では、境界フェザーをsmoothstep重みに変更し、同じ形のマスクを再利用するようにしています。分母マップは単一チャンネルで保持し、継ぎ目を増やさずタイルごとの余分なメモリ確保を減らします。

### ModelMerger / Extras / Upscaler系の最新反映

fork側の最新コミットとして、以下の更新も取り込んでいます。

```text
backend/diffusion_engine/base.py
backend/diffusion_engine/sd15.py
backend/diffusion_engine/sdxl.py
modules/extras.py
modules/ui_checkpoint_merger.py
modules/upscaler_utils.py
```

主にModelMerger、Extras、アップスケール処理、SD1/SDXL系処理まわりの更新です。

## 推奨起動

Windowsでは通常のForge起動を使います。

```powershell
.\webui-user.bat
```

`webui-user.bat` では、必要に応じて以下のような引数を設定します。

```text
--uv
--bnb
--api
--port 7861
--theme dark
--tiled-conv2d 128
--forge-ref-comfy-yaml <path-to-model-paths-yaml>
--esrgan-models-path <path-to-upscaler-models>
```

`--forge-ref-comfy-yaml` と `--esrgan-models-path` は環境に合わせて変更してください。

## Krea2の基本設定

Forge UIで以下を確認します。

```text
Preset: krea
Checkpoint: Krea2 diffusion model
VAE / Text Encoder:
  Krea2 compatible VAE
  Krea2 compatible text encoder
Diffusion in Low Bits: bnb-nf4
```

Krea2 t2iの実用初期値:

```text
Sampler: Euler
Scheduler: Simple
Steps: 8
CFG scale: 1.0
Distilled CFG: 1.15
```

Krea2 i2i / upscaleでノイズを抑える初期値:

```text
Sampler: DPM++ SDE
Scheduler: Simple
Steps: 12-15
CFG scale: 1.0
Distilled CFG: 1.15
Denoising strength: 0.27-0.30
```

元絵保持を優先する場合:

```text
Sampler: Euler
Scheduler: Simple
Steps: 15-30
Denoising strength: 0.27-0.30
```

避ける設定:

```text
Denoising strength 0.6以上
8Kの直接txt2img
VAEタイル化だけに頼った巨大解像度生成
```

## Krea2 t2i API例

Forge APIが `127.0.0.1:7861` で起動している前提です。

```python
import requests

payload = {
    "prompt": "your prompt here",
    "negative_prompt": "",
    "seed": -1,
    "sampler_name": "Euler",
    "scheduler": "Simple",
    "steps": 8,
    "cfg_scale": 1.0,
    "distilled_cfg_scale": 1.15,
    "width": 768,
    "height": 1152,
    "send_images": True,
    "save_images": True,
}

response = requests.post("http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload, timeout=1200)
response.raise_for_status()
```

`CFG scale: 1.0` の場合、Forgeではnegative promptが無視されます。negative promptを効かせたい場合は `CFG scale` を上げてください。

## Krea2 8K helperの使い方

入力画像を明示して実行します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --dry-run
```

問題なければ本番:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>'
```

より保守的にする場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --denoise 0.24 --steps 10
```

より描き直す場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --denoise 0.33 --steps 12
```

## 代表的なコマンドライン引数

このforkで特によく使うもの:

```text
--api
  WebUI APIを有効化

--port 7861
  API/UIのポートを7861に固定

--uv
  uv経由で依存関係を高速インストール

--bnb
  bitsandbytesを有効化し、nf4などの低ビット推論に対応

--tiled-conv2d 128
  VAEなどのConv2d演算をタイル化してVRAM使用量を下げる

--forge-ref-comfy-yaml
  ComfyUI系のモデルパス設定をForgeから参照

--esrgan-models-path
  upscalerモデルの置き場所を指定
```

## インストール

通常のfork利用:

```bash
git clone <repository-url> --branch neo
cd <repository-directory>
```

Windowsでは `webui-user.bat` を編集し、必要ならモデルパスを自分の環境に合わせます。

初回起動:

```powershell
.\webui-user.bat
```

`uv` を使う場合は、事前にuvを入れておきます。

```powershell
uv venv venv --python 3.13 --seed
```

## 対応モデル概要

Neo系の画像生成モデル、編集モデル、低ビット量子化モデルに追従しています。

このforkでは特にKrea2のbnb-nf4運用を重視しています。

## 注意点

- モデル、VAE、text encoderはリポジトリに含めません。
- `logs/`、`output/`、ローカルYAML、生成画像は通常push対象外です。
- `webui-user.bat` にはローカル環境向けパスが含まれる場合があります。
- 巨大解像度は非常に時間がかかります。
- 8Kはまずdry-runで入力サイズとリクエスト内容を確認してください。
- 8K生成中にVRAMが張り付いた場合は `/sdapi/v1/interrupt`、`/sdapi/v1/skip`、`/sdapi/v1/unload-checkpoint` の順で止めます。

## トラブルシュート

### text encoderの読み込みで失敗する

Krea2対応ではないtext encoderを選んでいる可能性があります。

Krea2用のtext encoderを選びます。

### Krea2 bnb-nf4でshape mismatchする

このforkの `detection.py` 修正が入っているか確認してください。

```text
def tensor_shape(...)
QuantState.from_dict(...)
```

が入っていれば、bitsandbytesの元shape復元に対応しています。

### 8Kで止まる、またはVRAMが張り付く

直接t2iで巨大解像度を作らず、`tools/krea2_8k_img2img.py` を使います。

まずdry-run:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --dry-run
```

問題なければ本番:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>'
```

### white speckle / 粒状ノイズが乗る

i2i upscaleではdenoiseが高すぎる可能性があります。

まず以下を試します。

```text
Sampler: DPM++ SDE
Scheduler: Simple
Steps: 12-15
Denoising strength: 0.27-0.30
```

`0.35` 以上は再描画が強くなり、肌や背景に粉っぽい再描画が出る場合があります。

生成後の画像だけを補正する場合は、単体ツールで小さな粒状ノイズを検出して補修できます。

まずマスクと確認用プレビューだけ出す場合:

```powershell
.\venv\Scripts\python.exe .\tools\despeckle_image.py --input '<image>' --mode mask --mask-out '<mask.png>' --preview-out '<preview.png>'
```

周辺画素で埋める場合:

```powershell
.\venv\Scripts\python.exe .\tools\despeckle_image.py --input '<image>' --output '<fixed.png>' --mode local-inpaint
```

検出した小領域だけForgeで再生成する場合:

```powershell
.\venv\Scripts\python.exe .\tools\despeckle_image.py --input '<image>' --output '<fixed.png>' --mode forge-inpaint --denoise 0.24 --steps 8
```

白い粒が残る場合は `--threshold 24`、実線や装飾まで削れる場合は `--threshold 40 --max-area 24` から調整します。暗い粒も対象にする場合は `--polarity both` を使います。

## upstreamへの敬意

このforkは以下のプロジェクトをベースにしています。

- AUTOMATIC1111 Stable Diffusion WebUI
- lllyasviel Stable Diffusion WebUI Forge
- Haoming02 sd-webui-forge-classic / Neo
- ComfyUI系バックエンド実装

元プロジェクトと関連コントリビューターに感謝します。
