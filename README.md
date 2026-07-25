# Stable Diffusion WebUI Forge - NeoW 日本語README

## まずこれだけ: Krea2 INT8 ConvRot一括セットアップ

初学者向けに、Krea2 用のモデル、text encoder、VAE を自動でダウンロードして Forge 標準フォルダへ配置するスクリプトを用意しています。

Windowsで次をダブルクリックします。

```text
download_krea2_int8_convrot_models.bat
```

ダウンロードするもの:

```text
models/Stable-diffusion/krea2_turbo_int8_convrot.safetensors
models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors
models/VAE/qwen_image_vae.safetensors
```

合計で約17.69GiBを Hugging Face からダウンロードします。途中で止まった場合は、同じbatをもう一度ダブルクリックすると `.part` から再開します。配布元の更新でURLや内容が変わらないよう、各URLは実在を確認したimmutable revisionへ固定しています。既に正しいサイズ、SHA-256、safetensorsヘッダーのファイルがある場合は再ダウンロードしません。

完了したら次をダブルクリックして起動します。

```text
webui-user.bat
```

Forge UIでは以下を選びます。

```text
Preset: krea
Checkpoint: krea2_turbo_int8_convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen3vl_4b_fp8_scaled.safetensors
Diffusion in Low Bits: Automatic
```

Kreaプリセットの新規設定もこのcheckpoint、追加モジュール、`Automatic`を初期値にします。既存の`config.json`に保存済みの選択がある場合は、そのユーザー設定を維持します。選択後はそのまま `txt2img` で生成できます。まずは以下の初期値から試します。

```text
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 4
CFG scale: 1.0
Model shift: 1.15 (Krea2 fixed default)
Size: 1024x1024
```

このリポジトリは、`Stable Diffusion WebUI Forge - Neo` をベースにした作業forkです。

主目的は、Forge Neoの最新系を使いながら、Krea2、低ビット量子化モデル、巨大解像度のimg2img/upscale、モデルマージまわりを実用しやすくすることです。

## このforkの現在地

- ベース: `Haoming02/sd-webui-forge-classic` の `neo` 系
- 取り込み済みupstream: `origin/neo` の `120b01fc`（2026-07-25、tag `2.27` から41 commit後）
- 作業ブランチ: `neo`
- WebUI: Forge Neo / Gradio 4系
- 目的: Krea2、低ビット量子化、高解像度img2img/upscaleを扱いやすくする

## 最新upstream `neo` との差

このブランチは `Haoming02/sd-webui-forge-classic` の `neo`、commit `120b01fc` をmerge済みです。以下の「最新upstream」はこのcommitを指し、将来のupstream更新を自動で含むという意味ではありません。

| 分類 | 最新upstream `neo` の基盤 | このforkで追加している差分 |
|---|---|---|
| Krea2・量子化 | Krea2 Turbo / Raw、`int8_convrot`、`convrot_w4a4` | 固定revision downloader、BF16からのstreaming INT8 ConvRot変換、bnb-NF4元shape検出、Qwen3-VL入力検証、checkpoint・VAE・text encoderのruntime preflight、Krea2 presetと安全な既定値 |
| 高解像度workflow | PiD Integrated、tiled Conv2d、標準img2img/upscale | Smart 4K/8K、2-stage upscale、VRAM-Canvas、PhaseWeave / DetailWeave、Local Supersample、Focused ROI、subject/tile refine、B5 whole-tile regeneration、approved-reference Identity Guard |
| 生成的4K/8K処理 | upstream標準の生成・upscale経路 | built-in ExtensionのHyperWeave 4K/8KとProofWeave基盤。座標整合ノイズ、候補制約、周波数帯別合成、領域別pass、比較・品質gateを追加 |
| 品質管理・後処理 | Extras、標準postprocessing / upscaler | Color Flatten / Smooth Gradient、chroma-only色ムラ解析、孤立粒補修、targeted white-speck regeneration、PNG metadata・manifest・QA・比較tool |
| Forge連携・検証 | 最新Checkpoint Merger、Refiner、img2img / inpaint / mask処理 | model runtime status API、selectable script args overlay、Krea2 conditioning cache、Tiled VAEのsmoothstep合成・mask再利用・分母map削減、独自機能のテストと日本語技術資料 |

今回の同期では、upstream側のAnima Edit、PiD Integrated / PiD 1.5、Anima Region ControlNet、Checkpoint Merger rewrite、Refiner CFG / Refiner LoRA、img2img / inpaint / mask更新、Krea・Ernie・Lumina系の高速化、`comfy-kitchen==0.2.22`などの依存更新も取り込んでいます。これらはupstream由来であり、このfork独自機能としては扱いません。

HyperWeaveの設計と制約は [built-in Extension README](extensions-builtin/hyperweave/README.md)、PhaseWeave / DetailWeaveは [実装・実画像比較ノート](docs/krea2_phaseweave_4k_ja.md) と [B5短報](docs/detailweave_4k_b5_ja.pdf) を参照してください。

## このforkで追加・反映した主な変更

### Krea2 bnb-nf4モデル検出

Krea2のpre-quantized bitsandbytes safetensorsでは、通常の `tensor.shape` だけを見ると元の重みshapeではなく、量子化後のstorage shapeを拾ってしまうことがあります。

このforkでは、`modules_forge/packages/huggingface_guess/detection.py` に `tensor_shape(...)` を追加し、bitsandbytesの `QuantState` から元shapeを復元してKrea2設定検出に使います。

これにより、Krea2のbnb-nf4形式のpre-quantizedモデルをForgeが正しく検出できます。

対象の修正:

```text
modules_forge/packages/huggingface_guess/detection.py
```

### Krea2 native INT8 tensorwise + ConvRot

Forge Neoは、`comfy-kitchen`の`TensorWiseINT8Layout`を使うpre-quantized `int8_tensorwise` checkpointと、層ごとのConvRot metadata（group size 256）をネイティブにロードします。INT8ではactivationを別形式へ量子化せず、BF16 activation × ConvRot INT8 weightのweight-only kernelを使います。

既定checkpointはComfy-Org配布の`krea2_turbo_int8_convrot.safetensors`です。一括セットアップでは[Comfy-Org/Krea-2の固定revision](https://huggingface.co/Comfy-Org/Krea-2/tree/8038ce89b91b042141541ad0fa51b985ca262c5f/diffusion_models)から取得し、サイズ、SHA-256、INT8 ConvRot companion tensorを検証します。pre-quantized INT8のため、Forge側で追加の量子化をかけず`Diffusion in Low Bits: Automatic`でロードします。

同梱の変換CLIは、Krea2の28ブロック×8 projection＝224 weightだけを公式profileどおりINT8 ConvRot化し、`first`、`last`、`tmlp`、`tproj`、`txtfusion`、`txtmlp`などのsensitive layerを元精度のまま保持します。入力は必ずBF16 merged checkpointを指定してください。NF4や既量子化checkpointからの再量子化と、既存出力への上書きは拒否します。

```powershell
.\venv\Scripts\python.exe tools\convert_krea2_int8_convrot.py `
  "models\diffusion_models\krea2_custom_bf16_merged.safetensors" `
  "models\diffusion_models\krea2_custom_int8_convrot.safetensors"
```

変換は一時`.part`へ書き出し、全tensor shape/dtype、224個のscale、ConvRot metadataを再検証してから確定名へ移します。Forgeでは生成済みcheckpointを選び、`Diffusion in Low Bits`は`Automatic`にします。Krea2のtext encoderには公式`qwen3vl_4b_fp8_scaled.safetensors`、VAEにはQwen Image VAEを指定します。

### Krea2 bnb-nf4運用（任意）

INT8 ConvRotではなく、既存のbnb-nf4 checkpointを使う場合は次の構成に手動で切り替えます。

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

### Krea2 Smart 4K / high-resolution helper

Krea 2公式の解像度範囲は、Rawが最大1K、Turboが1K〜2Kです。native 4Kは現在のopen weightsの対象外で、公式も「最大2Kで生成し、4K以上はEnhancer/upscalerへ送る」構成を案内しています。

- [Krea 2公式GitHub Usage](https://github.com/krea-ai/krea-2#usage)
- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
- [Krea公式のnative 2K / 4K以上はEnhancerという説明](https://www.krea.ai/blog/krea-2-vs-lumion-enscape-d5-render)

このforkでは、4Kを直接拡散せず次の流れで作ります。

```text
元画像
→ 必要な場合だけ中間img2img
→ Rawは最大1K、Turbo/customは保守的に最大2Kのproxyでimg2img
→ MultiDiffusion Integratedでproxyの拡散をタイル化
→ Lanczosで指定した正確な納品寸法へ拡大
→ chroma-only色ムラ解析
→ 指標が改善する場合だけLab a/bを限定補正
→ PNG metadata・quality report・実seedを保存
```

追加スクリプト:

```text
tools/krea2_8k_img2img.py
scripts/krea2_2stage_upscale.py
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
Model profile: custom (conservative 2K proxy)
Upscale mode: auto
Long edge: 4096
Stage 1 long edge: auto
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 8
CFG: 1.0
Model shift: 1.15 (Krea2 fixed default)
Stage 1 denoising strength: 0.10
Final denoising strength: 0.12
Final diffusion long edge cap: profile auto (custom resolves to 2048)
No-progress timeout: 600
MultiDiffusion: enabled
Tile: 768x768
Overlap: 96
Tile batch size: 1
Smart chroma finish: enabled, improvement-gated
Isolated speckle repair: disabled (snow/stars/freckles protection)
API images: returned and validated locally
Forge-side raw save: disabled; validated final PNG is saved locally
```

各runは重複しないtimestamp directoryへ保存し、`krea2_highres.png`、`final_diffusion_img2img.png`、`quality_report.json`、`run_manifest.json`を残します。納品PNGの`parameters`内`Size`はproxyではなく実寸へ更新され、Smart Finish reportもPNGへ埋め込みます。

UHD 4Kを正確に指定する場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --width 3840 --height 2160
```

最終納品寸法は丸めません。拡散proxyだけを16px単位へ合わせるため、`3840x2160` はそのまま `3840x2160` で保存されます。

正方形4K:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --width 4096 --height 4096
```

公式Rawを使う場合はprofileを明示します。拡散proxyは自動的に最大1024pxになります。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --model-profile raw
```

`--model-profile` は解像度guardだけを切り替えます。RawとTurboでは公式sampler/CFG/scheduleが異なるため、sampling設定を自動混在させません。Rawではcheckpointのmodel cardに従って明示設定してください。このfork既定の8 steps / CFG 1.0 / shift 1.15はTurbo/custom向けです。

2048pxを超える拡散は、VRAM上動く場合でもKrea2のnative範囲外です。試す場合は `--allow-non-native-diffusion` を明示します。3584pxを超える場合はRTX 3090 24GB向けhardware guardも越えるため、さらに `--allow-unsafe-large-diffusion` が必要です。

入力がすでにproxy解像度に近い場合、既定の `--upscale-mode auto` は中間passを省きます。単段を常に使う場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --upscale-mode single-stage
```

入力自体が選択profileのproxy上限より大きい場合は、画質を落とす暗黙downscaleを行わず開始前に停止します。既存4Kを再生成せず整えるだけなら `krea2_smart_finish.py`、局所再描画なら `krea2_tiled_refine.py` / `krea2_subject_refine.py` を使ってください。

孤立した白/黒粒が本当にartifactだと分かっている画像だけ、次を追加します。雪、星、そばかす、粒子、意図したgrainがある画像では有効にしません。変更候補が画像の0.35%を超えると自動的に補修を拒否します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --smart-despeckle
```

生成済み画像だけをSmart Finishする単体CLI:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_smart_finish.py --input '<image>'
```

### AI画像の微細模様を滑らかなグラデーションへ戻す

Smart Finishは意図的にLab a/bだけを補正するため、輝度を含む全面の細かな網目・ざらつきは対象外です。この種類のartifactには、Forgeの `Extras` → `Color Flatten / 色ムラ補正` で次を選びます。

```text
Mode: Smooth Gradient / AI Noise
Strength: 1.0
Edge Protect: ON
Smooth Gradient Radius: 12 px
Smooth Gradient Detail Threshold: 8 Lab ΔE
孤立した白/黒粒を補修: OFF
```

約1K画像ではRadius `5〜6` が通常、`10〜12` がクリスタのグラデーションに近い強い平滑化の目安です。より大きい画像では、ノイズ模様の見かけの大きさに合わせてRadiusを増やします。Detail Thresholdは大きいほど低コントラストの模様まで除去し、小さいほど細部を保護します。

処理本体はLabのL/a/bすべてから滑らかな低周波面を作り、強い線・色境界・透明境界では元画素へ戻します。人物、髪、布、紙目、フィルム粒子などの低コントラストな質感とAIノイズは一枚の画像だけでは完全に区別できないため、必要な画像だけで選択し、まずEdge ProtectをONにして確認してください。平坦な背景だけを確実に均す場合はEdge ProtectをOFFにできます。

さらに大きい画像を局所的に再描画する場合は、全体を一発で拡散せず `tools/krea2_tiled_refine.py` を使います。各tileは開始時に確定した同じseedを使い、smoothstep featherの前に元cropへ低周波Lab a/bを合わせるため、tileごとの色castを抑えます。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_tiled_refine.py --input '<2x-output-image>' --long-edge 5760 --tile-size 1024 --overlap 192 --denoise 0.08 --steps 4
```

### VRAM-Canvas: VRAM予算固定の4K/8K段階リファイン

VRAM-Canvasは、納品キャンバス全体をGPUへ載せず、実際に4K/8K解像度の局所ディテールを追加する実験実装です。単純なtile画像の貼り合わせではありません。GUI版 `scripts/vram_canvas_highres.py` とCLI版 `tools/vram_canvas_highres.py` は同じ計画・周波数合成ロジックを使います。

```text
入力promptをprefixとして保持し、人物の顔・髪・衣装や建築・自然物など、入力に実在する材質だけを精密化した全体基準像
→ 各段階の拡大率が2倍以下になるようprogressive resize
→ VRAM予算から384～1280pxの処理tileを自動決定
→ coreの外側へhaloを付け、1 tileずつForge img2img
→ 生成tileと基準cropの高周波差分だけを抽出
→ 低周波構造差で各候補をstructure gate
→ 既存細部用safe残差と、2～8px帯域・輝度限定のbounded novel-detail残差を分離
→ phase内のsmoothstep重みを正規化
→ PhaseWeaveでは等間隔格子を画像外まで延長し、全phase・全tileを同じ入力寸法に保ったまま、端の最小寄与幅が最大になる共通起点を選択
→ 既定のconsensus mergeは2 phaseの一次・二次momentからagreement gate付き平均を作成
→ Krea2 PhaseWeave 4Kは各phaseを独立正規化し、低周波を入力へ寄せたうえで高周波量と入力忠実度を評価し、A・B・入力維持をしきい値±3%で三値選択
→ 選択得点を入力輪郭に沿わせ、3000px未満の選択島を整理して確定境界だけを5pxで接続。補助候補は位置と構造が揃う場合だけ最大10%、代表detailは0.90以上を保持
→ novel残差も選択したmerge方式を通し、輝度限定・bounded detailとして合成
→ Smart Finishは色補正を既定OFFとし、必要時だけstructure tensorで整合した既存detailを最大±5増強
→ 8K用float32 accumulatorはdisk memmapへ退避
```

GPU側の空間メモリ項は目標の `W x H` ではなく処理tileの `T x T` で上限が決まります。8K正方形を1024角tileで処理する場合、モデル重みを除く空間活性量は全画面処理の理論上 `1/64` です。これはnative 8K txt2imgではなく、全体構造を基準像へ固定した段階的img2img detail refinementです。

#### Forge GUIから使う

Forgeを再起動し、次の順で選びます。GUI経路はForge内部の処理を直接呼ぶため、`--api` は不要です。

```text
img2img
→ 通常のimg2imgへ入力画像とpromptを設定
→ Script: VRAM-Canvas 4K/8K Highres
→ 4K Smart - long edge 4096 + profile
→ Total VRAM Budget GiB: 0（GPUから自動取得）
→ Diffusion Tile Edge: 0（予算から自動計算）
→ Generate
```

`4K Smart` buttonは4096長辺、Krea2 Dense Detail 4K profile、任意画像向けのgeometry-preserving guidanceを一括適用します。`Krea2 PhaseWeave 4K` buttonは同じ4096長辺に対し `Krea2 PhaseWeave 4K (Experimental)` / `phaseweave_4k` profileと `phase_weave` mergeを適用します。guidanceは人物の同一性・顔・手指・文字・物体数・構図を固定し、入力に実在する髪、虹彩、布、木、石、植生、透明物、線画などだけを材質と画風に応じて精密化します。写真風の毛穴をアニメやflat-color領域へ強制しません。profile dropdownを選ぶと対応sliderへ即時反映され、`Apply Profile` でも同じ値を再適用できます。4Kを目視確認した後、その4Kをimg2img入力へ入れ直し、`8K Smart - exact 2x approved 4K + profile` を押すと、1024 tile・正確な各辺2倍で8Kへ進めます。通常の品質profileは2 phases、3～4 adaptive steps、4K用denoise 0.16→0.13、8K用0.12→0.11、detail gain 1.25、novel detail 1.0/0.8、最大差±8/±6をまとめて設定します。PhaseWeave profileは2 phases、各tile 6 Exact Steps、denoise 0.20→0.16、detail gain 1.55、novel detail 1.4に固定します。Smart Finishは既定ON、入力の意図した色を守るためSmart Chromaは既定0です。coherent detail guardは平坦部・強輪郭・clipを保護します。

GUIでもBatch Count / Batch Sizeは1、通常img2imgのみです。inpaint maskには対応しません。tileごとのCodeFormer/GFPGANによるidentity変化を避けるため、`Restore faces` とwrap-around `Tiling` は内部tile処理中だけ強制OFFにします。内部tileの間だけrequest-local overrideで `img2img_fix_steps=true` を有効にし、成功・例外・中断・skip・stop・OOMの全経路でprocessing objectと通常設定を元へ戻します。nested callでも受け取ったoverride objectを復元します。`Grid Phases = 2` は半strideずらした追加pass、`Save intermediate stage PNGs` は最終段より前の段階画像も保存します。各tileの生画像は保存せず、最終画像だけを通常のForge出力へ保存します。PNG infoが有効なら `parameters`、`vram_canvas`、`krea2_smart_highres`、`krea2_smart_finish` を保存し、PhaseWeaveではさらに `krea2_phaseweave` を保存します。manifestとPNG metadataの `exact_img2img_steps=true` / `exact_img2img_steps_scope=internal_tiles_only` は通常img2img設定を恒久変更したという意味ではありません。

#### Krea2 PhaseWeave 4K（実験）

`phase_weave` は2 phaseのdetail residualを先に平均しません。まず、端へ強制配置した最終tileが大きく重なる旧gridを使わず、正確なstrideの仮想gridを画像外まで延長します。全phase・全tileのモデル入力を同寸法に保ち、各軸について全phaseの左右端のcanvas交差幅の最小値が最大になる共通originを選びます。画像外は端画素で補い、canvasと交差するcoreだけを蓄積します。今回の2897×4096、core 960、stride 880ではoriginが308×688となり、最小交差幅は横388・縦328です。

各phaseを独立に完成させた後、候補差分を高周波と低周波へ分けます。低周波は輝度0.32、色差0.18へ抑え、明るい場所の輝度成分はさらに半減します。高周波量、phase内の安定度、入力との輪郭方向・低周波輝度・色差の一致を組み合わせて品質を評価し、正規化差が `+0.03` より大きければB、`-0.03` より小さければA、それ以外は未確定とします。忠実度0.42未満の候補は確定させません。未確定部は周囲の高信頼選択から伝播し、3000px未満のA/B島と512px未満の弱い穴を整理します。両候補を採らない領域は入力補間へ戻し、二候補が十分近い場合だけ±1pxの局所位置合わせ後に弱く融合します。整理後の境界だけを5pxで接続し、補助候補は構造が揃う場合だけ `0.10 × confidence²`、代表detailは0.90以上を保持します。manifestとPNGには `selection_mode=ternary_input_fallback` を含む全設定と、`grid_layout=uniform_virtual_edge_balanced`、`grid_origin`、`grid_stride`、`grid_phase_offset`、`grid_padding_mode=edge` を記録します。

CLIではprofileだけで共有値を適用できます。

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py `
  --input '<native-1k.png>' `
  --krea2-profile phaseweave_4k `
  --append-krea2-detail-prompt `
  --long-edge 4096
```

同じ実行で配置A単独、配置B単独、選択図も残す場合は `--save-phase-candidates` を追加します。この指定は `phase_weave` 専用で、別mergeでは開始前に拒否します。比較用に従来mergeだけへ戻す場合は、同じprofileへ `--merge-mode consensus` を追加します。PhaseWeaveは2組のdelta/weight/energy（novel枝を含む）をdisk memmapへ保持するため、consensusより一時disk量が増えます。候補3画像を保存する場合はさらに18 byte/pixelを事前見積りへ加えます。開始前に全段分を合算して空き容量を検査し、不足時はmodel処理前に失敗します。アルゴリズム、metadata、実画像比較の詳細は [Krea2 PhaseWeave 4K 実装・評価ノート](docs/krea2_phaseweave_4k_ja.md) を参照してください。

#### 承認済み4Kで顔・目を固定するIdentity Guard

高denoiseのdetail候補で背景や衣装は改善しても、顔・目・文字などの同一性が崩れた場合は `tools/apply_krea2_identity_guard.py` を使います。高精細候補を全体の土台にし、目視承認済み4Kから指定ellipseの中央を画素完全一致で戻し、境界だけsmoothstepで内側featherします。mask外は高精細候補と画素完全一致です。自動顔補正や新しい顔生成ではなく、承認済み画像へfail-closedで戻す最終品質ゲートです。

```powershell
.\venv\Scripts\python.exe .\tools\apply_krea2_identity_guard.py `
  --candidate '<high-detail-4k.png>' `
  --approved-reference '<approved-safe-4k.png>' `
  --output '<identity-safe-4k.png>' `
  --output-mask '<identity-guard-mask.png>' `
  --protect-ellipse 'character_a_face:880,650,1360,1200,96' `
  --protect-ellipse 'character_b_face:1690,1030,2200,1580,104'
```

`--protect-ellipse` は `label:x0,y0,x1,y1,feather` を画像pixel座標で指定し、複数回渡せます。出力PNGには `krea2_identity_guard` metadata、隣接JSONには入力hash、保護範囲、完全復元画素数、遷移画素数、候補との差分を保存します。顔の外周をぎりぎり囲まず、瞳・眉・鼻・口が完全復元coreへ入るよう少し広めに指定し、100% cropでfeather境界を確認してください。

#### Krea2 Local Supersample Detail（実験）

承認済み4Kへ局所描き込みを追加する実験的なimg2img Scriptです。通常modeは約512pxの固定payloadを1536または2048へ拡大し、モデルを通さないround-trip基準 `C0` と候補の同経路縮小 `C1` から、安全な帯域制限detail残差だけを中央coreへ合成します。顔などを実際に描き直す `Focused ROI Rewrite` は別経路です。tight target ROI全体を分割せず、その周囲を含む正方形contextを1枚の1536入力へ拡大してKrea2で再生成し、縮小後のフル `C1 - C0` をtarget ROI内だけへフェザー合成します。target外は元uint8画素をそのまま保持します。

標準手順:

1. Krea2でnative画像を生成します。
2. VRAM-Canvasで目視承認できる4Kを作成します。
3. その承認済み4Kを通常のimg2imgへ入れます。
4. `Script` で `Krea2 Local Supersample Detail` を選択します。
5. 全体の保守的なdetail追加は `Safe 1536` または `Ultra Detail 1536` を実行します。
6. 顔を描き直す場合は `Focused ROI Rewrite` / `Focused Face Rewrite 1536` を選び、顔または頭部を囲むtight boxを元画像pixel座標で指定します。
7. 100% cropで顔、目、髪、手、衣装、輪郭、tile境界を確認します。
8. 2048は通常の `ROI Boxes` で必要な領域だけに使います。
9. 局所仕上げ済み4Kを、必要に応じて既存8K経路へ渡します。

`Safe 1536` は512 payload / 384 core / overlap 64 / 4 steps / denoise 0.10 / 1候補、`Ultra Detail 1536` は5 steps / denoise 0.15 / 2候補です。`Focused Face Rewrite 1536` は6 steps / denoise 0.38 / 2候補 / context scale 2.0 / source側20px inward featherです。Focused modeでは `Crop Payload`、`Core Size`、`Core Overlap` は計画に使わず、各targetの長辺×Context Scaleを正方形context辺とします。context辺が1536以上で実拡大にならない指定、target同士の重なり、target未指定はmodel処理前に拒否します。`ROI Ultra 2048` はROI指定とROI modeが必須です。Full Image Gridの2048は `Allow expensive 2048 full-grid` を明示しない限り開始前に停止します。長時間処理の前にKrea2 checkpoint、Qwen Image VAE、Qwen3-VL、tile上限、一時disk容量を検査し、OOM時に1536へ自動fallbackして画質を変えることはありません。

顔を書き直すときは、`ROI Boxes / Focus Targets` に顔を囲むtight boxを指定します。たとえば120×150pxのtarget、Context Scale 2.0なら300×300pxの周辺contextを1枚として1536×1536へ拡大するため、実効倍率は5.12倍です。顔は複数tileへ分割されません。Krea2候補の縮小後フル差分を意図的に採用するので、通常modeより目、輪郭、明度、表情が変わり得ます。`Save QA crops` をONにし、`high_resolution_candidate.png` と4K書き戻し結果を必ず比較してください。

Batch Count / Batch Sizeは1、通常img2imgだけに対応します。内部candidate処理の間だけrequest-local overrideでExact Stepsを有効にします。内部処理中は標準画像保存、`Restore faces`、`Tiling`、maskを無効にし、成功・例外・中断・skip・stop・OOMのすべてでprocessing状態とExact Steps設定を復元します。residual/weight accumulatorはCPUのdisk memmapです。最終出力は入力と同寸法の1枚だけで、target外と通常modeの全候補no-op時の画素は元uint8値を保持します。Focused実行時は `focused_rewrite`、context box、payload辺、実効拡大率、feather、選択候補、元のquality-gate棄却理由をPNG metadataへ記録します。manifest/PNGには `exact_img2img_steps=true` / `exact_img2img_steps_scope=internal_tiles_only` も保存します。`Save QA crops` がONの場合だけ、timestamp別directoryへ代表領域の同一payload比較を保存します。

この機能は実画像ごとの目視比較が必要であり、必ず高画質になることや既存方式より優れることを保証しません。4096×1756の実機例では、120×150pxの顔targetに300×300pxのcontextを取り、1536へ5.12倍拡大して1回のcoherent regenerationを行いました。顔target内は16,646画素（92.48%）が変化し、target外は0画素、処理49.1秒、peak VRAM 20,957MiBでした。旧固定tile方式のfail-closed事例とは目的が異なります。詳細は [使用ガイド](docs/krea2_local_supersample_detail_ja.md)、[Focused ROI実測短報](docs/krea2_focused_roi_rewrite_b5_ja.pdf)、通常modeの [Fail-Closed実測短報](docs/krea2_local_supersample_b5_ja.pdf) を参照してください。

#### B5 Whole-Tile Oversampled Regeneration（実験）

`tools/krea2_b5_tile_regenerate.py` は、1Kの縦長画像をJIS B5向け2896×4096へ仕上げるCLIです。専用の超解像モデルは使いません。既定では入力をLanczosで2倍の作業キャンバスへ拡大し、重なり付きの各256×256 crop全体をKrea2へ1024×1024で渡し、返却画像全体を512×512へ縮小して2倍キャンバスへ合成します。中央512だけを切り出す処理ではありません。1024×1448入力なら4096×5792の生成キャンバスを作ってから、全体を2896×4096へ縮小するため、最終処理は拡大ではなく約0.707倍の縮小です。

局所cropへ元PNGの全景promptを繰り返し渡すと、背景の木目や布目がprompt内の人物・生物として再解釈されることがあります。そのため既定は、cropを「全景ではない拘束済み局所画像」と明示し、新規人物・顔・生物・文字・反復模様の追加を禁止する局所復元promptです。`low_anchor` mergeはKrea2候補の低周波を決定論的2倍基準へ戻し、RGB差を制限してから重なりをsmoothstep合成します。より保守的な `source_gate` は、元cropに既存の細線エネルギーがある場所だけへ高周波差分を通します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_b5_tile_regenerate.py `
  --input '<native-1k.png>' `
  --output-root '<b5-output-directory>' `
  --working-scale 2 `
  --tile-size 256 `
  --process-edge 1024 `
  --overlap 64 `
  --steps 6 `
  --denoise 0.35 `
  --merge-mode low_anchor `
  --low-anchor-sigma 16 `
  --max-tile-delta 16 `
  --protect-ellipse 'character_a_face:185,335,355,540,24' `
  --protect-ellipse 'character_b_face:670,370,840,585,24'
```

`--protect-ellipse` は元1K画像のpixel座標で指定します。各生成stageで自動scaleし、候補の重なり合成後に決定論的拡大像を戻すため、目や手などをtileごとの別seedで混ぜません。実行前に選択checkpoint名へKrea2、追加moduleへQwen Image VAEとQwen3-VLがあることを検査します。処理中はdisk memmapを使い、失敗時の `_incomplete_b5_tiles_*` directoryは指定output root直下であることを確認してから削除します。成功時はstage PNG、B5 PNG、入力hash・seed・全tile時間・保護範囲を含むmanifestをtimestamp directoryへ保存します。

低denoiseは安全とは限りません。この画像での実測では0.08～0.22のKrea2往復像は元の細線を少し平滑化し、0.50では本棚を人物の横顔へ再解釈しました。0.35＋局所prompt＋低周波固定＋RGB差±16は偽人物を抑えましたが、平坦部の微細変動はLanczos基準より増えました。顔・目・手・文字、24本のtile境界、単純Lanczos基準を100% cropで比較し、改善しない画像では採用しないでください。

#### Krea2 Smart 4K → 8K品質ゲート

指定promptからのnative生成、4K preflight、8K、Smart Finishを連続実行するCLIもあります。選択checkpoint名にKrea2、追加moduleにQwen Image VAEとQwen3-VLがない場合は長時間処理の前に停止します。`params.txt` の先頭行をbase promptとして使う例:

```powershell
$prompt = Get-Content -LiteralPath '.\params.txt' -First 1
.\venv\Scripts\python.exe .\tools\krea2_smart_8k.py --prompt $prompt --stop-after 4k
```

既存native画像を使う場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_smart_8k.py --prompt $prompt --source '<native-image>' --source-stage native --stop-after 4k
```

4Kが合格した場合だけ8Kへ継続します。4Kの縦横比を保った正確な2倍になるため、A-Series `2896x4096` は `5792x8192`、UHD `3840x2160` は `7680x4320` になります。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_smart_8k.py --prompt $prompt --source '<approved-4k.png>' --source-stage 4k --stop-after 8k --tile-size 1024
```

4K/8K gateは、全tile成功、skip 0、正確な寸法、gradient/high-pass保持率の下限と1.8倍の上限、Smart Finishの合格または安全なbit-identical no-op、平坦部変更0、channel clipping上限を検査します。4096px基準の4周波数帯metricも保存しますが、global high-pass値だけではnoiseもdetailと誤認できるため、顔・目・角・髪・透明物・tile境界の1024×1024原寸cropを固定7地点で保存し、目視を省略しません。VRAM・GPU使用率・温度は1秒pollingのJSONへ残します。subprocessのprompt/negative promptはcommand logで伏せ、hard timeoutはstdout EOFを待たずに監視します。

#### CLIから使う

Forgeを `--api --port 7861` で起動し、まずdry-runします。PNGにForge infotextがない場合は `--prompt` も渡します。

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py --input '<input-image>' --long-edge 4096 --dry-run
```

問題なければ `--dry-run` を外します。VRAM総量は `/sdapi/v1/memory` から自動取得し、取得できなければ安全側の8GiBとして計画します。明示する場合:

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py --input '<input-image>' --long-edge 4096 --vram-budget-gib 8
```

UHD 8Kを正確に指定する例。`--phase-count 2` は半strideずらした第2passを追加してtile境界をさらに分散しますが、処理時間も大きく増えます。

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py --input '<input-image>' --width 7680 --height 4320 --phase-count 2
```

主な既定値:

```text
Stage scale: 最大2倍
Tile: VRAM予算から自動（384～1280）
Halo: tile edgeの1/8
Core overlap: haloの1/2
Steps: 局所detail量に応じて2～4
Denoise: coarse 0.12 → final 0.08
Low-pass radius: 12px
Per-channel detail delta: 最大±32
Base detail protection: 6（0で無効）
Phase count: 1
Consensus noise floor: 8（0で無効）
Novel detail gain: 0（Dense Detail 4K/8K profileでは1.0/0.8、2 phases必須）
Novel detail maximum delta: ±8（Dense Detail 8Kでは±6）
Maximum delivery: 70MP
```

出力先は `output/vram_canvas/vram_canvas_<timestamp>/` です。最終PNG、各段階PNG、全tileの座標・seed・step・detail scoreを含む `run_manifest.json` を保存します。novel枝を有効にしたconsensusは概算48 byte/pixel、2 phaseを独立保持するPhaseWeaveは84 byte/pixelとして全段の一時memmap量を開始前に検査します。VRAM計画式はモデルとattention実装に依存する近似で、明示tileも予算に収まらなければ開始前に失敗します。

- [JIS B5判2ページ・実測図版入りアルゴリズム短報（日本語PDF・2段組）](docs/vram_canvas_b5_ja.pdf)
- [論文テキスト版](docs/vram_canvas_b5_ja.md)
- [Krea2 A-Seriesから1:√2・5792×8192へ拡張した実測ケーススタディ](docs/vram_canvas_krea2_case_study_ja.md)
- [Krea2 PhaseWeave 4K 実装・実画像比較ノート](docs/krea2_phaseweave_4k_ja.md)
- [DetailWeave 4K・JIS B5判2ページ短報（Lanczos／MultiDiffusion／提案法の実画像比較）](docs/detailweave_4k_b5_ja.pdf)
- [同短報の初心者向けテキスト版](docs/detailweave_4k_b5_ja.md)
- [Krea2 Local Supersample Detail 使用ガイド](docs/krea2_local_supersample_detail_ja.md)
- [Krea2 Local Supersample Detail・JIS B5判2ページ実測短報](docs/krea2_local_supersample_b5_ja.pdf)
- [同短報のテキスト版](docs/krea2_local_supersample_b5_ja.md)
- [添付の雪駅画像を1915x821から4096x1756へ変換した実測短報（日本語PDF・5ページ）](docs/krea2_smart4k_snow_station_case_study_ja.pdf)
- [同実測短報のテキスト版](docs/krea2_smart4k_snow_station_case_study_ja.md)

短報PDFは、完走した4K/8K manifestと原寸QA cropからJIS B5・2ページとして再構築します。

```powershell
.\venv\Scripts\python.exe .\tools\build_vram_canvas_paper.py --run-manifest '<complete-8k-manifest.json>' --preflight-manifest '<complete-4k-manifest.json>'
```

ケーススタディ用の候補画像、4K/8K manifest、Smart Finish reportがローカルにある環境では、図版入り6ページPDFを次で再構築できます。PDFと縮小図版は`output/pdf/`へ出力し、元の生成画像は上書きしません。

```powershell
.\venv\Scripts\python.exe .\tools\build_krea2_vram_canvas_case_study.py
```

雪駅画像の実測短報は、完走済みSmart 4K・VRAM-Canvas・Local SupersampleのmanifestとQA画像を検証しながら次で再構築します。PDFと比較図版は`output/pdf/`へ出力します。

```powershell
.\venv\Scripts\python.exe .\tools\build_krea2_smart4k_snow_station_paper.py
```

RTX 3090 24GiBとKrea2 NF4で、8192x6144、tile 1280、halo 160、2 phases（150 tile）のend-to-end実行を確認済みです。150/150 API呼出しが成功し、処理時間は約15分40秒、`nvidia-smi` の監視中最大使用量は約21.9GiBで、OOMや非有限値は発生しませんでした。速度とpeak VRAMはcheckpoint、attention実装、prompt、ホスト負荷で変動します。geometry、周波数残差、disk memmap、dry-run、模擬Forge APIを通したCLI処理、およびForge内部処理を模擬したGUI処理と状態復元もテストしています。

顔や体だけを描き直したい場合は、全体画像を再拡散せず、`tools/krea2_subject_refine.py` で指定領域だけをKrea2 img2imgしてフェザー合成します。処理に送るのは切り出した小さなcropだけなので、長辺6K級の画像でもGPU負荷を抑えられます。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_subject_refine.py --input '<large-output-image>' --box 1650,520,2850,2150 --padding 160 --process-long-edge 1536 --denoise 0.10 --steps 4
```

`--box` は `left,top,right,bottom` のピクセル指定です。複数箇所を処理する場合は `--box` を複数回渡します。画像サイズに依存しない指定が必要な場合は `--box-normalized 0.286,0.208,0.495,0.861` のように0..1の正規化座標を使います。顔だけなら小さめのboxと `--mask-shape ellipse`、体全体なら少し広めのboxと既定の `rectangle` が扱いやすいです。

Forge UIから使う場合は、img2imgタブの `Script` で `Krea2 2-Stage Upscale` を選びます。UI既定も最終長辺4096、profile連動proxy（Raw 1024、Turbo/custom 2048）、denoise 0.10/0.12、Smart chroma finish ON、孤立粒補修OFFです。通常のimg2img入力画像だけを対象にし、Batch Count / Batch Size / Tile Batch Sizeはいずれも1にします。

### Tiled VAE / tiled Conv2d

ForgeのVAE decodeは、通常は `Full` で実行され、VAE decode中にOOMした場合にTiled VAEへフォールバックします。

重要な点:

```text
Tiled VAEはVAE encode/decodeのメモリ対策
拡散ステップ本体のメモリ対策ではない
高解像度proxyを安定させるにはMultiDiffusionなどで拡散自体をタイル化する必要がある
```

このforkの `webui-user.bat` では、VAE系のConv2dをタイル化するために次を追加しています。

```text
--tiled-conv2d 128
```

これは高解像度VAE処理のメモリを下げる目的です。速度は落ちます。

Tiled VAEのタイル合成では、境界フェザーをsmoothstep重みに変更し、同じ形のマスクを再利用するようにしています。分母マップは単一チャンネルで保持し、継ぎ目を増やさずタイルごとの余分なメモリ確保を減らします。

### upstreamから継承するModelMerger / Extras / Upscaler

`origin/neo` の `120b01fc` を取り込んでいるため、Checkpoint Merger rewrite、Extras、upscaler、Refiner、img2img / inpaint / mask処理の更新は最新upstream基盤を継承します。これら自体はこのfork独自機能ではありません。このfork側では、その基盤にKrea2高解像度workflow、処理状態の安全なoverlay、品質gate、runtime検証を追加しています。

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
Checkpoint: krea2_turbo_int8_convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen3vl_4b_fp8_scaled.safetensors
Diffusion in Low Bits: Automatic
```

Krea2 t2iの実用初期値:

```text
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 4
CFG scale: 1.0
Model shift: 1.15 (Krea2 fixed default)
Size: 1024x1024
Hires.fix: Off
```

この初期値はTurbo系bnb-nf4 checkpointで行ったsampler/step比較の低ノイズ最良値です。既定のINT8 ConvRotは同じTurbo系として初期値を共有しますが、量子化形式が異なるため同一出力は保証しません。checkpointによって不安定な場合は `Euler / Simple / 6 steps` を安定側fallbackとして試します。公式Krea-2-Rawは推論用の第一候補ではなく、使用する場合は[公式model card](https://huggingface.co/krea/Krea-2-Raw)の `52 steps / guidance 3.5` を基準に別設定へ切り替えてください。

Krea2のQwen3-VL text encoderは自然文promptを前提とし、A1111形式の `(word:1.2)` を数値weightとして適用しません。この記法はQwenへliteral textとして1回だけ渡されます。強調したい内容は `prominent red subject` のように自然文で明示してください。

Krea2 i2i / upscaleでノイズを抑える初期値:

```text
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 8
CFG scale: 1.0
Model shift: 1.15 (Krea2 fixed default)
Stage 1 denoising strength: 0.10
Final denoising strength: 0.12
```

少し描き足す場合:

```text
Stage 1 denoising strength: 0.14-0.18
Final denoising strength: 0.16-0.22
```

避ける設定:

```text
Denoising strength 0.3以上を低ノイズ用途の既定にする
4K/8Kの直接txt2img
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
    "sampler_name": "DPM++ 2M SDE",
    "scheduler": "Simple",
    "steps": 4,
    "cfg_scale": 1.0,
    "distilled_cfg_scale": 1.15,
    "width": 1024,
    "height": 1024,
    "send_images": True,
    "save_images": True,
}

response = requests.post("http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload, timeout=1200)
response.raise_for_status()
```

現行のForge Krea2 backendはmodel shiftをcheckpoint設定の `1.15` へ固定しています。API例の `distilled_cfg_scale` はリクエストの自己記述性を保つために明示していますが、Krea2では可変guidanceとしては使われません。

`CFG scale: 1.0` の場合、Forgeではnegative promptが無視されます。negative promptを効かせたい場合は `CFG scale` を上げてください。

## Krea2 Smart 4K helperの使い方

入力画像を明示して実行します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --dry-run
```

問題なければ本番。既定ではKrea2 custom profileを安全側の2Kとして扱い、最終拡散を長辺2048に抑え、長辺4096の画像へローカルで保存します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>'
```

既定の `--upscale-mode auto` は、入力と2K proxyの差が十分ある場合だけ中間passを追加します。中間サイズを固定する場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --first-pass-long-edge 1792
```

さらに元絵保持を優先する場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --first-pass-denoise 0.08 --denoise 0.10 --steps 6
```

8K納品画像も作れますが、拡散は既定の2K proxyのままです。これはnative 8K生成ではなく、Krea2で整えた2K画像の決定論的拡大です。必要な顔・手・小物は `krea2_subject_refine.py` で局所的に処理します。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --long-edge 8192 --allow-unsafe-large-delivery
```

ローカルresize/Smart Finishは補正本体も実解像度で処理するため、既定guardは4K正方形を含む20MP以下です。8Kや8192正方形（約67MP）を意図的に処理する場合だけ、空きRAMを確認して `--allow-unsafe-large-delivery` を追加してください。

無進捗時の自動停止は既定で600秒です。変える場合は `--no-progress-timeout 900` のように明示します。

native範囲外でより描き直す実験をする場合は、危険性を明示してopt-inします。これは公式推奨ではありません。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_8k_img2img.py --input '<input-image>' --diffusion-long-edge-cap 3072 --allow-non-native-diffusion --first-pass-denoise 0.18 --denoise 0.22
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

このforkでは特にKrea2のnative INT8 tensorwise + ConvRot運用を重視し、bnb-nf4も任意構成として扱えます。

## 注意点

- モデル、VAE、text encoderはリポジトリに含めません。
- `logs/`、`output/`、ローカルYAML、生成画像は通常push対象外です。
- `webui-user.bat` にはローカル環境向けパスが含まれる場合があります。
- 巨大解像度は非常に時間がかかります。
- 4K/8K納品画像はまずdry-runでproxyとリクエスト内容を確認してください。
- 高解像度処理中にVRAMが張り付いた場合は `/sdapi/v1/interrupt`、`/sdapi/v1/skip`、`/sdapi/v1/unload-checkpoint` の順で止めます。

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

### 4K/8K処理で止まる、またはVRAMが張り付く

直接t2iで巨大解像度を作らず、`tools/krea2_8k_img2img.py` を使います。
Forge UIだけで試す場合は、img2imgタブの `Script` から `Krea2 2-Stage Upscale` を選びます。

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
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 8
Stage 1 denoising strength: 0.10
Final denoising strength: 0.12
```

`0.3` 以上は再描画が強くなり、肌や背景に粉っぽい再描画が出る場合があります。

生成後の画像だけを補正する場合は、まずSmart Finishを使います。色ムラはchroma-onlyで解析し、補正後の指標が改善しない候補は破棄します。孤立粒補修は既定OFFです。

```powershell
.\venv\Scripts\python.exe .\tools\krea2_smart_finish.py --input '<image>'
```

白/黒の孤立粒がartifactだと確認でき、雪や星ではない場合:

```powershell
.\venv\Scripts\python.exe .\tools\krea2_smart_finish.py --input '<image>' --despeckle
```

従来の詳細なmask確認が必要な場合は、単体despeckle toolも利用できます。

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

### 色ムラheatmapが髪や服の線だらけになる

旧checkerはLabのL/a/bすべてをΔEとしていたため、陰影や無彩色テクスチャも色ムラに数えていました。現在はLab a/bだけのchroma deltaへ分離し、L勾配・chroma edge・細部を保護します。4Kでは長辺1536へ縮小解析し、Overlay/Heatmapは必要時だけ明示して生成します。

## upstreamへの敬意

このforkは以下のプロジェクトをベースにしています。

- AUTOMATIC1111 Stable Diffusion WebUI
- lllyasviel Stable Diffusion WebUI Forge
- Haoming02 sd-webui-forge-classic / Neo
- ComfyUI系バックエンド実装

元プロジェクトと関連コントリビューターに感謝します。
