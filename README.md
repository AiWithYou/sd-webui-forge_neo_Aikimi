# Aikimi Neo

`Aikimi Neo`は、[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic)を生成基盤として、画像・動画・モデルの各ワークフローを統合するWindows向けAI生成workspaceです。Krea2の導入と高解像度ワークフロー、SenseNova U1.5 Studio、MiniMax H3 Studio、画像品質処理、Anima 3.8Bの軽量構成とLoRA互換性に加え、独自のブランドUIとキャラクターアシスタントを備えています。

Forge Neo由来の生成機能とExtension互換性はupstreamから継承しています。このREADMEでは、Aikimi Neo独自の機能と、初めて使うときの入口を中心に説明します。

| 項目 | 内容 |
|---|---|
| 作業ブランチ | `neo` |
| upstream | `Haoming02/sd-webui-forge-classic` の `neo` |
| 同期基準 | `origin/neo` `6009ffff99b5d5b4312dc8a8f6476ec0a69b37b1` |
| 同期日 | 2026-08-15 |
| 主な対象 | Krea2、4K/8K img2img、SenseNova U1.5、MiniMax H3、Anima 3.8B |

## 目的別の入口

| やりたいこと | 最初の入口 |
|---|---|
| 通常のForge画像生成を使う | `webui-user.bat` |
| ちびあいきみの状態表示を切り替える | `Settings` → `Aikimi Assistant` → `Show Aikimi` |
| Krea2 INT8モデルをまとめて導入する | `download_krea2_int8_convrot_models.bat` |
| Anima 3.8Bを軽量構成で導入する | `download_anima38_int8_convrot_models.bat` → 通常の`txt2img` |
| SenseNova U1.5正式版をINT8 ConvRotで使う | `download_sensenova_u15_int8.bat` → `SenseNova U1.5`タブ |
| Krea2画像を安全側で4Kまたは8Kへ仕上げる | `img2img` → `Krea2 2-Stage Upscale` |
| VRAM予算を固定して段階的に高解像度化する | `img2img` → `VRAM-Canvas 4K/8K Highres` |
| 承認済み4K画像の局所ディテールを増やす | `img2img` → `Krea2 Local Supersample Detail` |
| B5判・縦の印刷向け画像をタイル単位で再生成する | `img2img` → `Krea2 B5 Whole-Tile Regeneration` |
| 候補比較と品質gateを使って再作画upscaleする | `img2img` → `HyperWeave 4K/8K` |
| MiniMax H3で音声付き動画を生成する | `H3 Studio`タブ |
| 28層・40層・52層のAnima LoRAを使う | 通常どおりLoRAを選択。必要な場合だけ自動変換 |
| AI画像の色ムラや細かな網目を整える | `Extras` → `Color Flatten / 色ムラ補正` |

高解像度ワークフローの一部は実験機能です。入力に存在しない細部は生成モデルによる推定であり、未知の正解画像を復元する機能ではありません。

## Quick Start

この手順はWindows 11を前提にしています。先に次を用意してください。

- Git
- NVIDIA GPUに対応したドライバー
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

リポジトリを取得します。

```powershell
git clone --branch neo https://github.com/AiWithYou/sd-webui-forge-classic_neoW.git
cd sd-webui-forge-classic_neoW
```

通常は次をダブルクリックします。

```text
webui-user.bat
```

`webui-user.bat`はリポジトリ自身を作業フォルダーとして使い、既定でAPI、dark theme、BnB、高解像度VAE向けのtiled Conv2dを有効にします。

ComfyUIとモデル置き場を共有する場合は、リポジトリ直下へ`forge_neo_model_paths.yaml`を置きます。このファイルは任意です。存在しない場合は引数へ追加せず、Forge標準の`models`フォルダーを使います。

モデル、VAE、text encoder、LoRAはリポジトリに含まれません。それぞれの配布元ライセンスを確認して配置してください。

### Krea2 INT8を使う場合

次をダブルクリックします。

```text
download_krea2_int8_convrot_models.bat
```

固定したHugging Face revisionから約17.69 GiBを取得し、サイズ、SHA-256、safetensors headerを検証して次へ配置します。

```text
models/Stable-diffusion/krea2_turbo_int8_convrot.safetensors
models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors
models/VAE/qwen_image_vae.safetensors
```

中断した場合は同じbatを再実行すると`.part`から再開します。既に検証済みのファイルは再取得しません。配布元は[Comfy-Org/Krea-2の固定revision](https://huggingface.co/Comfy-Org/Krea-2/tree/8038ce89b91b042141541ad0fa51b985ca262c5f)です。

起動後、Forge UIで次を選びます。

```text
Preset: krea
Checkpoint: krea2_turbo_int8_convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen3vl_4b_fp8_scaled.safetensors
Diffusion in Low Bits: Automatic
```

最初のtxt2img設定は次が目安です。

```text
Sampler: DPM++ 2M SDE
Scheduler: Simple
Steps: 4
CFG scale: 1.0
Size: 1024x1024
```

Krea2のmodel shiftはcheckpoint設定の`1.15`を使います。Qwen3-VL text encoderでは、重要な内容を自然文で明示してください。A1111形式のweight構文は数値weightとして適用されません。

### Anima 3.8Bを軽量構成で使う場合

Anima 3.8Bは、52層へ拡張したDiT、Qwen3.5 4B、専用のprogressive cross-attention adapterを組み合わせるモデルです。Qwen3.5の配布weightは、約34.98億parameterがFP8、約6.40億parameterがBF16の混合精度です。Aikimi NeoはQwen3.5を再量子化せず、全BF16のDiTだけをINT8 ConvRotへ変換します。変換後のDiTは7,504,189,974 bytesから4,238,326,342 bytesになり、約43.5%小さくなります。

次をダブルクリックしてください。

```text
download_anima38_int8_convrot_models.bat
```

セットアップは固定したHugging Face revisionから3ファイルを取得します。BF16 DiTは変換用の一時ファイルとして扱い、変換、520層分のConvRot署名、ファイルサイズ、SHA-256 sidecarの検証が成功した後に削除します。元のBF16も残す場合は、PowerShellから`-KeepSource`を指定してください。

```text
一時取得: Anima-3.8B.safetensors
配置: models/Stable-diffusion/Anima-3.8B-int8-convrot.safetensors
配置: models/Stable-diffusion/Anima-3.8B-int8-convrot.safetensors.sha256
配置: models/text_encoder/qwen35_4b.safetensors
配置: models/text_encoder/Anima-3.8B-expanded_adapter.safetensors
```

既存のAnima共通ファイルも必要です。

```text
models/text_encoder/qwen_3_06b_base.safetensors
models/VAE/qwen_image_vae.safetensors
```

起動後、Forge UIで次を選びます。

```text
Preset: anima
Checkpoint: Anima-3.8B-int8-convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen_3_06b_base.safetensors
Diffusion in Low Bits: Automatic
Anima 3.8B (Qwen3.5): enabled
Adapter strength: 1.0
Use adapter on negative prompt: off
Standard Anima LoRA: None、または使用するLoRA
Standard LoRA strength: 1.0
Low VRAM encoder offload: off
```

`Standard Anima LoRA`には、`models/Lora`で検出した完全な28層・40層・52層のsafetensors LoRAだけを表示します。28層版と40層版は52層へ自動展開してから適用します。起動後にLoRAを追加した場合は`Refresh standard Anima LoRAs`を押してください。複数の通常LoRAを組み合わせる場合は、従来どおりLoRAタブまたは`<lora:name:weight>`タグを使えます。

最初は`832x1216`前後の約1MP、`res_multistep + Beta`、28〜50 Steps、CFG 7〜8を基準にしてください。Qwen3.5のembedding tableはCPUへ保持し、必要tokenだけをGPUへ送ります。同じpromptでSeedだけを変える場合はconditioning cacheを再利用します。VRAMを優先する場合に限り、Low VRAM encoder offloadを有効にしてください。

標準の`webui-user.bat`はRTX 3090向けに`--cuda-malloc`を有効化し、`cudaMallocAsync`でallocation stallと断片化を抑えます。`sd_forge_compile`の`max-autotune`と`reduce-overhead`はこのallocatorと併用できないため、将来そのCompile presetを使う場合は、起動引数から`--cuda-malloc`を外してどちらか一方を選んでください。

### SenseNova U1.5正式版のINT8 ConvRotを使う場合

次をダブルクリックします。

```text
download_sensenova_u15_int8.bat
```

このセットアップでは、次のファイルと実行環境を準備して検証します。

- 正式版config、tokenizer、INT8 ConvRot loader、SenseNova推論コードを固定revisionから取得し、各Git blob SHA-1を検証
- Forge内のPyTorch、Transformers、Comfy Kitchenなど、専用loaderに必要な依存関係を検証
- 約16.52 GiBの`SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors`を再開可能な方式で取得
- ConvRot署名、17,734,813,848 bytesのサイズ、SHA-256を検証
- 公式8-Step T2I LoRAを固定revisionから取得し、814,867,236 bytesのサイズ、294 target、SHA-256を検証

完了後にForgeを起動し、`SenseNova U1.5`タブを開きます。テキスト生成の既定値は公式8-Step LoRA、8 Steps、CFG 1.0です。画像編集はQuality 50-Step、CFG 4.0へ自動的に切り替わります。どちらも`正式版 · INT8 ConvRot`、`24GB Safe · 2K出力優先`、BF16計算を維持します。

このINT8 ConvRotは正式版SenseNova U1.5を基にしています。ただし、変換weightと専用loaderはコミュニティ管理であり、SenseNova公式配布のweightではありません。Studioは固定したcheckpointだけを受け付け、Preview、GGUF、BF16へ暗黙に切り替えません。

## Aikimi Neo独自の追加

### Aikimi Status

公式マスコットの「ちびあいきみ」が画面右下に常駐し、アプリ全体の状態を短い台詞と主要指標で知らせます。通常表示では生成進捗、待機Queue、VRAM使用量を確認でき、キャラクターを選ぶとモデル名、モデル読込時間、ETA、backend状態、エラー詳細が開きます。

キャラクターは設定資料に近い約2頭身で、`idle`、モデル読込、生成、完了、Queue、警告、エラー、VRAM不足、更新の各状態に専用ポーズを備えています。idleと生成中は6原画から10フレームのping-pongを構成し、完了時には7原画のone-shotを採用しました。穏やかな呼吸、作業中の淡い発光、短い喜びを、それぞれのAPNGで表現しています。

状態表示は既存の生成progress、Queue、モデルloader、VRAM情報を読み取り、従来の技術ログを置き換えません。`Settings`の`Aikimi Assistant`では、表示のON/OFFに加えて、キャラクターサイズ、画面四隅の表示位置、台詞、アニメーションを個別に変更できます。アニメーションを無効にした場合やOSでモーション低減を有効にした場合は、同じ状態の静止WebPへ自動的に切り替わります。

txt2imgとimg2imgには`Add to Queue`ボタンを配置しました。生成中でも現在の入力値を独立したジョブとして複数回追加でき、既存のFIFO GPU queueで1件ずつ順番に処理されます。待機件数はAikimi Statusで確認可能です。`Interrupt`と`Skip`は実行中のジョブを対象とし、追加済みの待機ジョブはその後も継続します。

APNGビルダー、配布アセット、Chromiumのモーション低減動作は、`.github/workflows/aikimi-assets.yml`でも自動検証します。CIではfixtureを入力にAPNG一式を構築し、その出力と配布ファイルのPNG chunk、duration、loop、透明余白、静止画との一致を確認します。

### SenseNova U1.5 Studio

`SenseNova U1.5`は、NEO-unify固有の生成ループをForgeの通常samplerへ変換せず、正式版対応の専用コードを隔離workerで実行するGUIです。

- テキスト生成と複数画像編集に対応
- 参照画像を最大64枚まで一括または個別に追加し、差し替え、削除、前後移動が可能
- 24GB Safeでは2K出力を維持し、参照2枚を各512²へ縮小して、過大な参照入力をモデル読込前に拒否
- 編集時に1枚目の比率を維持する約4MPの自動出力と、参照画像の縮小モードを独立して選択可能
- 正式版を基にしたINT8 ConvRotを固定し、Previewとの混在を防止
- 公式8-Step T2IとQuality 50-Stepを分離し、画像編集への高速LoRA誤適用を防止
- decoder layerの理解分岐と生成分岐を識別し、各forwardで必要なweightだけをGPUへ転送
- 生成前にruntime revision、Python依存、safetensorsのサイズ、ConvRot署名、完全性記録を検査
- 生成時に通常のForgeモデルを退避し、キャンセル時は隔離workerを停止してメモリを解放
- 完成PNGと生成条件JSONを`outputs/sensenova_u15`へ保存

詳しい設定、複数画像の順序、INT8 ConvRotの制約は[SenseNova U1.5 Studioガイド](extensions-builtin/sensenova-u15-studio/README.md)を参照してください。

### MiniMax H3 Studio

`H3 Studio`は、Aikimi NeoからローカルのComfyUI MiniMax H3 runtimeを操作する専用GUIです。

- テキスト、キーフレーム、参照素材の3モード
- 映像と32 kHzステレオ音声を同時生成
- runtime、必要node、モデル、Comfy Kitchen、RAM余力を生成前に検証
- Forgeが起動したruntimeだけを安全に再起動
- 完成MP4と生成条件を`outputs/minimax_h3`へ保存
- 最近の生成から設定を復元

MiniMax H3モデルと対応するローカルComfyUIは別途必要です。外部有料APIは呼び出しません。必要ファイル、runtime条件、参照タグの使い方は[MiniMax H3 Studioガイド](extensions-builtin/minimax-h3-studio/README.md)を参照してください。

### Krea2の導入と低ビット互換性

Aikimi Neoは、Krea2向けに次の機能を追加しています。

- 固定revisionとhash検証を使う一括downloader
- BF16 merged checkpointからINT8 ConvRotを作るstreaming変換tool
- bitsandbytes safetensorsの`QuantState`から元shapeを復元するKrea2検出
- Qwen3-VL入力と、checkpoint・VAE・text encoderのruntime preflight
- Krea2 presetと高解像度処理向けの安全側の既定値

現在の推奨入口はpre-quantized INT8 ConvRotと`Automatic`です。一方、Aikimi Neoは既存モデルとの互換性を守るため、upstream同期で削除対象となったBnB/NF4経路とGGUF対応を意図的に維持しています。NF4またはGGUFを使う場合は、対象モデルの形式に合わせて選択してください。

### Krea2高解像度ワークフロー

高解像度処理は目的に応じて選びます。

| Script | 向いている用途 | 性質 |
|---|---|---|
| `Krea2 2-Stage Upscale` | 初めての4K/8K納品 | Krea2のnative範囲内のproxyを使い、正確な納品寸法へ仕上げる |
| `VRAM-Canvas 4K/8K Highres` | VRAM予算を固定した段階処理 | 全キャンバスをGPUへ載せず、局所残差を合成する実験機能 |
| `Krea2 Local Supersample Detail` | 承認済み4Kの顔、衣装、小物 | 指定領域またはtileの候補を評価し、通過した局所残差だけを採用する実験機能 |
| `Krea2 B5 Whole-Tile Regeneration` | B5判・縦の印刷向け仕上げ | 画像全体を重複tileで再生成し、保護領域を元画像から復元する実験機能 |
| `HyperWeave 4K/8K` | 候補制約を使う生成的upscale | 独立Extension。構造、色、輪郭、境界を評価して候補残差を選ぶ |

高解像度VAE処理と拡散処理は別です。Tiled VAEやtiled Conv2dはVAE encode/decodeのメモリを減らしますが、巨大な拡散キャンバス自体を安全にするものではありません。

実装と評価資料:

- [Krea2 Local Supersample Detailガイド](docs/krea2_local_supersample_detail_ja.md)
- [PhaseWeave 4K実装・比較ノート](docs/krea2_phaseweave_4k_ja.md)
- [VRAM-Canvas Krea2ケーススタディ](docs/vram_canvas_krea2_case_study_ja.md)
- [Smart 4Kケーススタディ](docs/krea2_smart4k_snow_station_case_study_ja.md)
- [DetailWeave B5短報](docs/detailweave_4k_b5_ja.md)

### HyperWeave 4K/8K

HyperWeaveは、現在Forgeに読み込まれている生成モデルで入力画像を段階的に再作画する独立built-in Extensionです。

1. `img2img`を開き、入力画像、prompt、sampler、scheduler、CFGを設定します。
2. `Script`から`HyperWeave 4K/8K`を選んでください。
3. 最初は`4K long edge`、`Structure Safe`、少ない候補数で実行します。
4. 構図、顔、髪、境界を確認できたら、`Overdraw`や候補数を上げます。

初期latent noiseは絶対座標に合わせ、候補はround-trip、低周波誤差、色ずれ、輪郭、新規edge、tile境界などで評価します。すべて不合格なら基準画像へ戻します。

顔検出が不確かなイラストやアニメ画像では、入力と同じ座標系のManual Face Core Maskを指定してください。設計、推奨設定、manifest、既知の制限は[HyperWeaveガイド](extensions-builtin/hyperweave/README.md)を参照してください。

### 画像品質と後処理

`Extras`の`Color Flatten / 色ムラ補正`は、画像全体を再生成せず、色面や微細なAIノイズを整えます。

| Mode | 用途 |
|---|---|
| `Smooth Gradient / AI Noise` | 輝度を含む細かな網目、ざらつき、帯状のムラを滑らかにする |
| Color Flatten系 | 色面を整理しつつ、edge保護を使って境界を残す |

Krea2のSmart FinishはLab a/b中心のchroma-only解析を行い、指標が改善する候補だけを採用します。雪、星、そばかす、粒子を誤って消す可能性があるため、孤立した白黒粒の補修は既定で無効です。

### Anima 3.8BとLoRAの厳密な28層・40層・52層変換

AnimaおよびAnima-2.9Bの基本モデル対応、40層checkpoint検出、基本的な28→40 LoRA remapはupstream由来です。Aikimi Neoは52層checkpointの検出、Qwen3.5 adapter、INT8 ConvRot変換を追加しています。

Aikimi NeoはLoRA変換を次の方針で強化しています。

- 完全な`0..27`、`0..39`、`0..51`のblock coverageだけを自動変換
- Kohya、Forge generic、PEFT、Comfy形式のblock keyに対応
- 公開checkpointの挿入manifestに基づく28↔40↔52の双方向変換
- 28→40、28→52、40→52では同じtensor storageを共有し、変換だけでtensor本体を複製しない
- sparse LoRAや層限定LoRAは誤変換せず警告
- 52→40、52→28、40→28では追加層を破棄するため、不可逆であることを警告

モデルの配置例:

```text
models/Stable-diffusion/Anima-2.9B-preview-v1.safetensors
models/text_encoder/qwen_3_06b_base.safetensors
models/VAE/qwen_image_vae.safetensors
```

UIでは`Preset: anima`を選び、checkpoint、Qwen3-0.6B text encoder、Qwen Image VAEを指定します。

- [Anima-2.9B checkpoint](https://huggingface.co/Gazingstars123/Anima-2.9B)
- [Anima-3.8B checkpoint、Qwen3.5、adapter](https://huggingface.co/lylogummy/Anima-3.8B)
- [Forge Neo向けAnima 3.8B extension原典](https://github.com/GumGum10/forge-anima-3.8B)
- [Anima共通text encoder / VAE](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files)
- [CircleStone Labs公式Anima LoRA](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs)

### APIと実行状態の検証

高解像度toolと実モデルtestのため、次のAPIを追加しています。

```text
GET  /sdapi/v1/forge-model-status
POST /sdapi/v1/forge-model-status/ensure-loaded
```

loaded modelのarchitecture、transformer、VAE、text encoder、quantizationを返します。REST APIから選択式Scriptを呼ぶときは、そのScriptの引数範囲だけをoverlayし、他Scriptの既定値を壊さないようにしています。

## GUI中心の使い方

### Krea2画像を4Kへ仕上げる

1. `img2img`へ元画像を入れ、Krea2 checkpoint、VAE、text encoderが読み込まれていることを確認します。
2. `Script`から`Krea2 2-Stage Upscale`を選んでください。
3. 最初は`4K 納品`を選び、表示されたlive planを確認します。
4. denoising strengthは既定の低い値から開始します。
5. Smart Finishは有効、孤立粒補修は無効のまま比較してください。

入力が選択profileのproxy上限より大きい場合、暗黙に縮小せず開始前に停止します。Krea2 Raw、Turbo、customではnative解像度と推奨samplingが異なるため、profileを混同しないでください。

### B5 Whole-Tile Regeneration

このScriptは、縦向きJIS B5に近い入力を重複tileへ分け、各tileを拡大再生成して印刷向けの最終画像へ統合します。

1. `img2img`へB5判・縦の元画像を入れ、`Script`から`Krea2 B5 Whole-Tile Regeneration`を選びます。
2. `B5 4K 推奨設定を再適用`を押してください。
3. `JIS B5 2896×4096`のplanで作業倍率、stage数、tile設定、最大タイル数を確認します。
4. 変更したくない顔、文字、ロゴなどは、`保護楕円（元画像pixel座標）`の表へ追加してください。
5. 上部の通常img2img設定と、Script内のExact Steps、Denoise、合成方式を確認します。
6. 通常の`Generate`を押し、完了後に顔、文字、tile境界、印刷向けdetailを原寸で確認してください。

既定の1024×1448入力、作業倍率2×、1 stageでは、planは165 tile passを示し、最終画像を2896×4096へ仕上げます。入力の縦横比がB5から大きく外れる場合は、歪ませず開始前に停止します。

推奨設定は、作業倍率`2×（推奨）`、生成stage`1`、最大タイル数`256`、タイル用プロンプト`安全な局所復元ガイド（推奨）`、Exact Steps`6`、Denoise`0.35`、合成方式`低周波を元画像へ固定（推奨）`です。

保護表は`名前 / 左 / 上 / 右 / 下 / フェザー`の6列です。元画像のpixel座標で記入するため、拡大後の座標へ手計算で変換する必要はありません。処理中と最終仕上げの双方で、指定した楕円領域を元画像由来の画素へ戻します。

既定では最終PNGを通常のimg2img保存先へ出力し、同じbasenameのJSON manifestを隣へ保存します。`中間stage PNGも保存`は既定で無効です。必要な実行だけ詳細設定から有効にしてください。

たとえば、提示された2人の顔はGUIの表へ次のように入力します。

| 名前 | 左 | 上 | 右 | 下 | フェザー |
|---|---:|---:|---:|---:|---:|
| `character_a_face` | 185 | 335 | 355 | 540 | 24 |
| `character_b_face` | 670 | 370 | 840 | 585 | 24 |

### MiniMax H3で音声付き動画を作る

1. `H3 Studio`タブを開き、`実行環境とモデル`でローカルComfyUIを確認します。
2. `テキスト`、`キーフレーム`、`参照素材`からモードを選んでください。
3. Aspect、Quality、Duration、Steps、Seedを設定します。
4. `映像＋音声を生成`を押します。
5. 完成後、`最近の生成`でMP4と保存済み設定を確認してください。

参照素材モードでは、画面に表示された`<Picture 1>`、`<Video 1>`、`<Audio 1>`などのタグをpromptへ正確に記述します。未使用、未知、表記違いのタグは生成前にエラーになります。

## upstreamから継承する機能

通常のForge Neo機能とモデル対応は[Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)から継承しています。

同期基準`6009ffff`から取り込んだ主な改善には、Comfy Kitchen attention、CFG++ sampler修正、Krea 2 Identity Edit、Qwen3-VL vision入力、reference latent管理、VAE encode/decode共通化、Anima-2.9Bの基本対応などがあります。

Aikimi Neoは、その同期で削除対象となったBnB/NF4/GGUF互換経路を意図的に維持し、独自のKrea2、MiniMax H3、HyperWeave、高解像度、Anima LoRA処理と共存させています。

## 制限と安全上の注意

- モデル、dataset、checkpoint、生成画像、動画はGit管理対象ではありません。
- モデルと生成物の利用条件は、各配布元の最新ライセンスを確認してください。
- MiniMax H3 Studioを使う前に、対応するローカルComfyUI runtimeとモデルを別途用意してください。
- SenseNova U1.5正式版では、約16.52 GiBのINT8 ConvRot checkpointに加え、CPU RAM、GPU activation、画像token、decoder用のメモリも必要になります。
- SenseNovaのINT8 ConvRotはコミュニティ管理の変換weightであり、公式BF16と同一の品質や数値一致は保証されません。
- SenseNovaの参照画像数と入力解像度を増やすと、三分岐cacheと画像token列が急増します。RTX 3090では2参照、入力各512²、出力2048²、1 Stepを完走確認済みです。1664×2496でも参照入力を各512²へ縮小すれば同等画素数なので、24GB Safeの範囲に収まります。
- H3のContext-IRと2K Regenerateは外部有料API用であり、このStudioは呼び出しません。
- 4K/8K処理はVRAMだけでなくCPU RAM、一時disk、長い処理時間を必要とします。
- 高解像度処理は、まず4Kと低いdenoising strengthで構図と顔を確認してください。
- 8K、全候補保存、全debug出力を同時に有効にすると、一時disk使用量が大きくなります。
- 生成的upscaleは入力にない細部を推定します。顔、文字、ロゴ、精密形状は必ず原寸で確認してください。
- Animaの40→28 LoRA変換は追加12層を破棄するため不可逆です。
- Animaの52→40と52→28 LoRA変換も追加層を破棄するため不可逆です。
- sparseなAnima LoRAは28層版、40層版、52層版のいずれかを安全に断定できないため自動変換しません。
- Anima 3.8BのINT8 ConvRotはコミュニティ変換であり、配布元のBF16と同じ数値結果にはなりません。AdaLN、埋め込み、入出力、正規化層はBF16のまま保持します。
- Krea2でCFG 1.0を使う場合、negative promptは実質的に使われません。重要な条件はpositive promptにも書いてください。

## テスト

通常のテストはGPU、checkpoint、外部downloadを要求しません。

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tools\tests -p "test_*.py"
```

対象を絞る例:

```powershell
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_anima_lora
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_minimax_h3_bridge
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_sensenova_u15_bridge tools.tests.test_sensenova_u15_worker
.\venv\Scripts\python.exe -m unittest discover -s tools\tests -p "test_hyperweave_*.py" -v
```

Anima、Krea2、HyperWeaveには実checkpointまたは起動中のForge APIを使うopt-in live testもあります。明示的に有効化しない限り通常のtest discoveryではskipされます。

変更したPythonを構文確認する場合:

```powershell
.\venv\Scripts\python.exe -m compileall -q backend modules modules_forge scripts tools extensions-builtin
```

## Creditsとライセンス

Aikimi Neoは、次のプロジェクトと関連コントリビューターの成果を生成基盤として利用しています。

- [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)
- [SenseNova U1.5正式版](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [starsFriday/ComfyUI-SenseNova](https://github.com/starsFriday/ComfyUI-SenseNova)
- [joyfox/SenseNova-U1.5-8B-MoT-FP8](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8)

コードのライセンスは[LICENSE](LICENSE)を参照してください。モデルweight、VAE、text encoder、LoRA、生成物には、それぞれの配布元が定める別の条件が適用される場合があります。
