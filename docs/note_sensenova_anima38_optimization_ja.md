# RTX 3090でSenseNova U1.5とAnima 3.8Bを実用域へ――画質を守る高速化、低VRAM化、LoRA互換の記録

画像生成モデルは、動けば終わりではありません。

毎日の生成作業では、生成時間、VRAM使用量、モデル切り替え、LoRA互換、失敗時の後始末まで整って、ようやく「使える」状態になります。

今回は、Windows 11とRTX 3090 24GBの環境で、Forge NeoWにSenseNova U1.5とAnima 3.8Bを組み込み、次の改善を行いました。

- SenseNova U1.5の2048×2048画像編集を、出力を変えずに32.3%高速化
- SenseNova公式8-Step LoRAとQuality 50-Stepを用途別に分離
- Anima 3.8BのQwen3.5常駐VRAMを約1.18GiB削減
- Anima 3.8Bの32 steps生成を約16.0%高速化
- 選択式のLow VRAMモードで、sampling前に約5.14GiBを解放
- 通常のAnima LoRAを28層・40層・52層間で安全に変換
- 実際のLoRAを使い、ON/OFF、Hires 2倍、異種LoRAの誤検出まで確認

先に書いておくと、すべてを「軽量化」の一語でまとめるのは危険です。

最適化には、画素を変えずに処理時間を短縮する変更と、量子化のように数値結果を変える変更があります。速度を優先する蒸留LoRAと、編集品質を優先するbase modelも同列には扱えません。

この記事では、その境界を分けて書きます。

## 検証環境

主な環境は以下です。

```text
OS: Windows 11
Shell: PowerShell 7.6
GPU: NVIDIA GeForce RTX 3090 24GB
UI: Stable Diffusion WebUI Forge NeoW
Python: 3.13.14
PyTorch: 2.11.0 + CUDA 13.0
```

対象モデルは、[SenseNova U1.5公式モデル](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)と、[Anima 3.8B Pro52 / Qwen3.5 Edition](https://huggingface.co/lylogummy/Anima-3.8B)です。

SenseNovaのINT8 ConvRot weightと専用loader、Anima 3.8BのINT8 ConvRot checkpointはコミュニティ管理の変換物です。公式BF16 weightと同一の数値結果を保証するものではありません。

一方、この記事で「最適化前後も画素が変わらない」と書く場合は、同じ変換checkpointを使った旧実装と新実装の比較を指します。

## SenseNova U1.5で最初に詰まった点

SenseNova U1.5は、通常のStable Diffusion系checkpointとしてForgeのsamplerへ接続できるモデルではありません。

画像token化、画像decoder、三分岐guidance、生成ループに専用実装が必要です。公式のreference inferenceも、テキストからの生成と画像編集を専用スクリプトで実行します。

そこでForge側は、通常のKSamplerへ無理に変換せず、次の境界に分けました。

```text
Forge UI
  ├─ 入力検証
  ├─ 参照画像の順序管理
  ├─ 保存・キャンセル・進捗表示
  └─ 隔離workerを起動
          └─ SenseNova専用ランタイムで生成
```

この分離により、SenseNova固有の推論契約を維持しながら、ForgeのUIと運用機能だけを利用できます。

## SenseNovaのMoT分岐を、レイヤー単位で必要な分だけGPUへ送る

最大の改善点は、MoT分岐の転送方法でした。

旧実装では、GPUメモリを節約するためにCPU offloadを使っていましたが、理解系と生成系のweightを必要以上に往復させる場面がありました。

新実装では、各decoder layerのtensorを次の3種類に分類します。

```text
shared
understanding branch
generation branch
```

レイヤー実行時の流れは以下です。

```text
1. shared tensorをGPUへ送る
2. その時点で使うbranchだけをGPUへ送る
3. layer forwardを実行する
4. layer終了後、元のCPU storageへ戻す
5. 次のlayerへ進む
```

未知のkey、分岐が混在するlayer、想定外の例外が発生した場合は、最適化を強行しません。安全側へ倒し、例外時にも元のstorageを復元する設計です。

この変更は、使う計算を近似したものではありません。weightの置き場所と転送順序だけを変えています。

## SenseNovaの実測結果

RTX 3090で、参照画像2枚、入力各512²、出力2048×2048、画像編集1 stepを比較しました。

| 項目 | 改善前 | 改善後 | 結果 |
|---|---:|---:|---:|
| worker処理時間 | 28.933秒 | 19.580秒 | 32.3%短縮 |
| 出力PNG | 基準 | SHA-256同一 | 画素変化なし |
| Torch peak allocated | 未計測 | 4,202,799,104 bytes | 約3.91GiB |
| Torch peak reserved | 未計測 | 4,573,888,512 bytes | 約4.26GiB |
| OOM / retry | 未計測 | 0 / 0 | 完走 |

新実装では、branch別転送量、model load、前処理、sampling、decode、Torch peak、OOM、retryも生成metadataへ残します。

各工程の処理時間とメモリ消費を記録し、ボトルネックを後から追えるようにしました。

## 8-StepとQualityを同じボタンにしない

SenseNova公式は、テキスト生成向けの[8-Step蒸留LoRA](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)を公開しています。

公式推奨設定は以下です。

```text
Steps: 8
CFG Scale: 1.0
CFG Norm: none
Timestep Shift: 3.0
```

このLoRAは高速ですが、用途はText-to-Imageです。画像編集へそのまま適用するものではありません。

そこでUIを次の2プロファイルへ分けました。

```text
公式8-Step・高速T2I
Quality 50-Step
```

参照画像を追加して画像編集へ切り替えると、Quality 50-Stepへ自動的に戻ります。LoRAが未取得の場合も、壊れた高速設定へfallbackせず、Qualityを明示的に選択します。

512×512の公式8-Step実生成では、worker 25.305秒、Torch peak allocated約1.77GiB、OOM 0で完走しました。

ここで重要なのは、8-Stepを「画質を落とさない高速化」と呼ばなかったことです。蒸留LoRAは速度と品質の選択肢であり、分岐ストリーミングの出力を変えない最適化とは別物です。

<!-- note画像差し込み
素材: outputs/sensenova_u15_live_smoke/sensenova_u15_20260824_103903_ce36b3a4.png
キャプション: SenseNova U1.5公式8-Step LoRA、512×512、8 steps、CFG 1、Shift 3
-->

## 24GB Safeを検証済みの制約として実装する

参照画像を増やすと、画像token列と三分岐cacheが増えます。UIに注意書きを出すだけでは、誤操作を防げません。

そのため`24GB Safe`では、検証済みの範囲を入力段階で強制します。

```text
出力: 2048² pixels以下
参照画像: 2枚以下
参照入力: 1枚あたり約0.26MP以下
```

縦横比を維持して縮小し、32pxグリッドで生じる余白は画像端の画素で埋めます。上限を外す場合は、別の`Uncapped streaming`を明示的に選ぶ設計です。

## Anima 3.8Bは「3.8Bだけ」を読み込むモデルではない

Anima 3.8Bは、52-block DiTだけで完結しません。

[公式モデルカード](https://huggingface.co/lylogummy/Anima-3.8B)では、次の組み合わせとして説明されています。

```text
52-block DiT
Qwen3.5 4B text encoder
progressive cross-attention adapter
Anima標準Qwen3 0.6B encoder
```

Qwen3.5の意味特徴を、専用adapter経由でAnima標準conditioningへ加えます。Qwen3.5 4Bは別componentなので、prompt処理中のVRAMと読み込み時間も増えます。

推奨開始点は、832×1216前後の約1MP、adapter strength 1.0、CFG 7〜8、28〜50 steps、`res_multistep + Beta`です。

## DiTだけをINT8 ConvRotへ変換する

Anima 3.8Bの軽量版では、BF16のDiT全体を無差別に量子化していません。

attentionとMLPの主要520行列をINT8 ConvRotへ変換し、AdaLN、埋め込み、入出力、正規化層はBF16のまま残しました。

| 対象 | 変換前 | 変換後 | 削減率 |
|---|---:|---:|---:|
| Anima 3.8B DiT | 7,504,189,974 bytes | 4,238,326,342 bytes | 約43.5% |

Qwen3.5の配布weightは、FP8とBF16の混合精度です。こちらは再量子化せず、そのまま利用します。

## 1.2GiBのembedding tableをGPUへ常駐させない

Qwen3.5側で大きかったのは、語彙全体のembedding tableです。

promptで使うtokenは語彙の一部なのに、table全体をGPUへ置くと約1.2GiBを消費します。

そこで、embedding tableをCPU上の非登録weightとして保持し、promptに含まれるtoken rowだけをGPUへ送る`CPUEmbedding`を実装しました。

加えて、次の無駄を削減しました。

- 推論で使わないfinal projectionを転送しない
- Qwen中間出力の不要なcloneを作らない
- paddingなしの単一promptでは、巨大な明示的causal maskを作らずSDPAを使う

## cacheはprompt文字列だけで判定しない

conditioning cacheを高速化に使う場合、promptだけをkeyにすると危険です。

同じファイル名でも、Qwenやadapterの中身が差し替わる可能性があります。そこでcache signatureへ次を含めました。

```text
architecture
Qwen path / size / mtime
adapter path / size / mtime
positive strength
negative strength
通常LoRA設定
```

同じpromptでseedだけを変える場合はcacheを再利用できます。weightを差し替えた場合は、同じパスでも古いconditioningを使いません。

## Low VRAMは既定で有効にしない

conditioningが完成した後は、sampling中にQwen3.5とAnima標準encoderを使いません。

そこで、`Low VRAM: offload text encoders before sampling`を追加しました。

RTX 3090での実測は以下です。

```text
offload前 Torch active: 9,770,018,248 bytes
offload後 Torch active: 4,251,513,996 bytes
解放量: 5,518,504,252 bytes ≒ 5.14GiB
```

ただし、次のpromptではencoderを再読み込みします。VRAMに余裕がある環境では、GPUへ残した方が速いです。

そのため、Low VRAMは既定でOFFにしました。速度とVRAMのどちらを優先するか、利用者が明示的に選びます。

## Anima 3.8Bの実測結果

主な改善結果です。

| 項目 | 改善前 | 改善後 | 効果 |
|---|---:|---:|---:|
| Qwen GPU常駐量 | 4,557.58MiB | 3,345.08MiB | 1,212.50MiB削減 |
| 初回Qwen GPU転送 | 118.86秒 | 88.66秒 | 25.4%短縮 |
| 832×1216、32 steps | 29.770秒 | 25.010秒 | 16.0%短縮 |
| Low VRAM時のTorch active | 約9.10GiB | 約3.96GiB | 約5.14GiB解放 |

32 stepsの新旧画像には差があります。平均RGB差は5.19/255、PSNRは26.36dBでした。

人物、透明レインコート、蝶、左右の花、じょうろ、温室の配置は目視で維持していました。Anima 3.8B側は、速度と見た目を組み合わせて評価しました。

## 52-blockだけを最適化し、28-blockと40-blockを巻き込まない

Animaには28-block、Anima 2.9Bには40-block、Anima 3.8Bには52-blockの構成があります。

tensor fusion、`addcmul`、in-place residualなどの最適化を共通コードへ入れる場合、block数を限定しないと既存モデルへ影響します。

そこで、Anima 3.8B向け最適化は52-block構成だけで有効にしました。28-blockと40-blockは従来経路を維持し、専用テストで分離を確認しています。

生成成功時だけでなく、例外時にも一時flagとmonkeypatchを必ず解除します。生成後に別モデルへ切り替えたとき、Anima用の状態を残さないためです。

## 通常のAnima LoRAを3.8Bでも使う

次に必要だったのが、通常Anima LoRAとの互換性です。

Anima 3.8Bの専用adapterと、一般的なAnima LoRAは役割が異なります。前者はQwen3.5 conditioning用、後者はDiTやAnima標準encoderへ適用する追加weightです。

UIへ`Standard Anima LoRA`とstrengthを追加し、Forge標準LoRA経路へ接続しました。

変換対象は、完全な28層・40層・52層LoRAです。

```text
28 → 40 → 52
28 → 52
40 → 52
```

追加blockは、公開された拡張manifestに対応するsource blockのtensor storageを共有します。変換だけのためにtensor本体を複製しません。

sparse LoRAは、どのlayoutか安全に断定できないため自動変換しません。

## block数だけでAnima LoRAと判定すると危険だった

最初の候補検出では、28・40・52というblock coverageを中心に判定していました。

ブラウザでdropdownを開くと、WanやKrea2など、block数が同じ別アーキテクチャのLoRAまで候補に混ざりました。

そこで、次の条件を追加しました。

- Animaを示すsafetensors metadata
- AdaLN modulation
- Anima形式のcross-attention projection
- Anima形式のself-attention projection
- Anima形式のMLP key

同名LoRAが複数の追加フォルダーにある場合は、Forge標準registryと同じ優先順位で解決します。

この修正は、unit testだけでは見つけにくい問題でした。実際の複数LoRAフォルダーを読み込んだ画面を確認したことで発見できました。

## 実LoRAで確認したこと

互換コードは、syntheticなtest tensorだけで終わらせませんでした。

### 748cm_TA_EP4

```text
Source layout: 28-block
Target layout: 52-block
UNet loaded keys: 832
Skipped keys: 0
Strength: 0.8
Trigger: @748cm_style
```

ONでは顔、線、黒い衣服、ネオン反射の作風が強く変わり、構図にも影響しました。style LoRAは、見た目だけでなくカメラ距離や小物の扱いまで変える場合があります。

<!-- note画像差し込み
素材: outputs/anima_38b_smoke/748cm_TA_EP4_newprompt_lora_off_on_horizontal.png
キャプション: 748cm_TA_EP4、同一prompt・seed、左LoRA OFF、右LoRA ON 0.8
-->

### SimpleAnima

`SimpleAnima.safetensors`は、手元の`anima_v1_quality_lift_ileco.safetensors`とSHA-256が同一でした。trigger不要の品質向上ILECOです。

```text
Source layout: 28-block
Target layout: 52-block
UNet loaded keys: 520
Skipped keys: 0
Strength: 1.0
```

ONでは、人物配置、顔、襟、机、本棚などの構造が整理されました。主な効果は、情報設計と陰影を整える変化です。

<!-- note画像差し込み
素材: outputs/anima_38b_smoke/SimpleAnima_newprompt_lora_off_on_horizontal.png
キャプション: SimpleAnima、同一prompt・seed、左LoRA OFF、右LoRA ON 1.0
-->

### Anima in realとHires 2倍

`Ani2rel`をtriggerに使うLoRAでは、アニメ人物を維持しながら、背景の建物、店舗照明、濡れた路面、通行人を写実寄りにできました。

Hires.fixは、最初に`Latent`、2倍、denoise 0.35、Hires 14 stepsを試しましたが、顔とコートが過度に平滑化され、縦方向のにじみも出ました。

最終的には以下を採用しました。

```text
Upscaler: RealESRGAN_x4plus
Scale: 2.0
Denoise: 0.20
Hires steps: 20
Output: 1664×2432
```

通常版を832×1216へ固定し、同じsemantic promptとseedでHires版を生成しました。Hires版を通常解像度へ縮小した比較は、SSIM 0.796311、PSNR 26.5060dBでした。

<!-- note画像差し込み
素材: outputs/anima_38b_smoke/Anima_in_real_v2_normal_vs_hires2x_comparison.png
キャプション: Anima in real、左832×1216、右Hires.fix 2倍。比較画像では右を同じ表示寸法へ縮小
-->

## 長時間運用に向けた後始末

モデルが一度動いても、生成を繰り返すと別の問題が出ます。

今回、次も修正しました。

- trackedな`LoadedModel.model_unload()`経由で解放する
- weakref finalizerの蓄積を防ぐ
- 成功時と例外時の両方でpatchを解除する
- 同じpathのweight差し替えをfingerprintで検出する
- `cudaMallocAsync`を使い、長時間運用時の断片化を抑える

`cudaMallocAsync`は、平均VRAMを魔法のように減らす設定ではありません。allocation stallや断片化への対策です。PyTorchの詳しい設定は[CUDA memory managementの公式ドキュメント](https://docs.pytorch.org/docs/main/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)を参照してください。

Forgeのcompile presetによっては、`max-autotune`や`reduce-overhead`と併用できません。allocatorとcompile modeは、どちらかを選ぶ必要があります。

## 検証はunit test、live API、目視を分ける

最終的な自動テスト結果です。

```text
Ran 701 tests in 21.239s
OK (skipped=15)
```

`skipped=15`は、通常テストでは実行しない実weightのlive testです。SenseNovaとAnima 3.8Bの対象live testは、環境変数を有効にして別途実行しました。

今回の検証は、次の3層へ分けています。

```text
unit test
  key変換、cache、cleanup、入力制約を高速に確認

live API test
  実checkpoint、実LoRA、GPU generationを確認

visual QA
  顔、手、構図、小物、文字、背景、比較画像を原寸確認
```

unit testが通っても、候補dropdownへWan LoRAが混ざる問題は見つかりません。生成APIが200を返しても、Hires画像が平滑化されすぎていれば採用できません。

終了コード、metadata、ログ、画像を別々に確認する必要があります。

## 今回の改善から得た教訓

### 1. 「画質を落とさない」を数値で定義する

SenseNovaの分岐ストリーミングは、PNGのSHA-256が同じことを確認しました。

Anima 3.8Bの高速化は画像差があるため、平均RGB差、PSNR、構図の目視維持を報告しました。

同じ「高速化」でも、証明方法は同じではありません。

### 2. offloadは多ければよいわけではない

text encoderをsampling前に外せばVRAMは減りますが、次のpromptで再読み込みが必要です。

速度優先なら常駐、VRAM優先ならoffloadという選択をUIへ出す方が、隠れた自動切り替えより扱いやすくなります。

### 3. fallbackは便利さより誤生成を増やす場合がある

8-Step LoRAがないときに別のLoRAへ切り替える、52-block checkpointがないときに40-blockで続行する、といったfallbackは実装していません。

必須assetがない場合は、理由を示して停止します。

### 4. 互換性は構造で判定する

Anima adapterはarchitecture metadata、通常LoRAはblock coverageとmodule signature、SenseNovaは固定checkpointとrevisionで判定します。

名前が似ているだけのファイルを受け入れないことが、長期的には一番安全でした。

## まとめ

RTX 3090 24GBでも、SenseNova U1.5とAnima 3.8Bは実用的に動かせます。

ただし、単純に量子化weightを置くだけでは不十分でした。

- 専用推論契約を壊さない
- 必要なbranchとtokenだけをGPUへ送る
- cacheへweight fingerprintを含める
- offloadを利用者が選べるようにする
- block layoutを確認してLoRAを変換する
- 画素が変わらない結果、数値差、目視差を分けて報告する
- 例外時にも状態を完全に戻す

これらをまとめて初めて、日常的に使える生成環境になります。

実装は[Aikimi Neo](https://github.com/AiWithYou/sd-webui-forge_neo_Aikimi)の`neo`ブランチに反映しています。

主要commitは次の2つです。

- [SenseNovaとAnimaの推論最適化](https://github.com/AiWithYou/sd-webui-forge_neo_Aikimi/commit/012785c040610506a038722a58b46c40ca2c67ec)
- [Anima 3.8Bの通常LoRA選択対応](https://github.com/AiWithYou/sd-webui-forge_neo_Aikimi/commit/ad49ad0072731dc47c74ba4e49c0e76f7f8a1f52)

## 関連リンク

- [SenseNova U1.5公式モデル](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [SenseNova U1.5公式8-Step LoRA](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)
- [SenseNova公式reference inference](https://github.com/OpenSenseNova/SenseNova-U1)
- [Anima 3.8B公式モデルカード](https://huggingface.co/lylogummy/Anima-3.8B)
- [Anima 3.8B Forge extension原典](https://github.com/GumGum10/forge-anima-3.8B)
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)

#画像生成 #ローカルAI #StableDiffusion #SenseNova #Anima #RTX3090 #LoRA
