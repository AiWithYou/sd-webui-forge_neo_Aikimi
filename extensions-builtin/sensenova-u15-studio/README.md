# SenseNova U1.5 Studio

SenseNova U1.5 Studioは、正式版の[sensenova/SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)をForge NeoWから実行する専用GUIです。テキスト生成と、単一または複数の参照画像を使った画像編集に対応します。

既定のweightは、正式版を基にしたコミュニティ配布のINT8 ConvRotです。SenseNova固有の画像token化、画像decoder、三分岐guidance、生成ループを保つため、Forgeの通常checkpointやKSamplerには接続しません。Forgeは入力検証、画面、保存、キャンセルを担当し、生成は隔離workerが専用ランタイムを使って実行します。

## セットアップ

Forge NeoW直下の次のファイルを実行してください。

```text
download_sensenova_u15_int8.bat
```

既定のセットアップ内容は次のとおりです。

| 項目 | 内容 |
|---|---|
| 対象モデル | 正式版`SenseNova-U1.5-8B-MoT` |
| ConvRotランタイム | `starsFriday/ComfyUI-SenseNova`の固定revision `e6dfd45762eb46f805067fe079c14bcb643ccccd` |
| checkpoint | `SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors` |
| 配布元 | `joyfox/SenseNova-U1.5-8B-MoT-FP8` |
| 配布元revision | `57de22ad4e2fc24c77f56dfe45dbb87a60dfebee` |
| ファイルサイズ | 17,734,813,848 bytes（約16.52 GiB） |
| SHA-256 | `cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e` |
| 量子化構成 | `int8_tensorwise`、ConvRot group size 256、588 Linear層 |

checkpointは32 MiB単位に分け、最大16接続で取得します。中断した場合は同じbatを再実行すると、完成済み部分と未完了chunkの先頭から続行できます。完成ファイルは、サイズ、safetensors内のConvRot署名、SHA-256の各検証値がすべて同じ場合だけ利用可能です。

ランタイムだけを準備する場合はPowerShell 7から次を実行します。

```powershell
.\download_sensenova_u15_int8.ps1 -RuntimeOnly
```

checkpointだけを取得または再検証する場合は、次を実行します。

```powershell
.\download_sensenova_u15_int8.ps1 -ModelOnly
```

## テキストから生成

1. `SenseNova U1.5`タブを開き、`テキストから生成`を選びます。
2. 解像度、Steps、CFG、Seedを確認してから、プロンプトを入力してください。
3. 初回の読み込み確認には、`1024 × 1024 · 動作確認`と1〜2 Stepsが適しています。
4. `画像を生成`を押すとForgeモデルが退避し、その後に専用workerが起動する流れです。

品質確認の開始値は50 Steps、CFG 4.0、Timestep Shift 3.0です。24GB Safeの出力上限は2048² pixelsで、2048×2048や同等画素数の公式解像度bucketを選べます。VRAMを抑えるため、参照画像だけを各512²へ縮小してモデルへ渡します。

## 複数参照画像を編集

1. まず`複数画像を編集`へ切り替えてください。
2. `複数画像を一括選択`ならまとめて登録でき、`追加・差し替え画像`なら1枚ずつ扱えます。
3. 表示順は、対象画像を選択して`← 前へ`または`後ろへ →`で調整します。
4. プロンプト内では、`Image-1`、`Image-2`のように各画像と役割を対応させてください。
5. 最後に出力解像度と入力画像予算を確認し、生成を始めます。

モデルとGUIの上限は64枚で、参照画像は表示順を保ったまま`it2i_generate`へ渡されます。24GB Safeは、参照2枚以下、出力2048²以下、参照入力各512²に限定した実測済みの保護枠です。3枚以上の参照や高忠実度入力が必要な場合は、大容量GPUで`Uncapped streaming`を選択してください。

参照画像を増やすほど、画像token列、CPU転送量、処理時間が増えます。RTX 3090では2参照、入力各512²、出力2048²、1 Stepが完走し、観測ピークは約7.72 GiB、最終runのworker処理は28.933秒でした。一方、2参照を自動入力予算、出力1664×2496で実行するとsampling前にVRAMが不足します。このため、24GB Safeは大きな参照入力だけをモデル読込前に拒否します。

## INT8 ConvRot

このcheckpointでは、588個のLinear層を`int8_tensorwise`形式で保持します。各層はHadamard回転を戻しながらBF16へ復号され、Lowモードでは対象層だけをGPUへ移して演算後にCPUへ戻します。activationと演算精度はBF16です。

正式版のモデル構成とtokenizerを使いますが、INT8変換weightと専用ローダーはコミュニティが管理しています。公式BF16との品質一致や数値一致は保証されません。このStudioは固定した配布ファイルだけを完全性確認し、任意のGGUF、bitsandbytes、別のsafetensorsを自動変換しません。

pruned checkpointはテキスト出力用の`language_model.lm_head`を削除しています。テキストからの画像生成と画像編集には使えますが、Think mode、VQAのテキスト回答、テキストと画像の交互出力には使えません。

## VRAMモード

| モード | 用途 |
|---|---|
| `24GB Safe` | Transformerを1層ずつ転送し、出力2048²以下、参照2枚以下、参照入力各512²を強制するRTX 3090向け設定。 |
| `Uncapped streaming` | 同じ層streamingを使いながら画素数制限を解除。大容量GPU向けの実験設定で、実行可能性は条件ごとに確認が必要。 |
| `Full GPU` | 量子化weight全体をGPUへ配置する実験設定。必要量は解像度と参照数で変わり、24GB GPUでは利用不可。 |

生成開始時は通常のForge画像モデルを退避してVRAMを確保し、SenseNova workerは1回の生成後にweightとactivationをOSへ返します。この設計によりメモリを確実に解放できますが、次回の生成時にはモデルを先頭から読み込みます。

## 保存とキャンセル

完成ファイルは次へ保存します。

```text
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.png
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.json
```

JSONにはモデルID、固定revision、checkpoint、量子化方式、ロードしたINT8層数、解像度、Steps、CFG、Seed、入力画像数、入力順、出力SHA-256、処理時間を記録します。参照画像本体は一時jobフォルダーへ複製し、生成終了またはキャンセル後に削除します。

キャンセルは隔離workerを停止します。モデル読み込み中とsampling中のどちらでも停止できますが、次回はモデルを先頭から読み込みます。

## 既知の制限

- checkpointだけで約16.52 GiBあり、CPU RAM、GPU activation、画像token、decoderにもメモリが必要です。
- 参照画像を増やすほど、1枚ごとの入力解像度は下がります。
- 多数の参照画像を使う場合は、保持対象と変更対象をプロンプト内で明示してください。
- `FlashAttention`を指定した場合、互換wheelがない環境では生成前に失敗します。通常は`自動`または`PyTorch SDPA`を使ってください。
- 公式8-step LoRAはテキスト生成専用です。このStudioではまだ読み込みません。
- Prompt Enhanceは外部モデル呼び出しを伴うため、自動実行しません。

## 出典とライセンス

- [SenseNova U1.5正式版モデル](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [INT8 ConvRot配布リポジトリ](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8)
- [ConvRot対応ランタイム](https://github.com/starsFriday/ComfyUI-SenseNova)
- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)

ランタイムコードはApache-2.0です。モデルweightは配布元に記載されたライセンスへ従ってください。
