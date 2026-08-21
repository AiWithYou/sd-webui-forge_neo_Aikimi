# SenseNova U1.5 Studio

SenseNova U1.5 Studioは、[sensenova/SenseNova-U1.5-8B-MoT-Preview](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview)をForge NeoWから実行する専用GUIです。テキスト生成、単一画像編集、複数画像編集、Q8_0 GGUF（INT8）、公式BF16に対応します。

SenseNovaはNEO-unify固有の画像token化、scheduler、画像decoder、生成ループを持ちます。このため、Forgeの通常checkpointやKSamplerへweightだけを読み込む方式は採用していません。Forgeは入力検証、画面、保存、キャンセルを担当し、生成処理は公式推論コードを読み込む隔離workerが担当します。

## セットアップ

Forge NeoW直下の次のファイルを実行してください。

```text
download_sensenova_u15_int8.bat
```

既定のセットアップ内容は次のとおりです。

| 項目 | 内容 |
|---|---|
| 公式推論コード | `OpenSenseNova/SenseNova-U1`の固定revision `12a2bd9cba22a5317164b55db4f7c6209a371f83` |
| 取得範囲 | `src/sensenova_u1`と公式`LICENSE` |
| tokenizer依存 | `sentencepiece==0.2.1` |
| Q8 weight | `SenseNova-U1.5-8B-MoT-Preview-Q8.gguf` |
| Q8配布元revision | `smthem/SenseNova-U1-8B-MoT-Merger-gguf`の`e63b0a7e483bffdb1ff0463a39fbfd04ad3c85d9` |
| Q8サイズ | 19,947,887,936 bytes（約18.58 GiB） |
| Q8 SHA-256 | `8b655046f6e22c22258607556cacee3c1d82ae534146fb9c0faba04a0e4b3c8f` |

Q8の取得を中断した場合は、同じbatを再実行してください。`.part`ファイルから再開します。既存の完成ファイルは、サイズ、GGUF header、SHA-256がすべて一致した場合だけ再利用します。

推論コードだけを準備する場合はPowerShell 7から次を実行できます。

```powershell
.\download_sensenova_u15_int8.ps1 -RuntimeOnly
```

Q8 weightだけを取得または再検証する場合は、次を実行します。

```powershell
.\download_sensenova_u15_int8.ps1 -ModelOnly
```

## 基本操作

### テキストから生成

1. `SenseNova U1.5`タブを開き、`テキストから生成`を選択します。
2. プロンプトを入力したら、公式解像度bucket、Steps、CFG、Seedを確認してください。
3. 初回は`1024 × 1024 · 動作確認`と1〜2 Stepsを使い、読み込み経路だけを確認します。
4. 設定を確定し、`画像を生成`を押すと実行が始まります。

モデルカードの推奨値は50 Steps、CFG 4.0、Timestep Shift 3.0です。1024×1024は短い動作確認用であり、公式の学習解像度bucketではありません。品質確認では2048×2048などの公式bucketへ戻してください。

### 複数画像を編集

1. `複数画像を編集`へ切り替え、`追加・差し替え画像`へ最初の画像を入れます。
2. `末尾へ追加`を押し、必要な画像を同じ手順で登録してください。
3. 順序を直す場合は画像を選択し、`← 前へ`または`後ろへ →`を使います。
4. プロンプトには「1枚目の人物」「2枚目の衣装」「3枚目の照明」のように、各画像の役割を明記します。
5. 最後に`入力1枚目の比率を維持 · 約4MP`または公式解像度bucketを選び、生成を開始してください。

画像は画面に表示された順序のまま`it2i_generate`へ渡します。プロンプトに`<image>`を1つも書かない場合は、公式実装が入力画像数分を先頭へ補います。自分で記述する場合は、入力順と個数を合わせてください。

入力画像予算の`自動`は、1〜2枚を各2048²まで、3枚以上を合計約8.39 MPの範囲で均等配分し、各画像の下限を512²にします。画像数を増やすほど、個々の参照画像へ割り当てる解像度は下がります。

## INT8とBF16

`INT8 · Q8_0 GGUF`は、量子化されたLinear weightをGGUFとして保持し、公式SenseNovaコードがdiffusersの`GGUFLinear`経路で演算時に復号します。画面上ではINT8と表記しますが、activationと計算精度は別です。既定ではBF16計算を使い、configとの不整合を防ぐためモデルIDをPreviewへ固定します。

Q8変換weightはコミュニティが管理しています。SenseNova公式の配布形式はBF16であり、Q8と公式BF16の品質一致や再現性は保証されません。モデルカードと公式ComfyUI実装が案内する対応経路に限定しており、任意のGGUFやbitsandbytes形式を自動変換しません。

`公式 BF16`を選ぶと、モデルIDから公式weightを取得します。初回downloadと保存容量が大きく、RTX 3090では`Low`または`Balanced`のlayer offloadが必要です。

## VRAMモード

| モード | 用途 |
|---|---|
| `Low` | 1 layerずつ同期転送。RTX 3090で最初に確認する場合の推奨設定。 |
| `Balanced` | 複数layerを先読みし、CPUからGPUへの転送と計算を重ねる方式。 |
| `Fast` | 生成layerをVRAM予算内で保持。十分な余裕がある環境向け。 |
| `Full` | モデル全体をGPUへ配置。24 GiB級GPUには非推奨。 |

生成開始時は通常のForge画像モデルを退避してVRAMを確保します。SenseNova workerは1回の生成後に終了し、weightとactivationをOSへ返すため、次回生成時はモデルを先頭から再読込します。

## 保存とキャンセル

完成したファイルは次へ保存します。

```text
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.png
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.json
```

JSONにはモデルID、量子化方式、公式コードrevision、解像度、Steps、CFG、Seed、入力画像数、入力順、出力SHA-256、処理時間を記録します。参照画像本体は一時jobフォルダーへ複製し、生成終了またはキャンセル後に削除します。

キャンセルは隔離workerを停止します。モデル読み込み中とsampling中のどちらでも停止できますが、次回はモデルを先頭から読み込みます。

## 既知の制限

- Previewモデルは正式版前の公開weightです。モデルカードが示す既知の品質制限があります。
- Q8 weightは約18.58 GiBです。weight以外にCPU RAM、GPU activation、画像token、decoder用メモリが必要です。
- 参照画像数と解像度を増やすと、入力token列と処理時間が増えます。
- 複数画像編集では、保持対象と変更対象をプロンプト内で明示してください。入力が多いほど、被写体や構図がdriftする可能性があります。
- `FlashAttention`を明示した場合、互換wheelが未導入なら生成前に失敗します。通常は`自動`または`PyTorch SDPA`を使ってください。
- Prompt Enhanceは外部モデル呼び出しを伴うため、このStudioからは自動実行しません。

## 出典とライセンス

- [SenseNova U1.5 Preview model card](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview)
- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)
- [公式ComfyUI実装のGGUF説明](https://github.com/OpenSenseNova/SenseNova-U1/tree/main/apps/comfyui)
- [Q8_0配布リポジトリ](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf)

公式コードとモデルカードはApache-2.0です。Q8変換weightを含む各配布物については、取得時点の配布元ライセンスも確認してください。
