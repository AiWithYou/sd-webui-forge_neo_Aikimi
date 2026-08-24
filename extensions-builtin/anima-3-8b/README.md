# Anima 3.8B for Forge NeoW

Anima 3.8Bの52層DiTとQwen3.5 4B progressive cross-attention adapterを、Forge NeoWの通常の`txt2img`と`img2img`で使うためのbuilt-in extensionです。配布元のComfyUI workflowと同じく、Anima標準のQwen3-0.6B conditioningへQwen3.5の意味特徴を加えます。

実装は[GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B)のcommit `8af9bb4d391787030cb84205c47cf3ea1213795a`を基にしています。NeoWへの同梱時に、52層checkpointの必須検査、offline tokenizerの優先読込、INT8 ConvRotセットアップ、28↔40↔52 LoRA変換を追加しました。

## セットアップ

リポジトリ直下の次のファイルを実行してください。

```text
download_anima38_int8_convrot_models.bat
```

セットアップは、[lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B)の固定revisionから必要ファイルを取得します。一時的に取得したBF16 DiTのうち、主要なattentionとMLPの520行列だけがINT8 ConvRotへの変換対象です。AdaLN、埋め込み、入出力、正規化層はBF16のまま保持され、完成checkpointには再検証用のSHA-256 sidecarが付きます。

| 役割 | ファイル | 精度 |
|---|---|---|
| 52層DiT | `models/Stable-diffusion/Anima-3.8B-int8-convrot.safetensors` | 主要520行列がINT8 ConvRot、残りがBF16 |
| 意味encoder | `models/text_encoder/qwen35_4b.safetensors` | 配布済みのFP8/BF16混合 |
| 専用adapter | `models/text_encoder/Anima-3.8B-expanded_adapter.safetensors` | BF16 |
| Anima標準encoder | `models/text_encoder/qwen_3_06b_base.safetensors` | Anima共通 |
| VAE | `models/VAE/qwen_image_vae.safetensors` | Anima共通 |

Qwen3.5 tokenizerはextension内に同梱しています。生成時にHugging Face cacheや外部ネットワークへ接続しません。

## 使い方

Forgeを再起動し、次の値を選びます。

```text
Preset: anima
Checkpoint: Anima-3.8B-int8-convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen_3_06b_base.safetensors
Diffusion in Low Bits: Automatic
```

`Anima 3.8B (Qwen3.5)`を開いて有効にし、adapterを選びます。`Adapter strength`の学習時基準は`1.0`です。`0.0`を指定すると、Anima標準conditioningだけが使われます。

negative promptは既定でAnima標準encoderだけを使います。`Use adapter on negative prompt`を有効にすると、negative側のstrengthを個別に指定できます。

Qwen3.5の約1.2 GiBのembedding tableはCPUに保持し、promptで使うtoken行だけをGPUへ送ります。同じpromptでSeedだけを変えた場合は、ForgeのPersistent Cond Cacheがadapter設定とweight fingerprintを含む条件でconditioningを再利用します。

通常はtext encoderをGPUへ残す方が高速です。VRAMを優先する場合は`Low VRAM: offload text encoders before sampling`を有効にしてください。RTX 3090の実測では、生成後のTorch activeが9,766,217,180 bytesから4,247,712,928 bytesへ減り、約5.14 GiBを解放しました。一方、新しいpromptではencoderの再読込時間が加わります。

最初の生成条件は次を目安にしてください。

```text
Resolution: 832x1216前後の約1MP
Sampler / Scheduler: res_multistep + Beta
Steps: 28-50
CFG: 7-8
Adapter strength: 1.0
```

positive promptは、品質tagの後に`Description:`を置いて自然文を続けます。複数人物を指定する場合は代名詞を避け、人物名、位置、動作を文ごとに明示してください。

```text
masterpiece, best quality, high quality, newest,
Description:
A fox-girl jumps over a high fence.
```

## 読込と安全条件

- safetensors metadataに`anima_progressive_qwen35_cross_adapter_v1`を持つadapterだけを検出対象とします。
- Qwen3.5の候補は、ファイル名に`qwen35_4b`、`qwen3.5-4b`、`qwen3_5_4b`のいずれかを含むものです。
- 52層以外のAnima checkpointにはadapterを適用せず、組み合わせが誤っていれば理由を表示して停止する設計です。
- Qwen3.5はAnima標準encoderに追加して読み込むため、conditioning中のVRAMとRAM使用量、待ち時間が増えます。
- Qwen3.5の未使用final projectionと中間tensor copyを省き、paddingなしpromptではPyTorchのcausal SDPA経路を使う実装です。
- 同一promptのconditioning cacheはadapter、Qwen3.5、strength、negative設定が変わると無効になります。
- adapter名、positive strength、任意のnegative strengthは画像の生成parameterへの記録対象です。
- encoder offloadを有効にした場合は、解放量も生成parameterへ記録します。
- extensionを無効にした生成では、Forge標準のAnima処理を変更しません。

RTX 3090で保存済みの832×1216、32 Steps、CFG 7.0、同一prompt・Seedを再生成した比較では、wall timeが29.770秒から25.010秒へ短縮しました。平均RGB差は5.19/255で、人物、透明レインコート、蝶、左右の花、じょうろ、温室の配置を目視で維持していることを確認しました。

## ライセンスと由来

extensionコードのライセンスは[LICENSE](LICENSE)を参照してください。同梱tokenizerの由来は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)に記録しています。モデルweightには、Anima、Anima-2.9B、Qwen3.5、配布repositoryが定める条件が別途適用されます。
