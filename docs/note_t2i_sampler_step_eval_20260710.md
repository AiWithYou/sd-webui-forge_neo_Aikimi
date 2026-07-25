# Krea2 t2iで「ノイズが増えにくい」サンプラー設定を探したメモ

画像生成の t2i 設定で、サンプラーと step 数をいろいろ振ってみました。

目的は単純で、「細部は残したいけれど、肌・服・背景・暗部のザラつきは増やしたくない」という設定を探すことです。

今回の結論から書くと、自分の環境ではこれが一番バランスよく見えました。

```text
Sampler: DPM++ 2M SDE
Schedule type: Simple
Steps: 4
CFG Scale: 1.0
Distilled CFG Scale: 1.15
Size: 1024x1536
Hires.fix: Off
Batch Size: 1
Negative prompt: 空
```

意外だったのは、step を増やせば増やすほど良くなるわけではなかったことです。今回使った Krea2 turbo 系のモデルでは、6 step、8 step、10 step と伸ばすほど線や質感は増える一方で、滑らかな面のザラつきも増えていく傾向がありました。

## 検証環境

今回の検証条件は以下です。

```text
UI: Forge Neo
Model: turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors
Preset: krea
Diffusion in Low Bits: bnb-nf4
VAE: qwen_image_vae
Text Encoder: qwen3vl_4b_bf16
GPU: RTX 3090 24GB
解像度: 1024x1536
CFG Scale: 1.0
Distilled CFG Scale: 1.15
Schedule type: Simple
Seed: 固定
```

プロンプトは、髪・肌・白い服・暗めの背景・細かい小物が出るようにして、サンプラー差が見えやすい構成にしました。

最初に 28 条件を比較し、そのあと暗部と雨の反射がある別プロンプトで上位候補を 5 条件だけ再チェックしました。

## 比較したもの

メイン検証では、以下のサンプラーと step 数を試しました。

```text
Euler: 4, 6, 8, 10, 12
DPM++ 2M: 4, 6, 8, 10, 12
DPM++ SDE: 4, 6, 8
DPM++ 2M SDE: 4, 6
Euler a: 4, 6, 8, 10
UniPC: 4, 6, 8, 10
LCM: 4, 6, 8
ER SDE: 4, 6
```

全体の比較画像は、公開時にはここに貼ると分かりやすいです。

```text
画像候補: contact_sheet_sampler_steps.jpg
```

上位候補だけを並べた画像も作りました。

```text
画像候補: top12_contact_sheet.jpg
```

## 結果

メイン検証の上位はこうなりました。

| 順位 | Sampler | Steps | Score | Noise | Speckle | Detail |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DPM++ 2M SDE | 4 | 0.8015 | 2.3469 | 0.8700% | 6.7687 |
| 2 | LCM | 4 | 0.7899 | 2.2253 | 0.6865% | 5.1134 |
| 3 | LCM | 6 | 0.7862 | 2.3704 | 0.6101% | 5.6260 |
| 4 | Euler a | 6 | 0.7402 | 2.5309 | 0.8877% | 6.3517 |
| 5 | Euler | 6 | 0.7259 | 2.6167 | 1.0662% | 7.1071 |
| 6 | DPM++ 2M | 4 | 0.7196 | 2.8247 | 1.1268% | 8.3969 |

数値だけ見ると LCM もかなり強いです。実際、ノイズは少ないです。

ただし目視すると、LCM は顔・手・服の細部が少し溶けやすく、最終絵としてはやや薄く感じました。プレビュー用途や、滑らかなラフを早く出す用途には良さそうですが、今回の「ノイズを抑えつつ画質も欲しい」という目的では、DPM++ 2M SDE の 4 steps が一番扱いやすいと感じました。

## 暗部での再チェック

ノイズは暗部で目立ちやすいので、雨の夜道・黒い服・濡れた路面反射が入る別プロンプトでも確認しました。

```text
画像候補: crosscheck_contact_sheet.jpg
```

暗部クロスチェックの結果です。

| 順位 | Sampler | Steps | Score | Noise | Speckle | Detail |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DPM++ 2M SDE | 4 | 0.8488 | 1.6952 | 0.3496% | 5.4535 |
| 2 | LCM | 6 | 0.7090 | 1.8406 | 0.3393% | 4.4068 |
| 3 | Euler | 6 | 0.5848 | 2.0121 | 0.4711% | 5.7567 |
| 4 | Euler | 8 | 0.2914 | 2.3208 | 0.6381% | 5.9833 |
| 5 | DPM++ 2M | 4 | 0.2000 | 2.5258 | 0.7560% | 8.1668 |

ここでも DPM++ 2M SDE / 4 steps が一番良かったです。

DPM++ 2M / 4 steps は細部が強いのですが、暗部や服の面でザラつきが増えやすい印象でした。シャープさを優先したいときには候補になりますが、低ノイズを重視するなら少し扱いにくいです。

## 実用上の使い分け

今回の結果から、自分なら以下のように使い分けます。

### 本命

```text
DPM++ 2M SDE / Simple / 4 steps
```

低ノイズで、暗部も荒れにくく、LCM より細部が残ります。今回の検証では一番バランスが良かったです。

### 安定フォールバック

```text
Euler / Simple / 6 steps
```

DPM++ 2M SDE ほど低ノイズではありませんが、構図・顔・全体のまとまりが安定していました。迷ったらこれでも十分使えます。

### 高速プレビュー

```text
LCM / Simple / 6 steps
```

ノイズは少ないです。ただし細部は弱めなので、最終出力というより候補出しや滑らかな下絵向きです。

### シャープ優先

```text
DPM++ 2M / Simple / 4 steps
```

線や服の質感は強く出ます。ただしザラつきも増えやすいので、肌や暗部をきれいにしたい場合は注意が必要です。

## 避けた方がよさそうだった設定

今回のモデルでは、以下はあまり良い結果になりませんでした。

```text
Euler 8 steps以上
DPM++ 2M 6 steps以上
UniPC 6 steps以上
DPM++ SDE
10-12 steps全般
```

もちろんモデルやプロンプトが変われば結果も変わります。ただ、少なくとも今回の Krea2 turbo 系では「step を増やすほど高画質」というより、「step を増やすほど細部と一緒にザラつきも増える」という挙動でした。

## 最終設定

しばらくはこの設定を基準にします。

```text
Sampler: DPM++ 2M SDE
Schedule type: Simple
Steps: 4
CFG Scale: 1.0
Distilled CFG Scale: 1.15
Size: 1024x1536
Hires.fix: Off
Batch Size: 1
```

必要に応じて、安定重視なら Euler 6、シャープ重視なら DPM++ 2M 4、プレビューなら LCM 6 を使う感じです。

## 注意点

今回の結果は、あくまで以下の条件での実測です。

```text
Krea2 bnb-nf4
turbo_gpt0630_krea2_final_forge_bnb_nf4
CFG 1.0
Distilled CFG 1.15
1024x1536
Forge Neo
```

別の checkpoint、別の VAE / text encoder、別の LoRA、別の CFG では結果が変わる可能性があります。

また、CFG Scale 1.0 では negative prompt が実質効きません。ネガティブプロンプトで制御したい場合は、CFG を 1.2 から 1.5 くらいに上げて、別途比較した方がよさそうです。

自分の用途では、Krea2 turbo 系の t2i は「少ない step で当たりを探して、良い seed を後工程で育てる」方が合っていると感じました。高 step で一発仕上げに寄せるより、4-6 step でノイズを抑えて seed を選ぶ方が安定しそうです。
