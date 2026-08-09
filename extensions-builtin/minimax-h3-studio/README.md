# MiniMax H3 Studio

Forge Neo からローカルの ComfyUI MiniMax H3 runtime を操作する専用 GUI です。映像と 32 kHz ステレオ音声を同時に生成し、完成した MP4 と生成条件を Forge Neo の output フォルダーへ保存します。

## 必要なもの

- ComfyUI 0.30.0 以降と MiniMax H3 native nodes
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`
- 参照素材モードでは `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

Forge Neo は `forge_neo_model_paths.yaml` からモデルを持つ ComfyUI runtime を自動検出します。見つからない場合は H3 Studio の `Runtime & models` で ComfyUI フォルダーを指定してください。

素材保護のため、runtime はローカルディスク上に限定されます。UNC・ネットワークドライブ・外部URLは拒否し、接続中のloopback processが選択したComfyUIフォルダーそのものか、さらにloaderが5つのH3 modelを公開しているかを生成前に確認します。

## 使い方

1. `H3 Studio` タブを開きます。
2. `テキスト`、`キーフレーム`、`参照素材`からモードを選びます。
3. 映像のショット、カメラ、台詞、効果音、音楽を同じプロンプトへ記述します。
4. Aspect、Quality、Duration を選択します。Duration は自動で H3 の `17k+5` frame grid に揃います。
5. `映像＋音声を生成`を押します。backend が停止中なら、設定済みのローカル runtime を自動起動します。
6. 完成した MP4 は `outputs/minimax_h3` に保存され、Recent generations から再表示できます。

参照素材では `<Picture 1>`、`<Video 1>`、`<Audio 1>` のように画面へ表示されたタグを Prompt へ記述します。未使用・未知・表記違いのタグは生成前にエラーとして案内します。参照動画は2〜15秒・24fpsへ揃えてください。音声だけの参照は H3 の入力条件を満たさないため、画像または動画も追加してください。

ローカル公開 weight は H3 Base です。Context-IR と 2K Regenerate は MiniMax の外部有料 API 専用で、この extension は呼び出しません。

- [MiniMax H3 official model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Official ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
