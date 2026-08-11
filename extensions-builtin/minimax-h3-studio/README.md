# MiniMax H3 Studio

Forge Neo からローカルの ComfyUI MiniMax H3 runtime を操作する専用 GUI です。映像と 32 kHz ステレオ音声を同時に生成し、完成した MP4 と生成条件を Forge Neo の output フォルダーへ保存します。

## 必要なもの

- ComfyUI 0.31.0 以降と、2026-08-11のH3 peak-memory修正を含むcore
- `comfy-kitchen 0.2.30` 以降
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

生成設定は、まず `動作確認`、通常は `おすすめ`、完成版だけ `最終出力` を選ぶと迷いません。3つともH3の公式20 Stepsを維持し、解像度だけで速度と品質を切り替えます。設定カードの「相対負荷」は `おすすめ / 5秒 / 20 Steps` を1.00倍とした比較値で、所要時間の予測ではありません。

## 高速・省メモリ構成

H3 Studioの専用runtimeは、生成品質とは独立した2つの起動profileを明示選択できます。

- `高速（推奨）`: DynamicVRAM + Async Offload 2 streams + Pinned Memory
- `省RAM（低速）`: node cache、Pinned Memory、Async Offloadを無効化
- OS用VRAM 2 GiBを予約し、追加headroomは確保しない
- USB SSDでは不利になる `--fast-disk` を使用しない
- H3のUNetだけに `ModelAttentionBackend = comfy kitchen attention` を適用
- preview、custom nodes、cloud API nodesを読み込まない

profileを変えただけでは接続中のprocessを書き換えません。キューが空の状態で `選択設定で再起動` を押すと、このForgeセッションが起動したbackendだけを安全に再起動します。外部ランチャーで起動したprocessは自動停止しません。生成前には選択profileと実際の引数を値・競合指定まで検査し、一致しなければ明示的に停止します。

生成の直前には、空き物理RAMに加えてWindowsのOS commit余力も確認し、少ない方が設定ごとの安全目安を下回る場合は送信前に停止します。Runtimeカードの `RAM余力` から現在の制限要因を確認できるため、不足時は他アプリを閉じる、`動作確認`へ下げる、または省RAM profileを明示選択してください。安全目安を自動的に緩めるfallbackは行いません。

ローカルAPIはHTTP接続を再利用し、短い生成の最初の60秒は2秒間隔、長時間生成は5秒間隔で状態を確認します。完成動画とJSONは一時ファイルへ書いてから公開するため、コピー失敗時に途中のMP4を履歴へ残しません。

Comfy Kitchen INT8 attentionはH3ワークフロー内だけに限定しているため、同じComfyUIにある他モデルのattentionは変更しません。Kitchenが利用できない環境では標準attentionへ黙って切り替えず、生成前に更新方法を表示して停止します。生成JSONには実際に選んだattention、起動profile、ComfyUI revision、ComfyUI/Kitchen versionを保存します。

この構成には、公式coreのH3 video VAE chunked I/OとQ/K/V peak-memory修正も含まれます。起動前にローカルGit revisionが最低commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69` を含むことを検査し、版番号だけではready扱いしません。モデルは従来どおり公式のINT8 ConvRot DiTとNVFP4-AWQ text encoderを使い、互換性未検証のthird-party INT4/GGUF、generic FP8、Torch Compileは自動適用しません。

参照素材では `<Picture 1>`、`<Video 1>`、`<Audio 1>` のように画面へ表示されたタグを Prompt へ記述します。未使用・未知・表記違いのタグは生成前にエラーとして案内します。参照動画は2〜15秒・24fpsへ揃えてください。音声だけの参照は H3 の入力条件を満たさないため、画像または動画も追加してください。

ローカル公開 weight は H3 Base です。Context-IR と 2K Regenerate は MiniMax の外部有料 API 専用で、この extension は呼び出しません。

- [MiniMax H3 official model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Official ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
