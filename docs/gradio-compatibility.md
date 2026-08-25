# Gradio互換性方針

## 現在の固定構成

Aikimi Neoは、WindowsとPython 3.13を対象に、次の組み合わせを直接依存として固定しています。

- Gradio `6.17.3`
- gradio-client `2.5.0`
- FastAPI `0.141.1`
- Starlette `1.6.0`
- huggingface-hub `0.36.2`
- Transformers `4.57.6`

Gradioは`requirements.txt`から導入します。起動コードに別のGradioインストール指定はありません。PyTorchとtorchvisionはCUDA構成ごとにwheelが異なるため、従来どおり`TORCH_COMMAND`と`TORCH_INDEX_URL`を使って起動処理が管理します。

## 6.17.3を選んだ理由

旧版のGradio 4.40.0には、ファイル漏えい、CORS、SSRF、Windows上のパストラバーサルなど、Aikimi Neoの利用方法に関係する既知の脆弱性が複数あります。特に、Python 3.13以降のWindowsでは、認証を有効にしていても任意ファイルを読み取られる可能性があるため、6.7.0未満は利用できません。

2026年8月26日に、Gradio 6.17.3をPython 3.13の隔離環境へ導入し、`pip-audit`で既知の脆弱性が検出されないことを確認しました。また、6.17.3はhuggingface-hub 0.36.2と共存できるため、既存のAnima、SenseNova、MiniMax H3などが使用するTransformers 4.57.6を維持できます。

Gradio 6.18.0以降では、複数の同種コンポーネントを同時に表示したときに画面が停止する問題が修正済みです。ただし、huggingface-hub 1.2以降を要求するため、huggingface-hub 1.0未満を使うTransformers 4.57.6とは共存できません。次回のGradio更新は、モデル読込経路を含むTransformers更新とGPU実生成の回帰確認を伴う別作業として扱います。

参考資料:

- [GradioのSecurity Advisories](https://github.com/gradio-app/gradio/security/advisories)
- [WindowsとPython 3.13に関するGHSA-39mp-8hj3-5c49](https://github.com/gradio-app/gradio/security/advisories/GHSA-39mp-8hj3-5c49)
- [Gradio 6 Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)

## 6.17.3の既知制約

Gradio 6.6.0から6.17.xには、1回のイベントで複数の同種コンポーネントを`visible=False`から`visible=True`へ変更すると、ブラウザーが応答しなくなる既知の問題があります。Aikimi Neoでは、次の方針で影響を抑えます。

- `visible=False`をGradio 6の`visible="hidden"`へ正規化し、非表示コンポーネントも初めからDOMへ配置します。
- callbackが返す`gr.update(visible=False)`にも同じ正規化を適用します。
- 1回のイベントで複数の同種コンポーネントを同時に初表示しません。
- visibilityを変更するUIには、実ブラウザーを使った応答性テストを追加します。
- UIを追加するときは、Python側のBlocks構築テストだけで完了としません。

短期固定版には、この機能上の制約と回避策が残ります。Gradio 6.18.0以降へ移行できる状態になった時点で、回避策を再評価します。

## 更新時の確認項目

GradioまたはFastAPI関連の固定版を変更する場合は、少なくとも次を確認します。

1. `uv pip install --dry-run -r requirements.txt`で依存関係を解決できる。
2. `python -m pip check`が成功する。
3. `pip-audit`でGradio関連の既知脆弱性が検出されない。
4. WindowsとPython 3.13でWebUIを構築できる。
5. 画像アップロード、Gallery、動画出力、認証、`allowed_paths`、`blocked_paths`を確認できる。
6. Krea2、Anima、SenseNova、MiniMax H3のUI構築テストが成功する。
7. 実ブラウザーで主要なvisibility変更後も画面が応答する。
8. GPU依存ライブラリーを更新した場合は、対象ワークフローごとに実生成を確認する。

ローカルの基本確認には次を使用します。

```powershell
uv pip install --dry-run --python .\venv\Scripts\python.exe -r requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m unittest discover -s tools\tests -p "test_*.py" -q
```
