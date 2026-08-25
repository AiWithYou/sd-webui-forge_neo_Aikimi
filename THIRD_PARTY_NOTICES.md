# Third-party notices

この文書は、Aikimi Neoが利用または案内する主なthird-party成果物への索引です。各成果物のlicense本文と配布元の条件が優先されます。この文書は、モデルやassetの利用許諾を新たに与えるものではありません。

## コード基盤

| 成果物 | 用途 | licenseまたはnotice |
|---|---|---|
| AUTOMATIC1111 Stable Diffusion WebUI | WebUI基盤 | ルート[LICENSE](LICENSE)とupstream notice |
| Stable Diffusion WebUI Forge | Forge backend | ルート[LICENSE](LICENSE)とupstream notice |
| Stable Diffusion WebUI Forge - Neo | 現在のupstream | ルート[LICENSE](LICENSE)とupstream notice |
| ComfyUI由来package | workflowとmodel処理 | [modules_forge/packages/comfy/LICENSE](modules_forge/packages/comfy/LICENSE) |
| GGUF package | GGUF読込 | [modules_forge/packages/gguf/LICENSE](modules_forge/packages/gguf/LICENSE) |
| built-in ControlNet／IP-Adapter | built-in extension | 各extension directoryの`LICENSE` |

ルートのコードはAGPL-3.0です。nested directoryに別のlicenseがある場合は、そのcopyright noticeと条件を保持します。

## Anima 3.8B extensionとtokenizer

`extensions-builtin/anima-3-8b`は、`GumGum10/forge-anima-3.8B`のcommit`8af9bb4d391787030cb84205c47cf3ea1213795a`を基にしています。extension codeのMIT Licenseは、[extensions-builtin/anima-3-8b/LICENSE](extensions-builtin/anima-3-8b/LICENSE)にあります。

同梱するQwen3.5 tokenizerの2ファイルは、`Qwen/Qwen3.5-4B`のrevision`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`由来です。出典は[extensions-builtin/anima-3-8b/THIRD_PARTY_NOTICES.md](extensions-builtin/anima-3-8b/THIRD_PARTY_NOTICES.md)にあります。配布元はApache-2.0と表示しています。release担当者は、配布物にApache-2.0本文が含まれることも確認してください。

## model installerが取得する成果物

モデルweightはGitリポジトリに含めません。`tools/aikimi_setup.py list`は、固定revisionとlicense URLを表示します。

| profile | 主な配布元 | 条件の確認先 |
|---|---|---|
| Krea2 | Comfy-Org/Krea-2、Qwen | 固定revisionのKrea 2 Community License PDFと各Qwen license |
| Anima 3.8B | lylogummy/Anima-3.8B、circlestone-labs/Anima、Qwen | 各model card、CircleStone Labs license、Qwen license |
| SenseNova U1.5 | SenseNova、starsFriday、joyfox | 各固定revisionのmodel cardとruntime LICENSE |
| MiniMax H3 | MiniMaxAI、ComfyUI | MiniMax H3 Community LicenseとComfyUI側のnotice |

Animaの配布repositoryは、upstream AnimaとNVIDIA由来条件の確認を求めています。条件を短く言い換えて断定せず、利用時点の原文を確認してください。Krea2にも独自のcommunity licenseがあります。

SenseNova runtimeは、固定commit`e6dfd45762eb46f805067fe079c14bcb643ccccd`から取得します。runtime directoryへApache-2.0の`LICENSE`も配置します。

## fontとUI asset

`modules/Roboto-Regular.ttf`は、font metadataでApache-2.0を示しています。`modules/web/fonts/sourcesanspro`にはSource Sans Proのwoff2が含まれます。release担当者は、fontの配布元と必要なlicense本文をrelease前に再確認してください。

`assets/aikimi`の画像はAiWithYouのGit履歴から追加されていますが、原画の由来、権利保有者、再配布条件を説明するnoticeは監査時点でありません。この文書は、画像の権利やlicenseを推測しません。第三者向けreleaseでは、権利者が確認したasset noticeを追加するまで、この項目を未解決として扱ってください。

docsのPDF、`html/ui.webp`、`html/card-no-preview.jpg`も配布assetです。新しいreleaseへ含める場合は、対応するsourceと利用条件をrelease checklistで確認してください。

## 外部サービス

MiniMax H3 Studioは、既定でローカルComfyUIだけを使います。H3のContext-IRと2K Regenerateは外部有料API向けですが、Aikimi NeoのStudioは呼び出しません。将来外部APIを追加する場合は、送信データ、費用、利用規約、秘密情報の保存方法を別途明記してください。
