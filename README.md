# Aikimi Neo

Aikimi Neoは、[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic)を基盤に、画像、動画、高解像度処理、model setup、診断をまとめたWindows向けAI生成workspaceです。通常起動は`127.0.0.1`だけを使い、LANやインターネットへ自動公開しません。

| 項目 | 内容 |
|---|---|
| default branch | `neo` |
| upstream | `Haoming02/sd-webui-forge-classic`の`neo` |
| 最終同期基準 | `f27709a4bf756078b89c947bd256d1c19ada71ef` |
| 主対象 | Windows 11、Python 3.13、NVIDIA GPU |
| code license | AGPL-3.0。modelとassetには別条件が適用される場合があります。 |

## Quick Start

先に次を用意してください。

- Git
- Python 3.13
- PowerShell 7の`pwsh`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- 対象PyTorch buildに対応したNVIDIA driver

```powershell
git clone --branch neo https://github.com/AiWithYou/sd-webui-forge_neo_Aikimi.git
cd sd-webui-forge_neo_Aikimi
.\aikimi-launch.ps1 -Profile LocalSafe
```

`aikimi-launch.bat`をダブルクリックした場合も、`LocalSafe`で起動します。起動後に<http://127.0.0.1:7861>を開いてください。

Local Safeは次を有効にします。

- WebUIとAPIを`127.0.0.1`へbind
- Gradio share、ngrok、LAN公開を無効化
- dark theme、BnB、tiled Conv2d、cudaMallocAsync
- `forge_neo_model_paths.yaml`がある場合だけ共有model pathを追加

初回起動は依存関係を導入します。現在のlauncher既定はPyTorch`2.11.0+cu130`とtorchvision`0.26.0+cu130`です。通常は対応NVIDIA driverと、launcherが導入するPyTorch wheelを使います。追加CUDA Toolkitの要否はcustom extensionごとに確認してください。

## 起動profile

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
.\aikimi-launch.ps1 -Profile LocalAPI
.\aikimi-launch.ps1 -Profile Development
.\aikimi-launch.ps1 -Profile LowVRAM
.\aikimi-launch.ps1 -Profile RTX3090Recommended
```

| profile | 用途 | 公開範囲 |
|---|---|---|
| `LocalSafe` | 通常のWebUIとAPI | loopbackのみ |
| `LocalAPI` | APIだけを起動 | loopbackのみ |
| `Development` | UI debug | loopbackのみ |
| `LowVRAM` | 低VRAM向け | loopbackのみ |
| `RTX3090Recommended` | RTX 3090向け既定 | loopbackのみ |
| `LANAuthenticated` | 認証付きLAN利用 | 明示的なremote opt-in |

### 認証付きLAN利用

`--listen`、loopback以外の`--server-name`、`--share`、`--ngrok`は、`--aikimi-remote`と認証がなければ起動前に失敗します。

`LANAuthenticated`を使う場合は、Git管理外の次の2ファイルを作成してください。

```text
secrets/gradio-auth.txt
secrets/api-auth.txt
```

各行は`username:password`形式です。

```powershell
.\aikimi-launch.ps1 -Profile LANAuthenticated
```

Basic認証だけでインターネットへ直接公開しないでください。TLS reverse proxyとfirewallを併用します。詳しくは[security model](docs/security-model.md)と[SECURITY.md](SECURITY.md)を参照してください。

## `webui-user.bat`のローカル設定

既存更新との互換を保つため、今回のreleaseでは`webui-user.bat`を追跡済みのthin wrapperとして残します。個人設定はGit管理外の`webui-user.local.bat`へ置き、認証値は書かないでください。秘密なしの例からlocal fileを作成できます。

```powershell
Copy-Item .\webui-user.example.bat .\webui-user.local.bat
```

次のmajor releaseでは、移行状況を確認したうえで`webui-user.bat`の追跡解除を検討します。

## Model Setup

model、VAE、text encoder、LoRAはリポジトリに含みません。統一CLIは固定revision、size、SHA-256を検査し、`.part`から再開します。

```powershell
python .\tools\aikimi_setup.py list
python .\tools\aikimi_setup.py install krea2
python .\tools\aikimi_setup.py install anima38
python .\tools\aikimi_setup.py install sensenova
python .\tools\aikimi_setup.py verify
python .\tools\aikimi_setup.py repair anima38 --dry-run
```

`--dry-run`はfilesystemとnetworkを変更しません。`--json`を付けると、相対pathと結果をJSONで返します。現在の固定artifactはpublicで、tokenを必要としません。tokenをcommandやURL queryへ書かないでください。

| profile | 最終配置または一時peak | 注意 |
|---|---:|---|
| Krea2 | 約17.68 GiB | checkpoint、Qwen3-VL、VAE |
| Anima 3.8B | 既存batの一時peak約15.47 GiB | 共通encoderとVAEも新規導入するCLIは約16.82 GiB |
| SenseNova U1.5 | 最終約17.28 GiB | 既存並列PowerShell installerの一時peakは約33.79 GiB |

filesystemと更新用の余裕を追加してください。Anima変換にはCUDA device 0と準備済みForge venvが必要です。

既存のbatも互換入口として残します。

```text
download_krea2_int8_convrot_models.bat
download_anima38_int8_convrot_models.bat
download_sensenova_u15_int8.bat
```

詳しい固定値、repair、licenseは[Model installation](docs/model-installation.md)を参照してください。

## 主な機能

| 機能 | backend | 状態と検証範囲 |
|---|---|---|
| Forge画像生成 | Forge Neo | upstream機能。checkpointごとの条件に従います。 |
| Krea2 INT8 | Forge low-bit loader | 1024²生成と4K／8K workflow。高解像度機能の一部は実験機能です。 |
| Anima 3.8B | built-in extension | 52層DiT、Qwen3.5、28／40／52層LoRA。RTX 3090で約1MPを実測しています。 |
| SenseNova U1.5 | 隔離worker | text生成と複数画像編集。24GB Safeは参照2枚、各約512²、出力2048²以下です。 |
| MiniMax H3 | loopback ComfyUI | 音声付き動画。対応runtimeとmodelを別途用意します。外部有料APIは呼びません。 |
| HyperWeave | built-in extension | 候補制約付き4K／8K生成的upscale。実験機能です。 |
| Aikimi Status | WebUI overlay | queue、model load、VRAM、errorを表示し、技術logも残します。 |
| Diagnostics | SettingsとAPI | Python、PyTorch、CUDA、GPU、disk、model、公開状態をReady／Warning／Blockedで表示します。 |

推測した最低VRAMや処理時間は掲載していません。実測条件は各guideに記載します。

- [Anima 3.8B guide](extensions-builtin/anima-3-8b/README.md)
- [SenseNova U1.5 Studio guide](extensions-builtin/sensenova-u15-studio/README.md)
- [MiniMax H3 Studio guide](extensions-builtin/minimax-h3-studio/README.md)
- [HyperWeave guide](extensions-builtin/hyperweave/README.md)
- [Krea2 high-resolution notes](docs/krea2_local_supersample_detail_ja.md)

## DiagnosticsとAPI

`Settings`のDiagnosticsはlocal checkだけを実行し、model downloadや生成を始めません。read-only APIは次です。

```text
GET /aikimi/api/v1/health
GET /aikimi/api/v1/status
GET /aikimi/api/v1/capabilities
```

Local Safeでの確認例:

```powershell
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/health
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/capabilities
```

remote modeでは認証が必要です。APIはtoken、password、認証file path、機密性が高い絶対pathを返しません。

## Update

通常利用者はAikimi Neoの`neo`をfast-forwardで更新します。

```powershell
git switch neo
git pull --ff-only
```

model、output、secret、`webui-user.local.bat`はGit管理外です。更新前に`git status`を確認し、追跡fileへ書いた個人設定を退避してください。

Forge Neo upstreamとの同期はmaintainer作業です。

```powershell
git remote add upstream https://github.com/Haoming02/sd-webui-forge-classic.git
git fetch upstream neo
```

同期では、Aikimiのsecurity guard、BnB／NF4／GGUF互換、Anima、SenseNova、MiniMax H3、高解像度workflowを個別に検証します。詳しくは[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。

## Test

通常のCI相当testは、CPU、offline、外部model downloadなしで動きます。

```powershell
.\venv\Scripts\python.exe .\tools\run_ci_tests.py --verbosity 1
```

setup CLIだけを短く確認する場合:

```powershell
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_aikimi_setup
.\venv\Scripts\python.exe -m ruff check tools\aikimi_setup.py tools\tests\test_aikimi_setup.py
.\venv\Scripts\python.exe -m ruff format --check tools\aikimi_setup.py tools\tests\test_aikimi_setup.py
```

CIは用途別に、lint、unit tests、Windows smoke、installer、secret scan、dependency audit、CodeQL、dependency reviewを実行します。GPU live testは通常testと分け、未実施の機能を成功扱いしません。release前の全gateは[Release checklist](docs/release-checklist.md)にあります。

## Troubleshooting

起動、remote auth、CUDA、model setup、SenseNova、MiniMax H3の確認手順は[Troubleshooting](docs/troubleshooting.md)にあります。

logやsysinfoを共有する前に、認証情報、URL query、絶対path、prompt、入力basenameを目視してください。自動redactionだけで安全を保証できません。

## Documentation

- [Security policy](SECURITY.md)
- [Security model](docs/security-model.md)
- [Architecture](docs/architecture.md)
- [Model installation](docs/model-installation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md)
- [Release checklist](docs/release-checklist.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Licenseと配布条件

codeは[AGPL-3.0](LICENSE)です。nested directoryに別licenseがある場合は、そのcopyright noticeと条件を保持します。

model、VAE、text encoder、LoRA、font、画像asset、生成物には別条件が適用される場合があります。特にKrea2、Anima、SenseNova、MiniMax H3の条件は利用時点の配布元で確認してください。

`assets/aikimi`の由来、権利保有者、再配布条件は、監査時点のリポジトリだけでは確認できません。権利を推測して断定せず、第三者向けrelease前にasset noticeを整備します。確認済みの出典と未解決項目は[Third-party notices](THIRD_PARTY_NOTICES.md)に記録しています。
