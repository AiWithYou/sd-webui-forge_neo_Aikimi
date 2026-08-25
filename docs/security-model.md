# Aikimi Neo security model

## 既定の境界

Aikimi Neoは、信頼済みのWindows利用者が同じ端末のブラウザーから使う構成を既定とします。通常起動ではWebUIとAPIを`127.0.0.1`へbindし、LAN、Gradio share、ngrokへ公開しません。

推奨入口は次です。

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
```

`aikimi-launch.bat`を使う場合は、profile名を第1引数へ渡します。

```text
aikimi-launch.bat LocalSafe
```

## 起動profile

| profile | bindと用途 | 主な引数 |
|---|---|---|
| `LocalSafe` | loopback WebUIとAPI | `--server-name 127.0.0.1 --api` |
| `LocalAPI` | loopback APIのみ | `--nowebui --server-name 127.0.0.1` |
| `LANAuthenticated` | 認証付きLAN公開 | `--aikimi-remote --listen --api`と2つのauth file |
| `Development` | loopback UI debug | `--ui-debug-mode` |
| `LowVRAM` | loopback、低VRAM | `--lowvram --tiled-conv2d 64` |
| `RTX3090Recommended` | loopback、RTX 3090向け | BnB、tiled Conv2d、cudaMallocAsync |

`--listen`、loopback以外の`--server-name`、`--share`、`--ngrok`は、`--aikimi-remote`がなければ起動前に失敗します。WebUIを公開する場合はGradio認証、APIを公開する場合はAPI認証が必要です。両方を有効にする場合は、両方の認証を設定します。

## 認証ファイル

`LANAuthenticated`は次を読みます。

```text
secrets/gradio-auth.txt
secrets/api-auth.txt
```

各行は`username:password`形式です。複数利用者は1行ずつ記載できます。空のusername、空のpassword、64 KiBを超えるファイルは拒否します。`secrets/`はGit管理外です。

```powershell
New-Item -ItemType Directory -Force .\secrets | Out-Null
notepad .\secrets\gradio-auth.txt
notepad .\secrets\api-auth.txt
.\aikimi-launch.ps1 -Profile LANAuthenticated
```

認証値を`COMMANDLINE_ARGS`へ直接書かないでください。Basic認証情報を暗号化されていないHTTPでLAN外へ送らないでください。外部公開では、別のreverse proxyでTLSを終端し、firewallでも接続元を制限してください。

## container

containerも`127.0.0.1`が既定です。外部bindには`AIKIMI_CONTAINER_REMOTE=1`が必要です。さらに、`COMMANDLINE_ARGS`またはcontainer引数でGradioとAPIのauth fileを指定しなければ起動しません。

```text
AIKIMI_CONTAINER_REMOTE=1
--gradio-auth-path /run/secrets/gradio-auth.txt
--api-auth-path /run/secrets/api-auth.txt
```

secretはimageへ含めず、read-only volumeやcontainer secretとしてmountしてください。

## 信頼境界

```text
Browser
  -> Gradio / FastAPI authentication
  -> Forge request validation and options policy
  -> GPU model runtime
     -> SenseNova isolated worker
     -> local MiniMax H3 ComfyUI bridge
  -> managed outputs / temporary files
```

- Browser入力: API schema、OptionInfo policy、path policyによる検証
- API経由の設定変更: `restrict_api`と型の検査
- URL画像入力: safe fetch policyによるscheme、DNS、redirect、size、Content-Typeの確認
- Gradio公開path: 管理されたoutput、temporary、static assetへの限定
- SenseNova: Forge UI、bridge、専用workerの責務分離
- MiniMax H3: loopback ComfyUIと選択runtime identityの検査
- model installer: 固定revision、size、SHA-256の確認後に正式名へ移動

管理されたoutput、temporary、Canvas assetがsymlinkやjunctionを経由して管理root外へ解決される場合は、起動を停止します。外部保存先を使う場合でも、WebUIの公開pathへ任意directoryを追加せず、生成後の別工程で移動してください。

## API

Local Safeでは次のread-only endpointをloopbackから確認できます。

```text
GET /aikimi/api/v1/health
GET /aikimi/api/v1/status
GET /aikimi/api/v1/capabilities
```

remote modeでは、これらを含むAPI routeが認証境界を通ります。healthは秘密値を返さず、statusとcapabilitiesも認証情報や機密性が高い絶対パスを返しません。

## ログとsysinfo

共通redactionは、password、token、secret、auth、Cookie、Authorization、URL userinfo、credentialに見えるquery parameterをマスクします。例外をclientへ返す場合は、長さを制限した安全な要約を使います。

redactionは最後の防御です。利用者は、共有前に[SECURITY.md](../SECURITY.md)の確認項目を目視してください。

## 依存関係監査の期限付き例外

2026年8月26日の監査では、Gradio、FastAPI、Starlette、GitPython、protobuf、pipに未対応の既知脆弱性は残っていません。一方、現在のGPU生成経路を維持するため、次のadvisory IDを2026年9月30日までの期限付き例外としてCIへ登録しています。

- Diffusers 0.37.1: `PYSEC-2026-2446`、`PYSEC-2026-40`、`PYSEC-2026-41`
- diskcache 5.6.3: `PYSEC-2026-2447`
- setuptools 81.0.0: `PYSEC-2026-3447`
- Transformers 4.57.6: `PYSEC-2025-217`、`PYSEC-2026-2290`、`PYSEC-2026-2288`、`PYSEC-2026-2289`

Diffusers 0.38とTransformers 5系への更新は、Anima、SenseNova、Krea2、MiniMax H3を含むGPU実生成の回帰確認が必要です。diskcacheには監査時点で修正版がなく、setuptools 83はPyTorch 2.11の`setuptools<82`制約と両立しません。CIは上記IDだけを除外するため、新しいadvisory IDは即時に失敗します。期限までにモデルスタックを再検証し、解除できない場合は理由と次回期限を改めて審査します。

## 非目標

- zero-trust gatewayは、このリポジトリの対象外です。
- ローカル管理者や同じWindows accountを制御した攻撃者は、想定するsecurity boundaryの外側です。
- third-party extension、model、custom runtimeは、導入者が個別に安全性を確認します。
- model licenseと生成物の権利は、利用者が配布元の原文から判断してください。
