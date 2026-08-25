# Architecture

## 全体像

Aikimi Neoは、Forge Neo本体へすべてのruntimeを埋め込まず、機能ごとに責務を分けます。

```text
Windows launcher / Docker entrypoint
  -> startup security policy
  -> Forge WebUI and FastAPI
     -> standard txt2img / img2img / Extras
     -> Aikimi status and Diagnostics
     -> built-in workflow UI
        -> Anima 3.8B in-process extension
        -> SenseNova bridge -> isolated worker
        -> MiniMax H3 bridge -> local ComfyUI process
  -> managed output and temporary directories
```

## 起動とsecurity policy

`launch.py`と`modules/initialize_util.py`は、modelを読み込む前に起動引数を検証します。`modules/aikimi_security`は、次の共通処理を持ちます。

- loopback既定とremote opt-in
- GradioとAPIの認証file
- APIへ返す設定の公開policy
- log、sysinfo、URL、例外のredaction
- 外部URL画像入力のfetch policy
- Gradioへ公開するpathの限定

`aikimi-launch.ps1`は、用途別profileを明示的な引数へ変換します。`webui-user.bat`は既存更新との互換を保つ入口です。新しい設定はGit管理外の`webui-user.local.bat`へ置きます。

## WebUIとAPI

`webui.py`がGradioとFastAPIを構築し、`modules/api/api.py`が標準APIとAikimi独自routeを登録します。APIの設定変更は`modules/options.py`の型と`restrict_api`を通ります。

`modules/aikimi_diagnostics.py`はDiagnostics UIを構築し、次のread-only APIからJSONを返します。

```text
/aikimi/api/v1/health
/aikimi/api/v1/status
/aikimi/api/v1/capabilities
```

API routeは共通の認証処理を使います。healthは最小情報、statusとcapabilitiesはredaction済みの状態だけを返します。

## Aikimi Status

`modules/aikimi_status.py`が生成状態、queue、VRAM、model読込状態を要約し、`javascript/aikimiStatus.js`がキャラクターとして表示します。技術logは残したまま、キャラクター向けmessageと詳細情報を分離しています。

assetは`assets/aikimi`から読みます。assetの読込失敗はUI全体の起動を止めません。animationを無効にした場合やreduced motion環境では、still assetを使います。

## Anima 3.8B

`extensions-builtin/anima-3-8b`は、通常のForge `txt2img`と`img2img`へQwen3.5 adapterを追加します。52層checkpointの検査、encoder、adapter、UI callbackはextension内で分けています。

`modules_forge/anima_lora.py`は、28層、40層、52層のLoRA layoutを検査し、安全に特定できる完全なcoverageだけを変換します。sparseまたは曖昧なLoRAは自動変換しません。

INT8 ConvRot変換は`tools/convert_anima38_int8_convrot.py`から共通streaming converterを使います。出力は一時ファイルへ書き、tensor layoutとmetadataを検証してから正式名へ移します。

## SenseNova U1.5

SenseNovaは通常のForge samplerへ変換しません。境界は次のとおりです。

```text
extensions-builtin/sensenova-u15-studio
  -> modules_forge/sensenova_u15_bridge.py
     -> tools/sensenova_u15_worker.py
        -> pinned SenseNova runtime and checkpoint
```

Studioは入力とUI、bridgeはjob、VRAM解放、worker process、保存、cancel、workerは固定runtimeでの推論を担当します。workerを別processにすることで、終了時にweightとactivationをOSへ返します。

## MiniMax H3

MiniMax H3 Studioは、Aikimi NeoへComfyUI runtime全体を組み込みません。

```text
extensions-builtin/minimax-h3-studio
  -> modules_forge/minimax_h3_bridge.py
     -> loopback ComfyUI runtime
```

bridgeは、runtime path、process identity、ComfyUI revision、必要node、model一覧を検査します。Forgeが起動したprocessだけを再起動し、外部launcherが所有するprocessを自動停止しません。

## model setup

`tools/aikimi_setup.py`はstdlibだけで動作し、固定manifest、download、resume、size、SHA-256、path containment、disk preflight、verify、repairを共通化します。Anima conversionだけは、準備済みForge venvとCUDAを使って既存converterを実行します。

既存のKrea2、Anima、SenseNova用batとPowerShell scriptは互換入口として残します。詳しい契約は[model-installation.md](model-installation.md)を参照してください。

## 保存先

- 通常画像はForgeの`output`または`outputs`
- SenseNovaは`outputs/sensenova_u15`
- MiniMax H3は`outputs/minimax_h3`
- setup partialは正式ファイルと同じdirectoryの`.part`
- repairで隔離したファイルは`tmp/aikimi-setup/quarantine`

管理対象外の任意pathをcleanup対象へ加えません。model、output、cache、log、secretはGit管理外です。
