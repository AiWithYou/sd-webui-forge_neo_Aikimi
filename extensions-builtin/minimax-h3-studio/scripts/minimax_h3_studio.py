from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules.paths import data_path, script_path
from modules_forge.minimax_h3_bridge import (
    H3BridgeError,
    H3Request,
    H3_SERVER_URL,
    MODE_KEYFRAMES,
    MODE_REFERENCES,
    MODE_TEXT,
    RUNTIME_PROFILE_FAST,
    RUNTIME_PROFILE_LOW_RAM,
    append_prompt_section,
    cache_history_video,
    cancel_generation,
    discover_runtime_root,
    generation_preset_values,
    history_choices,
    history_html,
    inspect_readiness,
    list_history,
    normalize_file_list,
    progress_html,
    prompt_template,
    readiness_html,
    reference_guide_html,
    restart_runtime,
    resolve_runtime_root,
    run_generation,
    settings_summary_html,
    ensure_ready,
    validate_request,
)


CONFIG_PATH = Path(script_path) / "forge_neo_model_paths.yaml"
OUTPUT_DIRECTORY = Path(data_path) / "outputs" / "minimax_h3"
LOG_DIRECTORY = Path(data_path) / "logs" / "minimax_h3"

MODE_HELP = {
    MODE_TEXT: (
        "TEXT · FL2VA",
        "言葉だけから映像と32kHzステレオ音声を同時生成します。最初の一作におすすめです。",
    ),
    MODE_KEYFRAMES: (
        "KEYFRAMES · FL2VA",
        "開始画像、終了画像、または両方を指定して、その間の動きと音を生成します。",
    ),
    MODE_REFERENCES: (
        "REFERENCES · REF2VA",
        "人物・画風・動き・声を画像／動画／音声から参照します。タグ順序を確認して使います。",
    ),
}

QUALITY_LABELS = {
    "draft": "動作確認 · 約0.2 MP · 速い",
    "preview": "おすすめ · 約0.4 MP",
    "balanced": "バランス · 約0.5 MP",
    "native": "最終出力 · Native 768p · 重い",
}

PRESET_STATE_LABELS = {
    "quick": "動作確認",
    "recommended": "おすすめ",
    "final": "最終出力",
    "custom": "カスタム",
}


def _initial_runtime() -> Path | None:
    try:
        return discover_runtime_root(CONFIG_PATH)
    except H3BridgeError:
        return None


def _mode_help_html(mode: str) -> str:
    eyebrow, message = MODE_HELP.get(mode, MODE_HELP[MODE_TEXT])
    return (
        '<div class="h3-mode-help">'
        f'<span>{html.escape(eyebrow)}</span><p>{html.escape(message)}</p>'
        "</div>"
    )


def _mode_updates(
    mode: str,
    aspect: str = "16:9",
    quality: str = "preview",
    duration: float = 5.0,
    steps: int = 20,
    scheduler: str = "simple",
    ref_image_size: str = "match",
    current_validation: str = "",
):
    reference_mode = mode == MODE_REFERENCES
    effective_ref_image_size = str(ref_image_size) if reference_mode else "match"
    return (
        gr.update(visible=mode == MODE_KEYFRAMES),
        gr.update(visible=reference_mode),
        _mode_help_html(mode),
        gr.update(visible=True) if reference_mode else gr.update(visible=False, value=effective_ref_image_size),
        settings_summary_html(
            aspect,
            quality,
            duration,
            steps,
            scheduler,
            effective_ref_image_size,
        ),
        _clear_validation_targets(current_validation, "keyframes", "references"),
    )


def _status_error(message: str) -> str:
    return (
        '<div class="h3-runtime-card" data-tone="error" role="alert">'
        '<div class="h3-runtime-badges"><span data-tone="error"><i></i>確認が必要</span></div>'
        f'<p>{html.escape(message)}</p></div>'
    )


def _preset_state_html(preset: str) -> str:
    label = PRESET_STATE_LABELS.get(preset, PRESET_STATE_LABELS["custom"])
    safe_preset = preset if preset in PRESET_STATE_LABELS else "custom"
    return (
        f'<p class="h3-preset-state" data-h3-preset="{safe_preset}" '
        'role="status" aria-live="polite" aria-atomic="true">'
        f'現在の設定: <strong>{html.escape(label)}</strong></p>'
    )


def _apply_generation_preset(preset: str, aspect: str):
    return (*generation_preset_values(preset, aspect), _preset_state_html(preset))


def _apply_quick_preset(aspect: str):
    return _apply_generation_preset("quick", aspect)


def _apply_recommended_preset(aspect: str):
    return _apply_generation_preset("recommended", aspect)


def _apply_final_preset(aspect: str):
    return _apply_generation_preset("final", aspect)


def _mark_preset_custom(_value: object | None = None) -> str:
    return _preset_state_html("custom")


def _custom_settings_updates(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    scheduler: str,
    ref_image_size: str,
    current_validation: str = "",
) -> tuple[str, str, str | dict]:
    return (
        settings_summary_html(aspect, quality, duration, steps, scheduler, ref_image_size),
        _mark_preset_custom(),
        _clear_validation_targets(current_validation, "settings"),
    )


def _aspect_settings_updates(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    scheduler: str,
    ref_image_size: str,
    current_validation: str = "",
):
    return (
        settings_summary_html(aspect, quality, duration, steps, scheduler, ref_image_size),
        _clear_validation_targets(current_validation, "settings"),
    )


def _input_validation_html(message: str, target: str) -> str:
    safe_target = target if target in {"prompt", "keyframes", "references", "settings"} else "settings"
    return (
        f'<div class="h3-input-error" data-h3-invalid="{safe_target}" role="alert">'
        '<strong>入力を確認してください</strong>'
        f'<span id="h3-input-validation-message">{html.escape(message)}</span>'
        "</div>"
    )


def _validation_target(request: H3Request) -> str:
    if not request.prompt or not request.prompt.strip() or len(request.prompt) > 20_000:
        return "prompt"
    try:
        request.dimensions
        request.frame_count
    except (H3BridgeError, TypeError, ValueError):
        return "settings"
    if not 1 <= int(request.steps) <= 100:
        return "settings"
    if int(request.seed) < -1 or int(request.seed) >= 2**63:
        return "settings"
    if request.scheduler not in {"simple", "beta", "normal"}:
        return "settings"
    if request.ref_image_size not in {"match", "max"}:
        return "settings"
    if request.mode == MODE_KEYFRAMES:
        return "keyframes"
    if request.mode == MODE_REFERENCES:
        return "references"
    return "settings"


def _clear_validation_targets(current_validation: str, *targets: str):
    rendered = str(current_validation or "")
    if any(f'data-h3-invalid="{target}"' in rendered for target in targets):
        return ""
    return gr.update()


def _clear_prompt_validation(current_validation: str):
    return _clear_validation_targets(current_validation, "prompt")


def _clear_keyframe_validation(current_validation: str):
    return _clear_validation_targets(current_validation, "keyframes")


def _clear_settings_validation(current_validation: str):
    return _clear_validation_targets(current_validation, "settings")


def _reference_guide_updates(
    image_values,
    video_values,
    audio_values,
    current_validation: str,
):
    return (
        reference_guide_html(image_values, video_values, audio_values),
        _clear_validation_targets(current_validation, "references"),
    )


def _prompt_action_updates(prompt: str, action: str, current_validation: str):
    updated_prompt = prompt_template(prompt) if action == "template" else append_prompt_section(prompt, action)
    return updated_prompt, _clear_prompt_validation(current_validation)


def _prompt_template_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "template", current_validation)


def _prompt_camera_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "camera", current_validation)


def _prompt_dialogue_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "dialogue", current_validation)


def _prompt_sfx_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "sfx", current_validation)


def _prompt_music_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "music", current_validation)


def _connect_runtime(runtime_value: str, server_url: str, runtime_profile: str) -> str:
    try:
        root = resolve_runtime_root(runtime_value)
        readiness = ensure_ready(
            root,
            server_url,
            LOG_DIRECTORY,
            runtime_profile=runtime_profile,
        )
        return readiness_html(readiness, runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _restart_runtime(runtime_value: str, server_url: str, runtime_profile: str) -> str:
    try:
        root = resolve_runtime_root(runtime_value)
        readiness = restart_runtime(
            root,
            server_url,
            LOG_DIRECTORY,
            runtime_profile=runtime_profile,
        )
        readiness = ensure_ready(
            root,
            server_url,
            LOG_DIRECTORY,
            runtime_profile=runtime_profile,
        )
        return readiness_html(readiness, runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _rescan_runtime(runtime_value: str, server_url: str, runtime_profile: str) -> str:
    try:
        root = resolve_runtime_root(runtime_value)
        return readiness_html(inspect_readiness(root, server_url), runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _history_state(runtime_value: str):
    try:
        root = resolve_runtime_root(runtime_value) if runtime_value else None
    except H3BridgeError:
        root = None
    try:
        items = list_history(root, OUTPUT_DIRECTORY)
    except OSError as exc:
        raise H3BridgeError(f"生成履歴を読み込めません: {exc}") from exc
    return items, history_html(items), history_choices(items)


def _refresh_history(runtime_value: str):
    try:
        _, rendered, choices = _history_state(runtime_value)
        selected = choices[0][1] if choices else None
        return rendered, gr.update(choices=choices, value=selected)
    except H3BridgeError as exc:
        return _status_error(str(exc)), gr.update()


def _load_history_video(selected: str, runtime_value: str):
    if not selected:
        return gr.update(), progress_html("idle", "表示する履歴を選択してください", 0.0)
    try:
        items, _, _ = _history_state(runtime_value)
        path = cache_history_video(selected, items, OUTPUT_DIRECTORY)
        return path, progress_html("complete", "履歴の動画を読み込みました", 1.0)
    except H3BridgeError as exc:
        return gr.update(), progress_html("error", str(exc), 0.0)


def _request_from_ui(
    mode,
    prompt,
    first_frame,
    last_frame,
    reference_images,
    reference_videos,
    reference_audios,
    aspect,
    quality,
    duration,
    steps,
    seed,
    scheduler,
    ref_image_size,
) -> H3Request:
    return H3Request(
        mode=str(mode),
        prompt=str(prompt or ""),
        first_frame=os.fspath(first_frame) if first_frame else None,
        last_frame=os.fspath(last_frame) if last_frame else None,
        reference_images=normalize_file_list(reference_images),
        reference_videos=normalize_file_list(reference_videos),
        reference_audios=normalize_file_list(reference_audios),
        aspect=str(aspect),
        quality=str(quality),
        duration_seconds=float(duration),
        steps=int(steps),
        seed=int(seed),
        scheduler=str(scheduler),
        ref_image_size=str(ref_image_size),
    )


def _generate(
    runtime_value,
    server_url,
    runtime_profile,
    mode,
    prompt,
    first_frame,
    last_frame,
    reference_images,
    reference_videos,
    reference_audios,
    aspect,
    quality,
    duration,
    steps,
    seed,
    scheduler,
    ref_image_size,
):
    yield (
        progress_html("prepare", "生成条件を確認しています", 0.02),
        gr.update(),
        "",
        gr.update(),
        gr.update(),
        gr.update(interactive=False),
        gr.update(value="生成を準備中…", interactive=False),
        "",
    )
    try:
        request: H3Request | None = None
        try:
            request = _request_from_ui(
                mode,
                prompt,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
            )
            validate_request(request)
        except (H3BridgeError, ValueError, TypeError) as exc:
            target = _validation_target(request) if request is not None else "settings"
            yield (
                progress_html("idle", "入力欄を確認して、もう一度生成してください", 0.0),
                gr.update(),
                "",
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
                gr.update(value="映像＋音声を生成", interactive=True),
                _input_validation_html(str(exc), target),
            )
            return
        root = resolve_runtime_root(runtime_value)
        for update in run_generation(
            request,
            root,
            server_url,
            LOG_DIRECTORY,
            OUTPUT_DIRECTORY,
            runtime_profile=runtime_profile,
        ):
            video_update = update.get("path") or gr.update()
            prompt_id = update.get("prompt_id") or ""
            rendered_history = gr.update()
            selector_update = gr.update()
            if update["stage"] == "complete":
                _, rendered_history, choices = _history_state(runtime_value)
                selected = choices[0][1] if choices else None
                selector_update = gr.update(choices=choices, value=selected)
            cancel_update = gr.update(
                interactive=bool(prompt_id) and update["stage"] not in {"complete", "error"}
            )
            generation_finished = update["stage"] in {"complete", "error"}
            generate_update = gr.update(
                value="映像＋音声を生成" if generation_finished else "生成中…",
                interactive=generation_finished,
            )
            yield (
                progress_html(
                    update["stage"],
                    update["message"],
                    update.get("progress", 0.0),
                    update.get("elapsed"),
                ),
                video_update,
                prompt_id,
                rendered_history,
                selector_update,
                cancel_update,
                generate_update,
                gr.update(),
            )
    except (H3BridgeError, OSError, ValueError, TypeError) as exc:
        yield (
            progress_html("error", str(exc), 0.0),
            gr.update(),
            "",
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(value="映像＋音声を生成", interactive=True),
            gr.update(),
        )


def _cancel(prompt_id: str, server_url: str):
    try:
        if prompt_id:
            cancel_generation(prompt_id, server_url)
            return progress_html("active", "停止要求を送りました", 0.0), "", gr.update(interactive=False)
        return (
            progress_html("idle", "実行中の H3 ジョブはありません", 0.0),
            "",
            gr.update(interactive=False),
        )
    except H3BridgeError as exc:
        return progress_html("error", str(exc), 0.0), prompt_id, gr.update(interactive=bool(prompt_id))


def _build_ui():
    runtime_root = _initial_runtime()
    runtime_value = os.fspath(runtime_root) if runtime_root else ""
    readiness = inspect_readiness(runtime_root, H3_SERVER_URL)
    initial_history, initial_history_html, initial_history_choices = _history_state(runtime_value)
    initial_history_value = initial_history_choices[0][1] if initial_history_choices else None

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Column(elem_id="h3-studio", elem_classes=["h3-studio-shell"]):
            gr.HTML(
                """
                <section class="h3-hero">
                  <div class="h3-hero-copy">
                    <span class="h3-eyebrow">FORGE NEO · MINIMAX H3 STUDIO</span>
                    <h2>映像と音を、ひとつの流れで。</h2>
                    <p>テキスト、キーフレーム、参照素材から、映像と32kHzステレオ音声を一度に生成します。</p>
                  </div>
                  <div class="h3-hero-mark" aria-hidden="true"><span>H3</span><small>AV</small></div>
                </section>
                """
            )
            runtime_status = gr.HTML(
                value=readiness_html(readiness, RUNTIME_PROFILE_FAST),
                elem_id="h3-runtime-status",
            )

            with gr.Row(elem_classes=["h3-main-grid"]):
                with gr.Column(scale=6, min_width=420, elem_classes=["h3-compose"]):
                    gr.HTML('<div class="h3-section-kicker"><span>01</span><strong>つくり方</strong></div>')
                    mode = gr.Radio(
                        choices=[
                            ("テキスト", MODE_TEXT),
                            ("キーフレーム", MODE_KEYFRAMES),
                            ("参照素材", MODE_REFERENCES),
                        ],
                        value=MODE_TEXT,
                        label="生成モード",
                        show_label=False,
                        elem_id="h3-mode",
                        elem_classes=["h3-mode-selector"],
                    )
                    mode_help = gr.HTML(value=_mode_help_html(MODE_TEXT), elem_id="h3-mode-help")

                    with gr.Group(elem_classes=["h3-prompt-card"]):
                        with gr.Row(elem_classes=["h3-card-heading"]):
                            gr.Markdown("### Prompt\n映像・カメラ・台詞・効果音・音楽を、同じ時系列で書きます。")
                            prompt_template_button = gr.Button(
                                "構成テンプレート",
                                size="sm",
                                elem_id="h3-prompt-template",
                                elem_classes=["h3-quiet-button"],
                            )
                        prompt = gr.Textbox(
                            value="",
                            placeholder=(
                                "例: 雨上がりの東京。赤い傘を持つ人物を低いカメラで追う。\n"
                                "3秒で振り返り、遠くの雷に合わせて街の環境音が一瞬静まる。"
                            ),
                            lines=9,
                            max_lines=16,
                            label="H3 prompt",
                            show_label=False,
                            elem_id="h3-prompt",
                        )
                        input_validation = gr.HTML(value="", elem_id="h3-input-validation")
                        with gr.Row(elem_classes=["h3-prompt-chips"]):
                            camera_button = gr.Button("＋ Camera", size="sm")
                            dialogue_button = gr.Button("＋ Dialogue", size="sm")
                            sfx_button = gr.Button("＋ SFX", size="sm")
                            music_button = gr.Button("＋ Music", size="sm")

                    with gr.Group(visible=False, elem_id="h3-keyframes", elem_classes=["h3-media-panel"]) as keyframe_group:
                        gr.Markdown("### Keyframes\n片方だけでも使えます。両方指定すると、その間の動きを補間します。")
                        with gr.Row():
                            first_frame = gr.Image(
                                type="filepath",
                                label="開始フレーム",
                                height=190,
                                elem_id="h3-first-frame",
                            )
                            last_frame = gr.Image(
                                type="filepath",
                                label="終了フレーム",
                                height=190,
                                elem_id="h3-last-frame",
                            )

                    with gr.Group(visible=False, elem_id="h3-references", elem_classes=["h3-media-panel"]) as reference_group:
                        gr.Markdown(
                            "### References\n"
                            "画像9枚・動画3本・音声3本、合計12個まで。動画と音声の合計は15秒以内です。"
                        )
                        reference_images = gr.File(
                            label="参照画像 · Picture 1–9",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                            elem_id="h3-reference-images",
                        )
                        reference_videos = gr.File(
                            label="参照動画 · Video 1–3（2〜15秒）",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            elem_id="h3-reference-videos",
                        )
                        reference_audios = gr.File(
                            label="参照音声 · Audio 1–3（2〜15秒）",
                            file_count="multiple",
                            file_types=["audio"],
                            type="filepath",
                            elem_id="h3-reference-audios",
                        )
                        reference_guide = gr.HTML(
                            value=reference_guide_html(None, None, None),
                            elem_id="h3-reference-guide",
                        )

                    gr.HTML('<div class="h3-section-kicker"><span>02</span><strong>生成設定</strong></div>')
                    with gr.Group(elem_classes=["h3-settings-card"]):
                        gr.Markdown("#### 用途で選ぶ\n解像度・5秒・20 Stepsをまとめて安全な値へ揃えます。")
                        with gr.Row(elem_classes=["h3-speed-presets"]):
                            quick_preset_button = gr.Button(
                                "動作確認\n速い・低解像度",
                                size="sm",
                                elem_id="h3-preset-quick",
                            )
                            recommended_preset_button = gr.Button(
                                "おすすめ\n公式Fast Preview相当",
                                size="sm",
                                elem_id="h3-preset-recommended",
                            )
                            final_preset_button = gr.Button(
                                "最終出力\nNative・重い",
                                size="sm",
                                elem_id="h3-preset-final",
                            )
                        preset_state = gr.HTML(
                            value=_preset_state_html("recommended"),
                            elem_id="h3-preset-state",
                        )
                        with gr.Row():
                            aspect = gr.Dropdown(
                                choices=["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                                value="16:9",
                                label="Aspect",
                                elem_id="h3-aspect",
                            )
                            quality = gr.Dropdown(
                                choices=[(label, value) for value, label in QUALITY_LABELS.items()],
                                value="preview",
                                label="解像度・速度",
                                elem_id="h3-quality",
                            )
                        duration = gr.Slider(
                            minimum=5,
                            maximum=15,
                            step=0.5,
                            value=5,
                            label="Duration · seconds",
                            elem_id="h3-duration",
                        )
                        settings_summary = gr.HTML(
                            value=settings_summary_html("16:9", "preview", 5, 20),
                            elem_id="h3-settings-summary",
                        )

                        with gr.Accordion("Advanced", open=False, elem_id="h3-advanced"):
                            with gr.Row():
                                steps = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    step=1,
                                    value=20,
                                    label="Steps",
                                    elem_id="h3-steps",
                                )
                                seed = gr.Number(
                                    value=-1,
                                    precision=0,
                                    label="Seed · -1 = random",
                                    elem_id="h3-seed",
                                )
                            with gr.Row():
                                scheduler = gr.Dropdown(
                                    choices=[
                                        ("simple · 公式workflow推奨", "simple"),
                                        ("beta · 実験的", "beta"),
                                        ("normal · 実験的", "normal"),
                                    ],
                                    value="simple",
                                    label="Scheduler",
                                    elem_id="h3-scheduler",
                                )
                                ref_image_size = gr.Radio(
                                    choices=[
                                        ("Match · 推奨・高速", "match"),
                                        ("Max · 同一性優先・非常に重い", "max"),
                                    ],
                                    value="match",
                                    label="Reference image size",
                                    visible=False,
                                    elem_id="h3-ref-image-size",
                                )

                    with gr.Row(elem_classes=["h3-generate-row"]):
                        generate_button = gr.Button(
                            "映像＋音声を生成",
                            variant="primary",
                            size="lg",
                            elem_id="h3-generate",
                            elem_classes=["h3-generate-button"],
                        )
                        cancel_button = gr.Button(
                            "停止",
                            size="lg",
                            interactive=False,
                            elem_id="h3-cancel",
                            elem_classes=["h3-cancel-button"],
                        )

                with gr.Column(scale=5, min_width=390, elem_classes=["h3-output"]):
                    gr.HTML('<div class="h3-section-kicker"><span>03</span><strong>Output</strong></div>')
                    result_video = gr.Video(
                        label="MiniMax H3 result",
                        show_label=False,
                        interactive=False,
                        elem_id="h3-result-video",
                        elem_classes=["h3-video-stage"],
                    )
                    progress = gr.HTML(
                        value=progress_html("idle", "準備ができています", 0.0),
                        elem_id="h3-progress",
                    )
                    gr.HTML(
                        '<div class="h3-output-note"><span>NATIVE AUDIO</span>'
                        '<p>映像と32kHzステレオ音声は同じモデル推論から生成され、MP4へ同期保存されます。</p></div>'
                    )

                    with gr.Accordion("Recent generations", open=True, elem_id="h3-history-accordion"):
                        history_panel = gr.HTML(value=initial_history_html, elem_id="h3-history")
                        history_selector = gr.Dropdown(
                            choices=initial_history_choices,
                            value=initial_history_value,
                            label="履歴を選択",
                            show_label=False,
                            elem_id="h3-history-selector",
                        )
                        with gr.Row():
                            load_history_button = gr.Button("プレイヤーで表示", size="sm")
                            refresh_history_button = gr.Button("履歴を更新", size="sm")

            with gr.Accordion("Runtime & models", open=False, elem_id="h3-runtime-setup"):
                gr.Markdown(
                    "Forge Neo はローカルの ComfyUI H3 runtime を使用します。外部URLには素材を送信しません。"
                    " H3だけにComfy Kitchen INT8 attentionを適用します。生成品質とは別に、"
                    "高速または省RAMの起動profileを明示選択できます。"
                )
                runtime_profile = gr.Radio(
                    choices=[
                        (
                            "高速・推奨 · Pinned Memory + Async 2",
                            RUNTIME_PROFILE_FAST,
                        ),
                        (
                            "省RAM・低速 · cacheなし + Pinned/Async無効",
                            RUNTIME_PROFILE_LOW_RAM,
                        ),
                    ],
                    value=RUNTIME_PROFILE_FAST,
                    label="Runtime profile",
                    elem_id="h3-runtime-profile",
                )
                gr.Markdown(
                    "profile変更は選択しただけでは反映されません。キューが空のときに"
                    "「選択設定で再起動」を押してください。外部起動processは自動停止しません。"
                )
                with gr.Row():
                    runtime_path = gr.Textbox(
                        value=runtime_value,
                        label="ComfyUI folder",
                        placeholder=r"H:\path\to\ComfyUI",
                        elem_id="h3-runtime-path",
                    )
                    server_url = gr.Textbox(
                        value=H3_SERVER_URL,
                        label="Local backend URL",
                        elem_id="h3-server-url",
                    )
                with gr.Row():
                    connect_button = gr.Button("接続 / 起動", variant="primary", elem_id="h3-connect")
                    restart_button = gr.Button("選択設定で再起動", elem_id="h3-restart")
                    rescan_button = gr.Button("状態を再確認", elem_id="h3-rescan")
                gr.HTML(
                    """
                    <div class="h3-license-note">
                      <strong>Local Base · 768p</strong>
                      <span>ローカル公開weightはBaseです。Context-IRと2K Regenerateは外部の有料API専用で、この画面からは呼びません。</span>
                      <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">MiniMax H3 Community License</a>
                      <a href="https://docs.comfy.org/tutorials/video/minimax/minimax-h3" target="_blank" rel="noopener noreferrer">Official ComfyUI guide</a>
                    </div>
                    """
                )

            prompt_id_state = gr.State("")

        summary_inputs = [aspect, quality, duration, steps, scheduler, ref_image_size]
        mode.change(
            fn=_mode_updates,
            inputs=[mode, *summary_inputs, input_validation],
            outputs=[
                keyframe_group,
                reference_group,
                mode_help,
                ref_image_size,
                settings_summary,
                input_validation,
            ],
            queue=False,
        )
        prompt.input(
            fn=_clear_prompt_validation,
            inputs=[input_validation],
            outputs=[input_validation],
            queue=False,
            trigger_mode="always_last",
        )
        for component in [first_frame, last_frame]:
            component.change(
                fn=_clear_keyframe_validation,
                inputs=[input_validation],
                outputs=[input_validation],
                queue=False,
            )
        reference_inputs = [reference_images, reference_videos, reference_audios]
        for component in reference_inputs:
            component.change(
                fn=_reference_guide_updates,
                inputs=[*reference_inputs, input_validation],
                outputs=[reference_guide, input_validation],
                queue=False,
            )
        prompt_action_outputs = [prompt, input_validation]
        for button, callback in [
            (prompt_template_button, _prompt_template_updates),
            (camera_button, _prompt_camera_updates),
            (dialogue_button, _prompt_dialogue_updates),
            (sfx_button, _prompt_sfx_updates),
            (music_button, _prompt_music_updates),
        ]:
            button.click(
                fn=callback,
                inputs=[prompt, input_validation],
                outputs=prompt_action_outputs,
                queue=False,
            )

        aspect.input(
            fn=_aspect_settings_updates,
            inputs=[*summary_inputs, input_validation],
            outputs=[settings_summary, input_validation],
            queue=False,
            trigger_mode="always_last",
        )

        preset_outputs = [
            quality,
            duration,
            steps,
            scheduler,
            ref_image_size,
            settings_summary,
            preset_state,
        ]
        quick_preset_button.click(
            fn=_apply_quick_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_quick_preset",
        )
        recommended_preset_button.click(
            fn=_apply_recommended_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_recommended_preset",
        )
        final_preset_button.click(
            fn=_apply_final_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_final_preset",
        )
        for component in [quality, duration, steps, scheduler, ref_image_size]:
            component.input(
                fn=_custom_settings_updates,
                inputs=[*summary_inputs, input_validation],
                outputs=[settings_summary, preset_state, input_validation],
                queue=False,
                trigger_mode="always_last",
            )
        seed.input(
            fn=_clear_settings_validation,
            inputs=[input_validation],
            outputs=[input_validation],
            queue=False,
            trigger_mode="always_last",
        )

        connect_button.click(
            fn=_connect_runtime,
            inputs=[runtime_path, server_url, runtime_profile],
            outputs=[runtime_status],
        )
        restart_button.click(
            fn=_restart_runtime,
            inputs=[runtime_path, server_url, runtime_profile],
            outputs=[runtime_status],
        )
        rescan_button.click(
            fn=_rescan_runtime,
            inputs=[runtime_path, server_url, runtime_profile],
            outputs=[runtime_status],
            queue=False,
        )
        runtime_profile.change(
            fn=_rescan_runtime,
            inputs=[runtime_path, server_url, runtime_profile],
            outputs=[runtime_status],
            queue=False,
        )
        refresh_history_button.click(
            fn=_refresh_history,
            inputs=[runtime_path],
            outputs=[history_panel, history_selector],
            queue=False,
        )
        load_history_button.click(
            fn=_load_history_video,
            inputs=[history_selector, runtime_path],
            outputs=[result_video, progress],
        )

        generate_button.click(
            fn=_generate,
            inputs=[
                runtime_path,
                server_url,
                runtime_profile,
                mode,
                prompt,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
            ],
            outputs=[
                progress,
                result_video,
                prompt_id_state,
                history_panel,
                history_selector,
                cancel_button,
                generate_button,
                input_validation,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="minimax-h3-generation",
        )
        cancel_button.click(
            fn=_cancel,
            inputs=[prompt_id_state, server_url],
            outputs=[progress, prompt_id_state, cancel_button],
            queue=False,
        )

    return [(interface, "H3 Studio", "minimax_h3_studio")]


script_callbacks.on_ui_tabs(_build_ui, name="minimax_h3_studio")
