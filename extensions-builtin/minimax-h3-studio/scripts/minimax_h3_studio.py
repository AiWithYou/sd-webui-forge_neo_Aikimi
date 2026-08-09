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
    append_prompt_section,
    cache_history_video,
    cancel_generation,
    discover_runtime_root,
    history_choices,
    history_html,
    inspect_readiness,
    list_history,
    normalize_file_list,
    progress_html,
    prompt_template,
    readiness_html,
    reference_guide_html,
    resolve_runtime_root,
    run_generation,
    settings_summary_html,
    start_runtime,
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
    "draft": "下書き · 0.2 MP",
    "preview": "プレビュー · 0.4 MP",
    "balanced": "バランス · 0.5 MP",
    "native": "Native · 768p",
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


def _mode_updates(mode: str):
    return (
        gr.update(visible=mode == MODE_KEYFRAMES),
        gr.update(visible=mode == MODE_REFERENCES),
        _mode_help_html(mode),
        gr.update(visible=mode == MODE_REFERENCES),
    )


def _status_error(message: str) -> str:
    return (
        '<div class="h3-runtime-card" data-tone="error" role="alert">'
        '<div class="h3-runtime-badges"><span data-tone="error"><i></i>確認が必要</span></div>'
        f'<p>{html.escape(message)}</p></div>'
    )


def _connect_runtime(runtime_value: str, server_url: str) -> str:
    try:
        root = resolve_runtime_root(runtime_value)
        readiness = start_runtime(root, server_url, LOG_DIRECTORY)
        return readiness_html(readiness)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _rescan_runtime(runtime_value: str, server_url: str) -> str:
    try:
        root = resolve_runtime_root(runtime_value)
        return readiness_html(inspect_readiness(root, server_url))
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _history_state(runtime_value: str):
    try:
        root = resolve_runtime_root(runtime_value) if runtime_value else None
    except H3BridgeError:
        root = None
    items = list_history(root, OUTPUT_DIRECTORY)
    return items, history_html(items), history_choices(items)


def _refresh_history(runtime_value: str):
    items, rendered, choices = _history_state(runtime_value)
    selected = choices[0][1] if choices else None
    return rendered, gr.update(choices=choices, value=selected)


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
    )
    try:
        root = resolve_runtime_root(runtime_value)
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
        for update in run_generation(
            request,
            root,
            server_url,
            LOG_DIRECTORY,
            OUTPUT_DIRECTORY,
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
            )
    except (H3BridgeError, ValueError, TypeError) as exc:
        yield (
            progress_html("error", str(exc), 0.0),
            gr.update(),
            "",
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
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
            runtime_status = gr.HTML(value=readiness_html(readiness), elem_id="h3-runtime-status")

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
                                label="Quality",
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
                                    choices=["simple", "beta", "normal"],
                                    value="simple",
                                    label="Scheduler",
                                    elem_id="h3-scheduler",
                                )
                                ref_image_size = gr.Radio(
                                    choices=[("Match · faster", "match"), ("Max · identity", "max")],
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

        mode.change(
            fn=_mode_updates,
            inputs=[mode],
            outputs=[keyframe_group, reference_group, mode_help, ref_image_size],
            queue=False,
        )
        reference_inputs = [reference_images, reference_videos, reference_audios]
        for component in reference_inputs:
            component.change(
                fn=reference_guide_html,
                inputs=reference_inputs,
                outputs=[reference_guide],
                queue=False,
            )
        prompt_template_button.click(fn=prompt_template, inputs=[prompt], outputs=[prompt], queue=False)
        camera_button.click(fn=lambda value: append_prompt_section(value, "camera"), inputs=[prompt], outputs=[prompt], queue=False)
        dialogue_button.click(fn=lambda value: append_prompt_section(value, "dialogue"), inputs=[prompt], outputs=[prompt], queue=False)
        sfx_button.click(fn=lambda value: append_prompt_section(value, "sfx"), inputs=[prompt], outputs=[prompt], queue=False)
        music_button.click(fn=lambda value: append_prompt_section(value, "music"), inputs=[prompt], outputs=[prompt], queue=False)

        summary_inputs = [aspect, quality, duration, steps]
        for component in summary_inputs:
            component.change(
                fn=settings_summary_html,
                inputs=summary_inputs,
                outputs=[settings_summary],
                queue=False,
            )

        connect_button.click(
            fn=_connect_runtime,
            inputs=[runtime_path, server_url],
            outputs=[runtime_status],
        )
        rescan_button.click(
            fn=_rescan_runtime,
            inputs=[runtime_path, server_url],
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
            outputs=[progress, result_video, prompt_id_state, history_panel, history_selector, cancel_button],
        )
        cancel_button.click(
            fn=_cancel,
            inputs=[prompt_id_state, server_url],
            outputs=[progress, prompt_id_state, cancel_button],
            queue=False,
        )

    return [(interface, "H3 Studio", "minimax_h3_studio")]


script_callbacks.on_ui_tabs(_build_ui, name="minimax_h3_studio")
