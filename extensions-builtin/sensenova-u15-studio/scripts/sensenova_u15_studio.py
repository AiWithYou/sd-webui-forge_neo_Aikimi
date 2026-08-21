from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from modules import script_callbacks
from modules.paths import data_path
from modules_forge.sensenova_u15_bridge import (
    DEFAULT_MODEL_ID,
    DEFAULT_Q8_PATH,
    DEFAULT_SOURCE_PATH,
    MODE_EDIT,
    MODE_TEXT,
    QUANT_BF16,
    QUANT_Q8,
    SenseNovaBridgeError,
    SenseNovaGenerationCancelled,
    SenseNovaRequest,
    cancel_generation,
    inspect_runtime,
    normalize_gallery_images,
    parse_resolution,
    progress_html,
    reference_order_html,
    request_summary_html,
    resolution_choices,
    run_generation,
    runtime_status_html,
    validate_request,
)


OUTPUT_DIRECTORY = Path(data_path) / "outputs" / "sensenova_u15"
CACHE_DIRECTORY = Path(data_path) / "cache" / "sensenova_u15"
LOG_DIRECTORY = Path(data_path) / "logs" / "sensenova_u15"


def _gallery_list(value: Any) -> list[Any]:
    return list(value or [])


def _append_reference(gallery: Any, upload: Image.Image | None):
    values = _gallery_list(gallery)
    if upload is None:
        return gr.update(), gr.update(), reference_order_html(values), -1
    values.append((upload.copy(), None))
    if len(values) > 8:
        values = values[:8]
        message = (
            '<p class="sn-inline-error" role="alert">参照画像は最大8枚です。</p>'
            + reference_order_html(values)
        )
    else:
        message = reference_order_html(values)
    return gr.update(value=values), gr.update(value=None), message, len(values) - 1


def _replace_reference(index: int, gallery: Any, upload: Image.Image | None):
    values = _gallery_list(gallery)
    if upload is None or index < 0 or index >= len(values):
        return gr.update(), gr.update(), reference_order_html(values), index
    values[index] = (upload.copy(), None)
    return (
        gr.update(value=values),
        gr.update(value=None),
        reference_order_html(values),
        index,
    )


def _remove_reference(index: int, gallery: Any):
    values = _gallery_list(gallery)
    if 0 <= index < len(values):
        values.pop(index)
    next_index = min(index, len(values) - 1) if values else -1
    return gr.update(value=values), reference_order_html(values), next_index


def _move_reference(index: int, gallery: Any, offset: int):
    values = _gallery_list(gallery)
    target = index + offset
    if 0 <= index < len(values) and 0 <= target < len(values):
        values[index], values[target] = values[target], values[index]
        index = target
    return gr.update(value=values), reference_order_html(values), index


def _clear_references():
    return gr.update(value=[]), reference_order_html([]), -1


def _select_reference(evt: gr.SelectData) -> int:
    return int(evt.index)


def _mode_updates(mode: str, current_resolution: str):
    choices = resolution_choices(mode)
    values = {value for _, value in choices}
    if mode == MODE_EDIT:
        selected = "auto"
    else:
        selected = (
            current_resolution
            if current_resolution in values and current_resolution != "auto"
            else "2048x2048"
        )
    return (
        gr.update(visible=mode == MODE_EDIT),
        gr.update(choices=choices, value=selected),
        gr.update(visible=mode == MODE_EDIT),
        gr.update(visible=mode == MODE_EDIT),
        gr.update(
            value=(
                '<div class="sn-mode-note"><b>MULTI-IMAGE EDIT</b><span>入力順を保ったまま最大8枚を1本の画像トークン列へ渡します。</span></div>'
                if mode == MODE_EDIT
                else '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>参照画像を使わず、公式解像度バケットから生成します。</span></div>'
            )
        ),
    )


def _refresh_runtime(source_path: str, quantization: str, gguf_path: str):
    try:
        status = inspect_runtime(
            source_path, quantization=quantization, gguf_checkpoint=gguf_path
        )
        return runtime_status_html(status)
    except (OSError, ValueError, SenseNovaBridgeError) as exc:
        return (
            '<section class="sn-runtime sn-runtime-setup" role="alert">'
            f"<strong>準備状況を確認できません</strong><p>{html.escape(str(exc))}</p></section>"
        )


def _quantization_updates(quantization: str, source_path: str, gguf_path: str):
    q8 = quantization == QUANT_Q8
    note = (
        "Q8_0 GGUFは約18.58 GiB。重みはINT8のまま保持し、Linear演算時に公式diffusers経路で復号します。"
        if q8
        else "BF16は約36 GiB級です。初回にHugging Faceから公式重みを取得し、十分なRAM/VRAMが必要です。"
    )
    return (
        gr.update(visible=q8),
        f'<p class="sn-quant-note">{html.escape(note)}</p>',
        gr.update(value=DEFAULT_MODEL_ID, interactive=not q8),
        _refresh_runtime(source_path, quantization, gguf_path),
    )


def _request_from_ui(
    mode: str,
    prompt: str,
    gallery: Any,
    model_path: str,
    quantization: str,
    gguf_path: str,
    source_path: str,
    resolution: str,
    input_max_pixels: str,
    steps: int,
    cfg_scale: float,
    img_cfg_scale: float,
    timestep_shift: float,
    seed: int,
    vram_mode: str,
    attn_backend: str,
    dtype: str,
    *,
    should_validate: bool = True,
) -> SenseNovaRequest:
    width, height = parse_resolution(resolution, mode)
    images = normalize_gallery_images(gallery) if mode == MODE_EDIT else ()
    request = SenseNovaRequest(
        mode=mode,
        prompt=str(prompt or ""),
        model_path=str(model_path or ""),
        quantization=quantization,
        gguf_checkpoint=str(gguf_path or ""),
        source_path=str(source_path or ""),
        input_images=images,
        width=width,
        height=height,
        target_pixels=2048 * 2048,
        input_max_pixels=input_max_pixels,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        img_cfg_scale=float(img_cfg_scale),
        timestep_shift=float(timestep_shift),
        seed=int(seed),
        vram_mode=vram_mode,
        attn_backend=attn_backend,
        dtype=dtype,
    )
    if should_validate:
        validate_request(request)
    return request


def _summary_from_ui(
    mode,
    prompt,
    gallery,
    model_path,
    quantization,
    gguf_path,
    source_path,
    resolution,
    input_max_pixels,
    steps,
    cfg_scale,
    img_cfg_scale,
    timestep_shift,
    seed,
    vram_mode,
    attn_backend,
    dtype,
):
    try:
        request = _request_from_ui(
            mode,
            prompt,
            gallery,
            model_path,
            quantization,
            gguf_path,
            source_path,
            resolution,
            input_max_pixels,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
            should_validate=False,
        )
        try:
            validate_request(request)
            message = ""
        except (SenseNovaBridgeError, ValueError, TypeError, OverflowError) as exc:
            message = (
                f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>'
            )
        return request_summary_html(request), message
    except (SenseNovaBridgeError, ValueError, TypeError, OverflowError) as exc:
        return (
            gr.update(),
            f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>',
        )


def _generate(
    mode,
    prompt,
    gallery,
    model_path,
    quantization,
    gguf_path,
    source_path,
    resolution,
    input_max_pixels,
    steps,
    cfg_scale,
    img_cfg_scale,
    timestep_shift,
    seed,
    vram_mode,
    attn_backend,
    dtype,
):
    yield (
        progress_html("prepare", "生成条件を確認しています", 0.01),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "",
        gr.update(interactive=True),
        gr.update(value="生成を準備中…", interactive=False),
        "",
    )
    try:
        request = _request_from_ui(
            mode,
            prompt,
            gallery,
            model_path,
            quantization,
            gguf_path,
            source_path,
            resolution,
            input_max_pixels,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
        )
        for update in run_generation(
            request,
            output_directory=OUTPUT_DIRECTORY,
            cache_directory=CACHE_DIRECTORY,
            log_directory=LOG_DIRECTORY,
        ):
            completed = update["stage"] == "complete"
            yield (
                progress_html(
                    update["stage"],
                    update["message"],
                    update.get("progress", 0.0),
                    update.get("elapsed"),
                ),
                update.get("path") or gr.update(),
                update.get("path") or gr.update(),
                update.get("metadata_path") or gr.update(),
                update.get("metadata") or gr.update(),
                "" if completed else update.get("job_id", ""),
                gr.update(interactive=not completed),
                gr.update(
                    value="画像を生成" if completed else "生成中…",
                    interactive=completed,
                ),
                "",
            )
    except SenseNovaGenerationCancelled as exc:
        yield (
            progress_html("cancelled", str(exc), 0.0),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            gr.update(interactive=False),
            gr.update(value="画像を生成", interactive=True),
            "",
        )
    except (SenseNovaBridgeError, OSError, ValueError, TypeError, OverflowError) as exc:
        yield (
            progress_html("error", "生成を開始できませんでした", 0.0),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            gr.update(interactive=False),
            gr.update(value="画像を生成", interactive=True),
            f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>',
        )


def _cancel(job_id: str):
    message = cancel_generation(job_id or None)
    return progress_html("cancel", message, 0.0), gr.update(interactive=False)


def _build_ui():
    initial_status = inspect_runtime(
        DEFAULT_SOURCE_PATH, quantization=QUANT_Q8, gguf_checkpoint=DEFAULT_Q8_PATH
    )
    initial_request = SenseNovaRequest(mode=MODE_TEXT, prompt="")

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Column(elem_id="sensenova-u15-studio", elem_classes=["sn-shell"]):
            gr.HTML(
                """
                <section class="sn-hero">
                  <div>
                    <span class="sn-eyebrow">FORGE NEO · NATIVE MULTIMODAL</span>
                    <h2>SenseNova U1.5 Studio</h2>
                    <p>テキスト生成と、最大8枚を順番どおり使う複数画像編集。Q8_0 INT8をRTX 3090向け低VRAM経路で実行します。</p>
                  </div>
                  <div class="sn-hero-mark" aria-hidden="true"><b>U1.5</b><small>MoT</small></div>
                </section>
                """
            )
            runtime_status = gr.HTML(
                value=runtime_status_html(initial_status), elem_id="sn-runtime-status"
            )

            with gr.Row(elem_classes=["sn-main-grid"]):
                with gr.Column(scale=6, min_width=430, elem_classes=["sn-compose"]):
                    gr.HTML(
                        '<div class="sn-section-title"><span>01</span><h3>つくる</h3></div>'
                    )
                    mode = gr.Radio(
                        choices=[
                            ("テキストから生成", MODE_TEXT),
                            ("複数画像を編集", MODE_EDIT),
                        ],
                        value=MODE_TEXT,
                        label="生成モード",
                        show_label=False,
                        elem_id="sn-mode",
                        elem_classes=["sn-mode"],
                    )
                    mode_note = gr.HTML(
                        '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>参照画像を使わず、公式解像度バケットから生成します。</span></div>',
                        elem_id="sn-mode-note",
                    )
                    with gr.Group(elem_classes=["sn-prompt-card"]):
                        gr.Markdown(
                            "### プロンプト\n被写体、構図、質感、照明、文字、残したい要素を具体的に書きます。"
                        )
                        prompt = gr.Textbox(
                            label="SenseNovaプロンプト",
                            show_label=False,
                            lines=8,
                            max_lines=18,
                            placeholder="例: 1枚目の人物と2枚目の衣装を組み合わせ、3枚目の照明と色調を保った広告写真にする。背景の構図は変えない。",
                            elem_id="sn-prompt",
                        )
                        gr.HTML(
                            '<div class="sn-prompt-meta"><span>Ctrl / ⌘ + Enter で生成</span>'
                            '<span id="sn-draft-status">下書きをこの端末に自動保存</span>'
                            '<span id="sn-prompt-count">0 / 20,000</span></div>'
                        )
                        validation = gr.HTML(value="", elem_id="sn-validation")

                    with gr.Group(
                        visible=False,
                        elem_classes=["sn-reference-card"],
                        elem_id="sn-references",
                    ) as reference_group:
                        gr.Markdown(
                            "### 参照画像\n画像を追加した順番が `<image>` の順番です。選択後に左右へ移動できます。"
                        )
                        reference_gallery = gr.Gallery(
                            value=[],
                            type="pil",
                            interactive=True,
                            label="参照画像（最大8枚）",
                            show_label=False,
                            columns=4,
                            rows=2,
                            height=330,
                            allow_preview=True,
                            object_fit="contain",
                            elem_id="sn-reference-gallery",
                        )
                        selected_reference = gr.State(-1)
                        reference_order = gr.HTML(
                            value=reference_order_html([]), elem_id="sn-reference-order"
                        )
                        with gr.Row():
                            upload = gr.Image(
                                type="pil",
                                sources=["upload", "clipboard"],
                                label="追加・差し替え画像",
                                height=210,
                                elem_id="sn-reference-upload",
                            )
                            with gr.Column(min_width=190):
                                append_button = gr.Button(
                                    "末尾へ追加", variant="secondary"
                                )
                                replace_button = gr.Button(
                                    "選択画像を差し替え", variant="secondary"
                                )
                                with gr.Row():
                                    move_left_button = gr.Button("← 前へ", size="sm")
                                    move_right_button = gr.Button("後ろへ →", size="sm")
                                remove_button = gr.Button(
                                    "選択画像を削除", variant="stop"
                                )
                                clear_button = gr.Button("すべて消去", variant="stop")

                    gr.HTML(
                        '<div class="sn-section-title"><span>02</span><h3>生成設定</h3></div>'
                    )
                    with gr.Group(elem_classes=["sn-settings-card"]):
                        resolution = gr.Dropdown(
                            choices=resolution_choices(MODE_TEXT),
                            value="2048x2048",
                            label="出力解像度",
                            elem_id="sn-resolution",
                        )
                        with gr.Row():
                            steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                            seed = gr.Number(
                                value=42, precision=0, label="Seed", minimum=0
                            )
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                0, 20, value=4.0, step=0.1, label="CFG"
                            )
                            img_cfg_scale = gr.Slider(
                                0,
                                20,
                                value=1.0,
                                step=0.1,
                                label="Image CFG",
                                visible=False,
                                elem_id="sn-image-cfg",
                            )
                            timestep_shift = gr.Slider(
                                0.1, 20, value=3.0, step=0.1, label="Timestep Shift"
                            )
                        input_max_pixels = gr.Dropdown(
                            choices=[
                                ("自動 · 画像枚数に応じて配分", "auto"),
                                ("2048² / 画像 · 高忠実度", str(2048 * 2048)),
                                ("1024² / 画像 · 省メモリ", str(1024 * 1024)),
                                ("512² / 画像 · 動作確認", str(512 * 512)),
                            ],
                            value="auto",
                            label="参照画像ごとの入力予算",
                            visible=False,
                        )
                        with gr.Accordion("詳細設定", open=False):
                            with gr.Row():
                                vram_mode = gr.Dropdown(
                                    choices=[
                                        ("Low · RTX 3090推奨", "low"),
                                        ("Balanced · 転送を並列化", "balanced"),
                                        ("Fast · 余裕のあるGPU", "fast"),
                                        ("Full · 全重みをGPUへ", "full"),
                                    ],
                                    value="low",
                                    label="VRAMモード",
                                )
                                attn_backend = gr.Dropdown(
                                    choices=[
                                        ("自動", "auto"),
                                        ("PyTorch SDPA", "sdpa"),
                                        ("FlashAttention", "flash"),
                                    ],
                                    value="auto",
                                    label="Attention",
                                )
                                dtype = gr.Dropdown(
                                    choices=[
                                        ("BF16計算", "bfloat16"),
                                        ("FP16計算", "float16"),
                                        ("FP32計算", "float32"),
                                    ],
                                    value="bfloat16",
                                    label="計算精度",
                                )

                with gr.Column(
                    scale=5, min_width=390, elem_classes=["sn-result-column"]
                ):
                    gr.HTML(
                        '<div class="sn-section-title"><span>03</span><h3>モデルと結果</h3></div>'
                    )
                    with gr.Group(elem_classes=["sn-model-card"]):
                        quantization = gr.Radio(
                            choices=[
                                ("INT8 · Q8_0 GGUF", QUANT_Q8),
                                ("公式 BF16", QUANT_BF16),
                            ],
                            value=QUANT_Q8,
                            label="重み形式",
                            elem_id="sn-quantization",
                        )
                        quant_note = gr.HTML(
                            '<p class="sn-quant-note">Q8_0 GGUFは約18.58 GiB。重みはINT8のまま保持し、Linear演算時に公式diffusers経路で復号します。</p>'
                        )
                        model_path = gr.Textbox(
                            value=DEFAULT_MODEL_ID,
                            label="モデルID / ローカルモデル",
                            interactive=False,
                        )
                        with gr.Group(visible=True) as q8_group:
                            gguf_path = gr.Textbox(
                                value=os.fspath(DEFAULT_Q8_PATH), label="Q8_0 GGUF"
                            )
                        with gr.Accordion("ランタイム場所", open=False):
                            source_path = gr.Textbox(
                                value=os.fspath(DEFAULT_SOURCE_PATH),
                                label="公式推論コードの src",
                            )
                            gr.Markdown(
                                "未準備の場合は Forge Neo 直下の `download_sensenova_u15_int8.bat` を実行します。"
                            )
                        refresh_button = gr.Button(
                            "準備状況を再確認",
                            variant="secondary",
                            elem_id="sn-refresh-runtime",
                        )

                    summary = gr.HTML(
                        value=request_summary_html(initial_request),
                        elem_id="sn-summary",
                    )
                    progress = gr.HTML(
                        value=progress_html("idle", "生成待ち", 0.0),
                        elem_id="sn-progress",
                    )
                    result_image = gr.Image(
                        type="filepath",
                        label="生成結果",
                        interactive=False,
                        height=520,
                        elem_id="sn-result-image",
                    )
                    with gr.Row():
                        result_file = gr.File(label="PNG", interactive=False)
                        metadata_file = gr.File(label="設定JSON", interactive=False)
                    metadata = gr.JSON(label="生成メタデータ", visible=True)
                    job_id = gr.State("")
                    with gr.Row(elem_classes=["sn-actions"]):
                        cancel_button = gr.Button(
                            "キャンセル",
                            variant="stop",
                            interactive=False,
                            elem_id="sn-cancel",
                        )
                        generate_button = gr.Button(
                            "画像を生成", variant="primary", elem_id="sn-generate"
                        )

        reference_gallery.select(
            _select_reference,
            outputs=[selected_reference],
            queue=False,
            show_progress=False,
        )
        reference_gallery.change(
            reference_order_html,
            inputs=[reference_gallery],
            outputs=[reference_order],
            queue=False,
            trigger_mode="always_last",
        )
        gallery_upload_event = reference_gallery.upload(
            reference_order_html,
            inputs=[reference_gallery],
            outputs=[reference_order],
            queue=False,
        )
        append_event = append_button.click(
            _append_reference,
            inputs=[reference_gallery, upload],
            outputs=[reference_gallery, upload, reference_order, selected_reference],
            queue=False,
        )
        replace_event = replace_button.click(
            _replace_reference,
            inputs=[selected_reference, reference_gallery, upload],
            outputs=[reference_gallery, upload, reference_order, selected_reference],
            queue=False,
        )
        remove_event = remove_button.click(
            _remove_reference,
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        move_left_event = move_left_button.click(
            lambda index, gallery: _move_reference(index, gallery, -1),
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        move_right_event = move_right_button.click(
            lambda index, gallery: _move_reference(index, gallery, 1),
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        clear_event = clear_button.click(
            _clear_references,
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )

        mode.change(
            _mode_updates,
            inputs=[mode, resolution],
            outputs=[
                reference_group,
                resolution,
                img_cfg_scale,
                input_max_pixels,
                mode_note,
            ],
            queue=False,
        )
        quantization.change(
            _quantization_updates,
            inputs=[quantization, source_path, gguf_path],
            outputs=[q8_group, quant_note, model_path, runtime_status],
            queue=False,
        )
        refresh_button.click(
            _refresh_runtime,
            inputs=[source_path, quantization, gguf_path],
            outputs=[runtime_status],
            queue=False,
        )

        request_inputs = [
            mode,
            prompt,
            reference_gallery,
            model_path,
            quantization,
            gguf_path,
            source_path,
            resolution,
            input_max_pixels,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
        ]
        for component in [
            mode,
            reference_gallery,
            model_path,
            quantization,
            gguf_path,
            source_path,
            resolution,
            input_max_pixels,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
        ]:
            component.change(
                _summary_from_ui,
                inputs=request_inputs,
                outputs=[summary, validation],
                queue=False,
                trigger_mode="always_last",
            )
        for reference_event in [
            gallery_upload_event,
            append_event,
            replace_event,
            remove_event,
            move_left_event,
            move_right_event,
            clear_event,
        ]:
            reference_event.then(
                _summary_from_ui,
                inputs=request_inputs,
                outputs=[summary, validation],
                queue=False,
            )

        generate_button.click(
            _generate,
            inputs=request_inputs,
            outputs=[
                progress,
                result_image,
                result_file,
                metadata_file,
                metadata,
                job_id,
                cancel_button,
                generate_button,
                validation,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="sensenova-u15-generation",
        )
        cancel_button.click(
            _cancel,
            inputs=[job_id],
            outputs=[progress, cancel_button],
            queue=False,
        )
        interface.load(
            _refresh_runtime,
            inputs=[source_path, quantization, gguf_path],
            outputs=[runtime_status],
            queue=False,
        )

    return [(interface, "SenseNova U1.5", "sensenova_u15_studio")]


script_callbacks.on_ui_tabs(_build_ui, name="sensenova_u15_studio")
