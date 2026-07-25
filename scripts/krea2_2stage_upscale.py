import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import devices, images, processing
from modules.krea2_quality import smart_finish_image, smart_finish_summary
from modules.processing import Processed
from modules.shared import opts, state
from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_DENOISE,
    KREA2_STAGE1_DENOISE,
    SAFE_DIFFUSION_LONG_EDGE,
    capped_diffusion_size,
    native_diffusion_long_edge,
    require_native_diffusion_size,
    require_safe_diffusion_size,
    replace_infotext_size,
    target_size,
    two_stage_sizes,
    validate_tile_geometry,
)


class KreaTwoStageUpscale(scripts.Script):
    def title(self):
        return "Krea2 2-Stage Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.HTML(
            """<p align="center">Krea2は解像度proxy（Turboは2K、Rawは1K、customは安全側に2Kと仮定）でimg2imgし、最後に正確な4K納品寸法へ拡大します。</p>"""
        )

        with gr.Row():
            final_long_edge = gr.Slider(
                label="Final Long Edge",
                minimum=512,
                maximum=8192,
                step=64,
                value=4096,
                elem_id=self.elem_id("final_long_edge"),
            )
            first_pass_long_edge = gr.Slider(
                label="Stage 1 Long Edge (0 = Auto)",
                minimum=0,
                maximum=8192,
                step=64,
                value=0,
                elem_id=self.elem_id("first_pass_long_edge"),
            )

        diffusion_long_edge_cap = gr.Slider(
            label="Final Diffusion Long Edge Cap (0 = Profile Auto)",
            minimum=0,
            maximum=SAFE_DIFFUSION_LONG_EDGE,
            step=64,
            value=0,
            elem_id=self.elem_id("diffusion_long_edge_cap"),
        )

        with gr.Row():
            model_profile = gr.Radio(
                label="Krea2 Resolution Guard Profile",
                choices=("custom", "turbo", "raw"),
                value="custom",
                elem_id=self.elem_id("model_profile"),
            )
            allow_non_native_diffusion = gr.Checkbox(
                label="Allow non-native diffusion above profile limit",
                value=False,
                elem_id=self.elem_id("allow_non_native_diffusion"),
            )

        with gr.Row():
            final_width = gr.Slider(
                label="Final Width",
                minimum=0,
                maximum=8192,
                step=16,
                value=0,
                elem_id=self.elem_id("final_width"),
            )
            final_height = gr.Slider(
                label="Final Height",
                minimum=0,
                maximum=8192,
                step=16,
                value=0,
                elem_id=self.elem_id("final_height"),
            )

        with gr.Row():
            first_pass_denoise = gr.Slider(
                label="Stage 1 Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=KREA2_STAGE1_DENOISE,
                elem_id=self.elem_id("first_pass_denoise"),
            )
            final_denoise = gr.Slider(
                label="Final Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=KREA2_DEFAULT_DENOISE,
                elem_id=self.elem_id("final_denoise"),
            )

        method = gr.Radio(
            label="MultiDiffusion Method",
            choices=("MultiDiffusion", "Mixture of Diffusers"),
            value="Mixture of Diffusers",
            elem_id=self.elem_id("method"),
        )

        with gr.Row():
            tile_width = gr.Slider(
                label="Tile Width",
                minimum=256,
                maximum=1280,
                step=64,
                value=768,
                elem_id=self.elem_id("tile_width"),
            )
            tile_height = gr.Slider(
                label="Tile Height",
                minimum=256,
                maximum=1280,
                step=64,
                value=768,
                elem_id=self.elem_id("tile_height"),
            )

        with gr.Row():
            tile_overlap = gr.Slider(
                label="Tile Overlap",
                minimum=0,
                maximum=1024,
                step=16,
                value=96,
                elem_id=self.elem_id("tile_overlap"),
            )
            tile_batch_size = gr.Slider(
                label="Tile Batch Size",
                minimum=1,
                maximum=1,
                step=1,
                value=1,
                elem_id=self.elem_id("tile_batch_size"),
            )

        save_stage1 = gr.Checkbox(
            label="Save Stage 1 output", value=True, elem_id=self.elem_id("save_stage1")
        )

        with gr.Row():
            smart_finish = gr.Checkbox(
                label="Smart chroma finish（改善時のみ採用）",
                value=True,
                elem_id=self.elem_id("smart_finish"),
            )
            smart_despeckle = gr.Checkbox(
                label="孤立粒を補修（雪・星・粒子ではOFF）",
                value=False,
                elem_id=self.elem_id("smart_despeckle"),
            )
            smart_color_strength = gr.Slider(
                label="Smart Chroma Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.80,
                elem_id=self.elem_id("smart_color_strength"),
            )

        return [
            final_long_edge,
            first_pass_long_edge,
            diffusion_long_edge_cap,
            final_width,
            final_height,
            first_pass_denoise,
            final_denoise,
            method,
            tile_width,
            tile_height,
            tile_overlap,
            tile_batch_size,
            save_stage1,
            model_profile,
            allow_non_native_diffusion,
            smart_finish,
            smart_despeckle,
            smart_color_strength,
        ]

    @staticmethod
    def _explicit_dimensions(width: int, height: int) -> tuple[int | None, int | None]:
        explicit_width = int(width) if int(width) > 0 else None
        explicit_height = int(height) if int(height) > 0 else None
        if (explicit_width is None) != (explicit_height is None):
            raise ValueError(
                "Final Width and Final Height must both be 0 or both be > 0."
            )
        return explicit_width, explicit_height

    @staticmethod
    def _effective_diffusion_cap(requested_cap: int, model_profile: str) -> int:
        requested_cap = int(requested_cap)
        if requested_cap < 0:
            raise ValueError("Final Diffusion Long Edge Cap must be >= 0.")
        if requested_cap == 0:
            return native_diffusion_long_edge(str(model_profile))
        return requested_cap

    @staticmethod
    def _enable_multidiffusion(
        p,
        method: str,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
    ):
        if p.scripts is None or p.script_args is None:
            raise RuntimeError("Krea2 2-Stage Upscale requires img2img script context.")

        multidiffusion = next(
            (
                script
                for script in p.scripts.alwayson_scripts
                if script.title() == "MultiDiffusion Integrated"
            ),
            None,
        )
        if multidiffusion is None:
            raise RuntimeError("MultiDiffusion Integrated script is not loaded.")

        replacement = [
            True,
            method,
            int(tile_width),
            int(tile_height),
            int(tile_overlap),
            int(tile_batch_size),
        ]
        if multidiffusion.args_to - multidiffusion.args_from != len(replacement):
            raise RuntimeError("MultiDiffusion Integrated argument layout changed.")

        script_args = list(p.script_args)
        script_args[multidiffusion.args_from : multidiffusion.args_to] = replacement
        p.script_args = script_args

    @staticmethod
    def _validate_run(
        p,
        first_pass_denoise: float,
        final_denoise: float,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
        smart_color_strength: float,
    ):
        if p.batch_size != 1 or p.n_iter != 1:
            raise ValueError(
                "Krea2 2-Stage Upscale supports Batch Count 1 and Batch Size 1 only."
            )
        if not p.init_images or p.init_images[0] is None:
            raise ValueError("Krea2 2-Stage Upscale requires an img2img input image.")
        if p.image_mask is not None or p.latent_mask is not None:
            raise ValueError(
                "Krea2 2-Stage Upscale supports normal img2img only; inpaint masks are not supported."
            )
        if not 0 <= first_pass_denoise <= 1:
            raise ValueError("Stage 1 Denoising Strength must be between 0 and 1.")
        if not 0 <= final_denoise <= 1:
            raise ValueError("Final Denoising Strength must be between 0 and 1.")
        if not 0 <= smart_color_strength <= 1:
            raise ValueError("Smart Chroma Strength must be between 0 and 1.")
        validate_tile_geometry(
            int(tile_width),
            int(tile_height),
            int(tile_overlap),
            int(tile_batch_size),
        )

    def run(
        self,
        p,
        final_long_edge: int,
        first_pass_long_edge: int,
        diffusion_long_edge_cap: int,
        final_width: int,
        final_height: int,
        first_pass_denoise: float,
        final_denoise: float,
        method: str,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
        save_stage1: bool,
        model_profile: str = "custom",
        allow_non_native_diffusion: bool = False,
        smart_finish: bool = True,
        smart_despeckle: bool = False,
        smart_color_strength: float = 0.80,
    ):
        self._validate_run(
            p,
            first_pass_denoise,
            final_denoise,
            tile_width,
            tile_height,
            tile_overlap,
            tile_batch_size,
            smart_color_strength,
        )
        processing.fix_seed(p)

        source = images.flatten(p.init_images[0], opts.img2img_background_color)
        explicit_width, explicit_height = self._explicit_dimensions(
            final_width, final_height
        )
        effective_diffusion_cap = self._effective_diffusion_cap(
            diffusion_long_edge_cap, model_profile
        )
        target_w, target_h = target_size(
            source.width,
            source.height,
            int(final_long_edge),
            explicit_width,
            explicit_height,
        )
        diffusion_w, diffusion_h = capped_diffusion_size(
            source.width,
            source.height,
            target_w,
            target_h,
            effective_diffusion_cap,
        )
        require_safe_diffusion_size(diffusion_w, diffusion_h)
        require_native_diffusion_size(
            diffusion_w,
            diffusion_h,
            str(model_profile),
            bool(allow_non_native_diffusion),
        )
        needs_final_resize = (diffusion_w, diffusion_h) != (target_w, target_h)
        if int(first_pass_long_edge) < 0:
            raise ValueError("Stage 1 Long Edge must be >= 0.")

        use_stage1 = max(diffusion_w, diffusion_h) >= max(source.size) + 128
        if use_stage1:
            (stage1_w, stage1_h), _ = two_stage_sizes(
                source.width,
                source.height,
                diffusion_w,
                diffusion_h,
                int(first_pass_long_edge),
            )
            stage1_long_edge = max(stage1_w, stage1_h)
            stage1_long_edge_label = (
                f"Auto ({stage1_long_edge})"
                if int(first_pass_long_edge) == 0
                else str(stage1_long_edge)
            )
            flow_label = "two-stage"
            stage1_status = "run"
        else:
            stage1_w, stage1_h = source.size
            stage1_long_edge_label = "Skipped (source near diffusion proxy)"
            flow_label = "single-stage"
            stage1_status = "skipped (source near diffusion proxy)"

        original_batch_size = p.batch_size
        original_n_iter = p.n_iter
        original_save_samples = p.save_samples()
        original_do_not_save_samples = p.do_not_save_samples
        original_do_not_save_grid = p.do_not_save_grid
        original_resize_mode = p.resize_mode
        original_width = p.width
        original_height = p.height
        original_denoising_strength = p.denoising_strength
        original_init_images = p.init_images
        original_script_args = p.script_args

        try:
            self._enable_multidiffusion(
                p, method, tile_width, tile_height, tile_overlap, tile_batch_size
            )
            p.batch_size = 1
            p.n_iter = 1
            p.resize_mode = 0
            p.do_not_save_grid = True

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": (
                        "Stage 1/2" if use_stage1 else "Single-stage diffusion"
                    ),
                    "Krea2 Upscale flow": flow_label,
                    "Krea2 Stage 1 status": stage1_status,
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": effective_diffusion_cap,
                    "Krea2 Model profile": str(model_profile),
                    "Krea2 Smart Finish": bool(smart_finish),
                    "Krea2 Smart Finish strength": float(smart_color_strength),
                    "Krea2 Smart Finish despeckle": bool(smart_despeckle),
                    "Krea2 Smart Finish analysis long edge": 1536,
                }
            )
            if use_stage1:
                p.width = stage1_w
                p.height = stage1_h
                p.denoising_strength = float(first_pass_denoise)
                p.init_images = [
                    source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
                ]
                p.do_not_save_samples = original_do_not_save_samples or not save_stage1
                state.job = "Krea2 2-Stage Upscale: Stage 1/2"
                state.textinfo = f"Krea2 2-Stage Stage 1/2: {stage1_w}x{stage1_h}"

                stage1 = processing.process_images(p)
                if state.interrupted or state.skipped or state.stopping_generation:
                    return stage1
                if not stage1.images:
                    raise RuntimeError(
                        "Krea2 2-Stage Upscale Stage 1 returned no image."
                    )

                stage1_image = images.flatten(
                    stage1.images[0], opts.img2img_background_color
                )
                p.latents_after_sampling.clear()
                p.pixels_after_sampling.clear()
                devices.torch_gc()
            else:
                stage1_image = source

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": (
                        "Stage 2/2" if use_stage1 else "Single-stage diffusion"
                    ),
                    "Krea2 Upscale flow": flow_label,
                    "Krea2 Stage 1 status": stage1_status,
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": effective_diffusion_cap,
                    "Krea2 Model profile": str(model_profile),
                    "Krea2 Smart Finish": bool(smart_finish),
                    "Krea2 Smart Finish strength": float(smart_color_strength),
                    "Krea2 Smart Finish despeckle": bool(smart_despeckle),
                    "Krea2 Smart Finish analysis long edge": 1536,
                }
            )
            p.width = diffusion_w
            p.height = diffusion_h
            p.denoising_strength = float(final_denoise)
            p.init_images = [
                stage1_image.resize(
                    (diffusion_w, diffusion_h), Image.Resampling.LANCZOS
                )
            ]
            # Stage 2 must never save before deterministic resize and Smart Finish.
            p.do_not_save_samples = True
            p.do_not_save_grid = True
            if use_stage1:
                state.job = "Krea2 2-Stage Upscale: Stage 2/2"
                state.textinfo = f"Krea2 2-Stage Stage 2/2: {diffusion_w}x{diffusion_h}"
            else:
                state.job = "Krea2 2-Stage Upscale: Single diffusion stage"
                state.textinfo = (
                    "Krea2 single-stage diffusion "
                    f"(Stage 1 skipped): {diffusion_w}x{diffusion_h}"
                )

            final = processing.process_images(p)
            if final is None:
                return Processed(p, [], p.seed, "")
            if state.interrupted or state.skipped or state.stopping_generation:
                return final

            p.latents_after_sampling.clear()
            p.pixels_after_sampling.clear()
            devices.torch_gc()

            if needs_final_resize:
                final.info = replace_infotext_size(
                    final.info,
                    diffusion_w,
                    diffusion_h,
                    target_w,
                    target_h,
                )
                final.infotexts = [
                    replace_infotext_size(
                        text,
                        diffusion_w,
                        diffusion_h,
                        target_w,
                        target_h,
                    )
                    for text in final.infotexts
                ]

            delivery_images = []
            quality_summaries = []
            for image in final.images:
                delivery = images.flatten(image, opts.img2img_background_color)
                if delivery.size != (target_w, target_h):
                    delivery = delivery.resize(
                        (target_w, target_h), Image.Resampling.LANCZOS
                    )

                if smart_finish:
                    state.job = "Krea2 2-Stage Upscale: Smart Finish"
                    state.textinfo = f"Krea2 Smart Finish: {target_w}x{target_h} (CPU)"
                    delivery, quality_report = smart_finish_image(
                        delivery,
                        color_strength=float(smart_color_strength),
                        analysis_long_edge=1536,
                        despeckle=bool(smart_despeckle),
                    )
                    quality_summaries.append(smart_finish_summary(quality_report))
                else:
                    quality_summaries.append("disabled")
                delivery_images.append(delivery)

            for index, summary in enumerate(quality_summaries):
                if index < len(final.infotexts):
                    final.infotexts[index] = (
                        f"{final.infotexts[index]}, Krea2 Smart Finish result: {summary}"
                    )
            if final.infotexts:
                final.info = final.infotexts[0]

            if opts.enable_pnginfo:
                for index, image in enumerate(delivery_images):
                    info = (
                        final.infotexts[index]
                        if index < len(final.infotexts)
                        else final.info
                    )
                    image.info["parameters"] = info

            final.images = delivery_images
            final.width = target_w
            final.height = target_h
            p.width = target_w
            p.height = target_h

            if original_save_samples:
                first_image = final.index_of_first_image
                for index, image in enumerate(
                    delivery_images[first_image:], start=first_image
                ):
                    info = (
                        final.infotexts[index]
                        if index < len(final.infotexts)
                        else final.info
                    )
                    images.save_image(
                        image,
                        p.outpath_samples,
                        "",
                        p.seed,
                        p.prompt,
                        opts.samples_format,
                        info=info,
                        p=p,
                    )
            return final
        finally:
            p.batch_size = original_batch_size
            p.n_iter = original_n_iter
            p.do_not_save_samples = original_do_not_save_samples
            p.do_not_save_grid = original_do_not_save_grid
            p.resize_mode = original_resize_mode
            p.width = original_width
            p.height = original_height
            p.denoising_strength = original_denoising_strength
            p.init_images = original_init_images
            p.script_args = original_script_args
