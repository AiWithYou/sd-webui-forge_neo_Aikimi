import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import devices, images, processing
from modules.processing import Processed
from modules.shared import opts, state
from modules_forge.krea2_upscale import (
    SAFE_DIFFUSION_LONG_EDGE,
    capped_diffusion_size,
    require_safe_diffusion_size,
    target_size,
    two_stage_sizes,
)


class KreaTwoStageUpscale(scripts.Script):
    def title(self):
        return "Krea2 2-Stage Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.HTML(
            """<p align="center">Run img2img at an intermediate size first, then upscale and run the final img2img pass.</p>"""
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
            label="Final Diffusion Long Edge Cap",
            minimum=1024,
            maximum=SAFE_DIFFUSION_LONG_EDGE,
            step=64,
            value=SAFE_DIFFUSION_LONG_EDGE,
            elem_id=self.elem_id("diffusion_long_edge_cap"),
        )

        with gr.Row():
            final_width = gr.Slider(
                label="Final Width",
                minimum=0,
                maximum=8192,
                step=64,
                value=0,
                elem_id=self.elem_id("final_width"),
            )
            final_height = gr.Slider(
                label="Final Height",
                minimum=0,
                maximum=8192,
                step=64,
                value=0,
                elem_id=self.elem_id("final_height"),
            )

        with gr.Row():
            first_pass_denoise = gr.Slider(
                label="Stage 1 Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.22,
                elem_id=self.elem_id("first_pass_denoise"),
            )
            final_denoise = gr.Slider(
                label="Final Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.28,
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
                maximum=2048,
                step=64,
                value=768,
                elem_id=self.elem_id("tile_width"),
            )
            tile_height = gr.Slider(
                label="Tile Height",
                minimum=256,
                maximum=2048,
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
                maximum=8,
                step=1,
                value=1,
                elem_id=self.elem_id("tile_batch_size"),
            )

        save_stage1 = gr.Checkbox(
            label="Save Stage 1 output", value=True, elem_id=self.elem_id("save_stage1")
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
    def _validate_run(p, first_pass_denoise: float, final_denoise: float):
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
    ):
        self._validate_run(p, first_pass_denoise, final_denoise)
        original_script_args = p.script_args
        self._enable_multidiffusion(
            p, method, tile_width, tile_height, tile_overlap, tile_batch_size
        )

        processing.fix_seed(p)

        source = images.flatten(p.init_images[0], opts.img2img_background_color)
        explicit_width, explicit_height = self._explicit_dimensions(
            final_width, final_height
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
            int(diffusion_long_edge_cap),
        )
        require_safe_diffusion_size(diffusion_w, diffusion_h)
        needs_final_resize = (diffusion_w, diffusion_h) != (target_w, target_h)
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

        original_batch_size = p.batch_size
        original_n_iter = p.n_iter
        original_do_not_save_samples = p.do_not_save_samples
        original_do_not_save_grid = p.do_not_save_grid
        original_resize_mode = p.resize_mode
        original_width = p.width
        original_height = p.height
        original_denoising_strength = p.denoising_strength
        original_init_images = p.init_images

        try:
            p.batch_size = 1
            p.n_iter = 1
            p.resize_mode = 0
            p.do_not_save_grid = True

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": "Stage 1/2",
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": int(diffusion_long_edge_cap),
                }
            )
            p.width = stage1_w
            p.height = stage1_h
            p.denoising_strength = float(first_pass_denoise)
            p.init_images = [
                source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
            ]
            p.do_not_save_samples = not save_stage1
            state.job = "Krea2 2-Stage Upscale: Stage 1/2"
            state.textinfo = f"Krea2 2-Stage Stage 1/2: {stage1_w}x{stage1_h}"

            stage1 = processing.process_images(p)
            if state.interrupted or state.skipped or state.stopping_generation:
                return stage1
            if not stage1.images:
                raise RuntimeError("Krea2 2-Stage Upscale Stage 1 returned no image.")

            stage1_image = images.flatten(
                stage1.images[0], opts.img2img_background_color
            )
            p.latents_after_sampling.clear()
            p.pixels_after_sampling.clear()
            devices.torch_gc()

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": "Stage 2/2",
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": int(diffusion_long_edge_cap),
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
            p.do_not_save_samples = (
                True if needs_final_resize else original_do_not_save_samples
            )
            p.do_not_save_grid = (
                True if needs_final_resize else original_do_not_save_grid
            )
            state.job = "Krea2 2-Stage Upscale: Stage 2/2"
            state.textinfo = f"Krea2 2-Stage Stage 2/2: {diffusion_w}x{diffusion_h}"

            final = processing.process_images(p)
            if final is None:
                return Processed(p, [], p.seed, "")
            if needs_final_resize:
                diffusion_size_text = f"Size: {diffusion_w}x{diffusion_h}"
                target_size_text = f"Size: {target_w}x{target_h}"
                final.info = final.info.replace(diffusion_size_text, target_size_text)
                final.infotexts = [
                    text.replace(diffusion_size_text, target_size_text)
                    for text in final.infotexts
                ]
                final.width = target_w
                final.height = target_h

                resized_images = []
                for index, image in enumerate(final.images):
                    resized = images.flatten(
                        image, opts.img2img_background_color
                    ).resize((target_w, target_h), Image.Resampling.LANCZOS)
                    if opts.enable_pnginfo and index < len(final.infotexts):
                        resized.info["parameters"] = final.infotexts[index]
                    resized_images.append(resized)
                final.images = resized_images
                p.width = target_w
                p.height = target_h
                if not original_do_not_save_samples:
                    for index, image in enumerate(
                        resized_images[final.index_of_first_image :],
                        start=final.index_of_first_image,
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
