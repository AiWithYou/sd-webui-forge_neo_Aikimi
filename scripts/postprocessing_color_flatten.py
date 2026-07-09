import gradio as gr

from modules import errors, scripts_postprocessing
from modules.color_flatten import FAST_MODE, SUPERPIXEL_MODE, color_flatten_pil
from modules.ui_components import FormRow, InputAccordion


class ScriptPostprocessingColorFlatten(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Flatten"
    order = 900

    def ui(self):
        with InputAccordion(True, label="Color Flatten / 色ムラ補正", elem_id="extras_color_flatten") as enable:
            with FormRow():
                mode = gr.Radio(
                    label="Mode",
                    choices=[FAST_MODE, SUPERPIXEL_MODE],
                    value=FAST_MODE,
                    elem_id="extras_color_flatten_mode",
                )
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Strength",
                    value=0.80,
                    elem_id="extras_color_flatten_strength",
                )
                edge_protect = gr.Checkbox(
                    label="Edge Protect",
                    value=True,
                    elem_id="extras_color_flatten_edge_protect",
                )

            with FormRow():
                mean_shift_sp = gr.Slider(
                    minimum=1,
                    maximum=80,
                    step=1,
                    label="Mean Shift Spatial",
                    value=12,
                    elem_id="extras_color_flatten_mean_shift_sp",
                )
                mean_shift_sr = gr.Slider(
                    minimum=1,
                    maximum=80,
                    step=1,
                    label="Mean Shift Color",
                    value=24,
                    elem_id="extras_color_flatten_mean_shift_sr",
                )

            with FormRow():
                n_segments = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    step=50,
                    label="SLIC Segments",
                    value=1200,
                    elem_id="extras_color_flatten_n_segments",
                )
                compactness = gr.Slider(
                    minimum=1.0,
                    maximum=40.0,
                    step=0.5,
                    label="SLIC Compactness",
                    value=12.0,
                    elem_id="extras_color_flatten_compactness",
                )

        return {
            "enable": enable,
            "mode": mode,
            "strength": strength,
            "edge_protect": edge_protect,
            "mean_shift_sp": mean_shift_sp,
            "mean_shift_sr": mean_shift_sr,
            "n_segments": n_segments,
            "compactness": compactness,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        enable=True,
        mode=FAST_MODE,
        strength=0.80,
        edge_protect=True,
        mean_shift_sp=12,
        mean_shift_sr=24,
        n_segments=1200,
        compactness=12.0,
    ):
        if not enable:
            return

        try:
            strength = float(strength)
            edge_protect = bool(edge_protect)
            mean_shift_sp = int(mean_shift_sp)
            mean_shift_sr = int(mean_shift_sr)
            n_segments = int(n_segments)
            compactness = float(compactness)

            pp.image = color_flatten_pil(
                pp.image,
                mode,
                strength,
                edge_protect,
                mean_shift_sp,
                mean_shift_sr,
                n_segments,
                compactness,
            )
        except Exception:
            errors.report("Error running Color Flatten postprocessing", exc_info=True)
            return

        pp.info["Color Flatten"] = mode
        pp.info["Color Flatten strength"] = strength
        pp.info["Color Flatten edge protect"] = edge_protect
