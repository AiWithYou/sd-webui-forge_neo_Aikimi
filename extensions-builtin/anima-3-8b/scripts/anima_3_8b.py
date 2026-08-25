from __future__ import annotations

import gradio as gr

from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion

from anima3b.files import adapters, standard_anima_loras
from anima3b.runtime import Anima3BRuntime
from anima3b.standard_lora import (
    NO_STANDARD_LORA,
    apply_standard_lora_selections,
)


STANDARD_LORA_SLOT_COUNT = 4


def _standard_lora_selection_pairs(
    standard_lora: str | None,
    standard_lora_strength: float,
    additional_values: tuple,
) -> list[tuple[str | None, float]]:
    values = (standard_lora, standard_lora_strength, *additional_values)
    if len(values) % 2:
        raise ValueError("Anima 3.8B LoRA controls must be name/weight pairs.")
    return [
        (values[index], values[index + 1])
        for index in range(0, len(values), 2)
    ]


class Anima3BScript(scripts.Script):
    sorting_priority = 260209301

    def __init__(self):
        super().__init__()
        self.runtime = Anima3BRuntime()

    def title(self):
        return "Anima 3.8B"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        choices = list(adapters())
        if not choices:
            choices = ["Anima-3.8B-expanded_adapter.safetensors"]
        lora_choices = [NO_STANDARD_LORA, *standard_anima_loras()]
        with InputAccordion(False, label="Anima 3.8B (Qwen3.5)") as enabled:
            adapter = gr.Dropdown(
                label="Adapter",
                choices=choices,
                value=choices[0],
                info="Detected by checkpoint metadata in models/text_encoder.",
            )
            strength = gr.Slider(
                label="Adapter strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="1.0 is trained strength; 0.0 is native Anima.",
            )
            negative = gr.Checkbox(
                label="Use adapter on negative prompt",
                value=False,
                info="Off keeps negatives on the native Anima encoder.",
            )
            negative_strength = gr.Slider(
                label="Negative adapter strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                visible=False,
                info="Adapter strength applied to the negative prompt.",
            )
            negative.change(
                fn=lambda value: gr.update(visible=bool(value)),
                inputs=[negative],
                outputs=[negative_strength],
                show_progress=False,
                queue=False,
            )
            standard_lora = gr.Dropdown(
                label="Standard Anima LoRA",
                choices=lora_choices,
                value=NO_STANDARD_LORA,
                info=(
                    "Complete 28/40/52-block LoRAs from models/Lora. "
                    "28/40-block layouts are expanded to 52 blocks automatically. "
                    "Up to four LoRAs can be combined with separate weights."
                ),
            )
            refresh_standard_loras = gr.Button(
                value="Refresh standard Anima LoRAs",
                variant="secondary",
            )
            standard_lora_strength = gr.Slider(
                label="Standard LoRA strength",
                minimum=-2.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                visible=False,
                info=(
                    "Weight for this LoRA. Prompt <lora:...> tags and the LoRA "
                    "tab remain available for additional combinations."
                ),
            )
            standard_lora_components = [(standard_lora, standard_lora_strength)]
            with gr.Accordion("Additional Standard Anima LoRAs", open=False):
                for slot in range(2, STANDARD_LORA_SLOT_COUNT + 1):
                    with gr.Row():
                        additional_lora = gr.Dropdown(
                            label=f"Standard Anima LoRA {slot}",
                            choices=lora_choices,
                            value=NO_STANDARD_LORA,
                        )
                        additional_lora_strength = gr.Slider(
                            label=f"LoRA {slot} strength",
                            minimum=-2.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            visible=False,
                        )
                    standard_lora_components.append(
                        (additional_lora, additional_lora_strength)
                    )

            for lora_component, strength_component in standard_lora_components:
                lora_component.change(
                    fn=lambda value: gr.update(
                        value=1.0,
                        visible=value not in {None, "", NO_STANDARD_LORA},
                    ),
                    inputs=[lora_component],
                    outputs=[strength_component],
                    show_progress=False,
                    queue=False,
                )

            refresh_outputs = [
                component
                for pair in standard_lora_components
                for component in pair
            ]

            def refresh_standard_lora_slots():
                refreshed_choices = [NO_STANDARD_LORA, *standard_anima_loras()]
                return [
                    update
                    for _ in standard_lora_components
                    for update in (
                        gr.update(
                            choices=refreshed_choices,
                            value=NO_STANDARD_LORA,
                        ),
                        gr.update(value=1.0, visible=False),
                    )
                ]

            refresh_standard_loras.click(
                fn=refresh_standard_lora_slots,
                inputs=[],
                outputs=refresh_outputs,
                show_progress=False,
                queue=False,
            )
            offload_encoders = gr.Checkbox(
                label="Low VRAM: offload text encoders before sampling",
                value=False,
                info=(
                    "Releases about 5 GiB after conditioning. A changed prompt "
                    "must load the encoders again; seed-only repeats use the cache."
                ),
            )
        additional_standard_lora_components = [
            component
            for pair in standard_lora_components[1:]
            for component in pair
        ]
        return [
            enabled,
            adapter,
            strength,
            negative,
            negative_strength,
            offload_encoders,
            standard_lora,
            standard_lora_strength,
            *additional_standard_lora_components,
        ]

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        adapter: str,
        strength: float,
        negative: bool = False,
        negative_strength: float = 1.0,
        offload_encoders: bool = False,
        standard_lora: str = NO_STANDARD_LORA,
        standard_lora_strength: float = 1.0,
        *additional_standard_lora_values,
        **kwargs,
    ):
        del adapter, strength, negative, negative_strength, offload_encoders, kwargs
        if not enabled:
            return
        selections = _standard_lora_selection_pairs(
            standard_lora,
            standard_lora_strength,
            additional_standard_lora_values,
        )
        apply_standard_lora_selections(
            p,
            selections,
            standard_anima_loras(),
        )

    def process_batch(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        adapter: str,
        strength: float,
        negative: bool = False,
        negative_strength: float = 1.0,
        offload_encoders: bool = False,
        standard_lora: str = NO_STANDARD_LORA,
        standard_lora_strength: float = 1.0,
        *additional_standard_lora_values,
        **kwargs,
    ):
        del standard_lora, standard_lora_strength, additional_standard_lora_values
        if not enabled:
            return
        self.runtime.install(
            p,
            adapter,
            strength,
            float(negative_strength) if negative else None,
        )

    def before_process(self, p: StableDiffusionProcessing, *args):
        self.runtime.restore_model(p.sd_model)

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        adapter: str,
        strength: float,
        negative: bool = False,
        negative_strength: float = 1.0,
        offload_encoders: bool = False,
        standard_lora: str = NO_STANDARD_LORA,
        standard_lora_strength: float = 1.0,
        *additional_standard_lora_values,
        **kwargs,
    ):
        del standard_lora, standard_lora_strength, additional_standard_lora_values
        if not enabled:
            return
        if offload_encoders:
            released = self.runtime.offload_text_encoders(p.sd_model)
            released = max(
                int(getattr(p, "_anima3b_encoder_vram_released", 0)),
                released,
            )
            p._anima3b_encoder_vram_released = released
            p.extra_generation_params["Anima 3.8B encoder offload"] = True
            p.extra_generation_params["Anima 3.8B encoder VRAM released"] = (
                f"{released / (1024**3):.2f} GiB"
            )

    def postprocess(self, p, processed, *args):
        self.runtime.restore(p)

    def on_process_cleanup(self, p, *args):
        self.runtime.restore(p)
