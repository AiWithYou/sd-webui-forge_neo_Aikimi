from __future__ import annotations

from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing
from modules.ui_components import FormRow, InputAccordion


class MuraParams:
    def __init__(
        self,
        blur_sigma: float = 18.0,
        edge_percentile: float = 96.0,
        edge_dilate: int = 5,
        heat_max_delta_e: float = 12.0,
        overlay_alpha: float = 0.65,
        alpha_threshold: int = 8,
        ignore_near_white_bg: bool = False,
        white_bg_luma_threshold: int = 245,
        white_bg_chroma_threshold: float = 5.0,
    ):
        self.blur_sigma = blur_sigma
        self.edge_percentile = edge_percentile
        self.edge_dilate = edge_dilate
        self.heat_max_delta_e = heat_max_delta_e
        self.overlay_alpha = overlay_alpha
        self.alpha_threshold = alpha_threshold
        self.ignore_near_white_bg = ignore_near_white_bg
        self.white_bg_luma_threshold = white_bg_luma_threshold
        self.white_bg_chroma_threshold = white_bg_chroma_threshold


class MuraMetrics:
    def __init__(
        self,
        valid_area_pct: float,
        mean_delta_e: float,
        median_delta_e: float,
        p90_delta_e: float,
        p95_delta_e: float,
        p99_delta_e: float,
        max_delta_e: float,
        area_delta_e_gt_2_pct: float,
        area_delta_e_gt_5_pct: float,
        area_delta_e_gt_10_pct: float,
        rough_judgement: str,
    ):
        self.valid_area_pct = valid_area_pct
        self.mean_delta_e = mean_delta_e
        self.median_delta_e = median_delta_e
        self.p90_delta_e = p90_delta_e
        self.p95_delta_e = p95_delta_e
        self.p99_delta_e = p99_delta_e
        self.max_delta_e = max_delta_e
        self.area_delta_e_gt_2_pct = area_delta_e_gt_2_pct
        self.area_delta_e_gt_5_pct = area_delta_e_gt_5_pct
        self.area_delta_e_gt_10_pct = area_delta_e_gt_10_pct
        self.rough_judgement = rough_judgement


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to Lab float: L*=0..100, a*/b* roughly -128..127."""
    lab8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.empty_like(lab8, dtype=np.float32)
    lab[..., 0] = lab8[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab8[..., 1] - 128.0
    lab[..., 2] = lab8[..., 2] - 128.0
    return lab


def make_valid_mask(rgba: np.ndarray, lab: np.ndarray, params: MuraParams) -> np.ndarray:
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    valid = alpha > int(params.alpha_threshold)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_edge = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.5, sigmaY=1.5)
    gx = cv2.Sobel(gray_edge, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_edge, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    edge_percentile = float(np.clip(params.edge_percentile, 0.0, 100.0))
    if edge_percentile < 100.0:
        edge_source = grad[valid] if np.any(valid) else grad
        edge_th = np.percentile(edge_source, edge_percentile)
        edge_th = max(float(edge_th), 0.08)
        edges = grad > edge_th
    else:
        edges = np.zeros_like(valid, dtype=bool)

    dilate = int(max(0, params.edge_dilate))
    if dilate > 0:
        k = 2 * dilate + 1
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1).astype(bool)

    valid &= ~edges

    if params.ignore_near_white_bg:
        luma = gray * 255.0
        chroma = np.sqrt(lab[..., 1] * lab[..., 1] + lab[..., 2] * lab[..., 2])
        near_white = (luma >= float(params.white_bg_luma_threshold)) & (
            chroma <= float(params.white_bg_chroma_threshold)
        )
        valid &= ~near_white

    return valid


def compute_mura(image: Image.Image, params: Optional[MuraParams] = None):
    """
    Return original RGBA, heatmap RGBA, overlay RGBA, delta-E map, metrics.
    The score is CIE76-like Delta E in Lab space against a locally blurred
    reference image. Structural edges are masked out so outlines are not counted
    as color mura.
    """
    params = params or MuraParams()

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    lab = rgb_to_lab_float(rgb)

    sigma = max(0.1, float(params.blur_sigma))
    lab_ref = cv2.GaussianBlur(lab, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    delta_e = np.linalg.norm(lab - lab_ref, axis=2).astype(np.float32)

    valid = make_valid_mask(rgba, lab, params)

    if np.any(valid):
        d = delta_e[valid]
        valid_area_pct = float(np.mean(valid) * 100.0)
        mean_de = float(np.mean(d))
        median_de = float(np.median(d))
        p90 = float(np.percentile(d, 90))
        p95 = float(np.percentile(d, 95))
        p99 = float(np.percentile(d, 99))
        max_de = float(np.max(d))
        gt2 = float(np.mean(d > 2.0) * 100.0)
        gt5 = float(np.mean(d > 5.0) * 100.0)
        gt10 = float(np.mean(d > 10.0) * 100.0)
    else:
        valid_area_pct = 0.0
        mean_de = median_de = p90 = p95 = p99 = max_de = 0.0
        gt2 = gt5 = gt10 = 0.0

    if p95 <= 4.0 and gt5 <= 1.0:
        judgement = "OK: 色むらは軽微"
    elif p95 <= 8.0 and gt5 <= 5.0:
        judgement = "CHECK: 軽い色むらあり"
    else:
        judgement = "NG: 目立つ色むら候補あり"

    metrics = MuraMetrics(
        valid_area_pct=valid_area_pct,
        mean_delta_e=mean_de,
        median_delta_e=median_de,
        p90_delta_e=p90,
        p95_delta_e=p95,
        p99_delta_e=p99,
        max_delta_e=max_de,
        area_delta_e_gt_2_pct=gt2,
        area_delta_e_gt_5_pct=gt5,
        area_delta_e_gt_10_pct=gt10,
        rough_judgement=judgement,
    )

    heat_max = max(1.0, float(params.heat_max_delta_e))
    norm = np.clip(delta_e / heat_max * 255.0, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(norm, cmap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    heat_rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    heat_rgba[..., :3] = heat_rgb
    heat_rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    overlay = rgba.copy().astype(np.float32)
    alpha = float(np.clip(params.overlay_alpha, 0.0, 1.0))
    overlay_rgb = overlay[..., :3]
    overlay_rgb[valid] = overlay_rgb[valid] * (1.0 - alpha) + heat_rgb[valid].astype(np.float32) * alpha
    overlay[..., :3] = overlay_rgb
    overlay_rgba = np.clip(overlay, 0, 255).astype(np.uint8)

    delta_e_nan = delta_e.copy()
    delta_e_nan[~valid] = np.nan

    return rgba, heat_rgba, overlay_rgba, delta_e_nan, metrics


def format_metrics(metrics: MuraMetrics) -> str:
    return (
        f"{metrics.rough_judgement} / "
        f"p95 ΔE={metrics.p95_delta_e:.2f}, "
        f">5 ΔE area={metrics.area_delta_e_gt_5_pct:.2f}%, "
        f">10 ΔE area={metrics.area_delta_e_gt_10_pct:.2f}%, "
        f"valid={metrics.valid_area_pct:.2f}%"
    )


def add_metrics_to_info(info: dict, metrics: MuraMetrics) -> None:
    info["Color mura judgement"] = metrics.rough_judgement
    info["Color mura valid area %"] = f"{metrics.valid_area_pct:.3f}"
    info["Color mura mean Delta E"] = f"{metrics.mean_delta_e:.3f}"
    info["Color mura median Delta E"] = f"{metrics.median_delta_e:.3f}"
    info["Color mura p90 Delta E"] = f"{metrics.p90_delta_e:.3f}"
    info["Color mura p95 Delta E"] = f"{metrics.p95_delta_e:.3f}"
    info["Color mura p99 Delta E"] = f"{metrics.p99_delta_e:.3f}"
    info["Color mura max Delta E"] = f"{metrics.max_delta_e:.3f}"
    info["Color mura area >2 Delta E %"] = f"{metrics.area_delta_e_gt_2_pct:.3f}"
    info["Color mura area >5 Delta E %"] = f"{metrics.area_delta_e_gt_5_pct:.3f}"
    info["Color mura area >10 Delta E %"] = f"{metrics.area_delta_e_gt_10_pct:.3f}"


class ScriptPostprocessingColorMuraChecker(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Mura Checker"
    order = 1100

    def ui(self):
        with InputAccordion(True, label="色むら確認 / Color Mura Checker", elem_id="extras_color_mura_checker") as enabled:
            with FormRow():
                output_modes = gr.CheckboxGroup(
                    choices=["Overlay", "Heatmap"],
                    value=["Overlay", "Heatmap"],
                    label="出力画像",
                    elem_id="extras_color_mura_outputs",
                )
                add_metrics = gr.Checkbox(
                    label="結果をPNG infoへ追加",
                    value=True,
                    elem_id="extras_color_mura_add_metrics",
                )
                ignore_near_white_bg = gr.Checkbox(
                    label="白背景を無視",
                    value=False,
                    elem_id="extras_color_mura_ignore_white_bg",
                )

            with FormRow():
                blur_sigma = gr.Slider(
                    minimum=3,
                    maximum=80,
                    step=1,
                    label="基準色ぼかし sigma",
                    value=18,
                    elem_id="extras_color_mura_blur_sigma",
                )
                edge_percentile = gr.Slider(
                    minimum=50,
                    maximum=100,
                    step=1,
                    label="輪郭除外 percentile（100で無効）",
                    value=96,
                    elem_id="extras_color_mura_edge_percentile",
                )
                edge_dilate = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1,
                    label="輪郭除外の太さ",
                    value=5,
                    elem_id="extras_color_mura_edge_dilate",
                )

            with FormRow():
                heat_max_delta_e = gr.Slider(
                    minimum=3,
                    maximum=30,
                    step=1,
                    label="ヒートマップ最大 ΔE",
                    value=12,
                    elem_id="extras_color_mura_heat_max_delta_e",
                )
                overlay_alpha = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label="Overlay alpha",
                    value=0.65,
                    elem_id="extras_color_mura_overlay_alpha",
                )
                alpha_threshold = gr.Slider(
                    minimum=0,
                    maximum=255,
                    step=1,
                    label="透明除外 alpha threshold",
                    value=8,
                    elem_id="extras_color_mura_alpha_threshold",
                )

        return {
            "color_mura_enabled": enabled,
            "color_mura_outputs": output_modes,
            "color_mura_add_metrics": add_metrics,
            "color_mura_ignore_near_white_bg": ignore_near_white_bg,
            "color_mura_blur_sigma": blur_sigma,
            "color_mura_edge_percentile": edge_percentile,
            "color_mura_edge_dilate": edge_dilate,
            "color_mura_heat_max_delta_e": heat_max_delta_e,
            "color_mura_overlay_alpha": overlay_alpha,
            "color_mura_alpha_threshold": alpha_threshold,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        color_mura_enabled=True,
        color_mura_outputs=("Overlay", "Heatmap"),
        color_mura_add_metrics=True,
        color_mura_ignore_near_white_bg=False,
        color_mura_blur_sigma=18,
        color_mura_edge_percentile=96,
        color_mura_edge_dilate=5,
        color_mura_heat_max_delta_e=12,
        color_mura_overlay_alpha=0.65,
        color_mura_alpha_threshold=8,
    ):
        if not color_mura_enabled:
            return

        outputs = set(color_mura_outputs or [])
        params = MuraParams(
            blur_sigma=float(color_mura_blur_sigma),
            edge_percentile=float(color_mura_edge_percentile),
            edge_dilate=int(color_mura_edge_dilate),
            heat_max_delta_e=float(color_mura_heat_max_delta_e),
            overlay_alpha=float(color_mura_overlay_alpha),
            alpha_threshold=int(color_mura_alpha_threshold),
            ignore_near_white_bg=bool(color_mura_ignore_near_white_bg),
        )

        _, heat_rgba, overlay_rgba, _, metrics = compute_mura(pp.image, params)

        if color_mura_add_metrics:
            add_metrics_to_info(pp.info, metrics)
            pp.info["Color mura summary"] = format_metrics(metrics)

        if "Overlay" in outputs:
            overlay = Image.fromarray(overlay_rgba, mode="RGBA")
            overlay_pp = pp.create_copy(overlay, nametags=["mura-overlay"], disable_processing=True)
            add_metrics_to_info(overlay_pp.info, metrics)
            overlay_pp.info["Color mura view"] = "Overlay"
            pp.extra_images.append(overlay_pp)

        if "Heatmap" in outputs:
            heatmap = Image.fromarray(heat_rgba, mode="RGBA")
            heat_pp = pp.create_copy(heatmap, nametags=["mura-heatmap"], disable_processing=True)
            add_metrics_to_info(heat_pp.info, metrics)
            heat_pp.info["Color mura view"] = "Heatmap"
            pp.extra_images.append(heat_pp)
