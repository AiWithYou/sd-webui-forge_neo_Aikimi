from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage import color as skimage_color
from skimage.segmentation import slic

FAST_MODE = "Fast Lab Chroma"
SUPERPIXEL_MODE = "Superpixel Lab Chroma"


def _validate_bgr_image(bgr: np.ndarray) -> None:
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Color Flatten expects a BGR uint8 image with 3 channels")


def _validate_strength(strength: float) -> float:
    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("Color Flatten strength must be between 0.0 and 1.0")

    return strength


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")

    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be greater than 0")

    return value


def _edge_strength_map(l_channel: np.ndarray, strength: float) -> np.ndarray:
    l_float = l_channel.astype(np.float32)
    grad_x = cv2.Sobel(l_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_float, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    scale = float(np.percentile(grad, 95))

    if scale <= 1e-6:
        return np.full(l_channel.shape, strength, dtype=np.float32)

    grad = np.clip(grad / scale, 0.0, 1.0)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.0)
    return (strength * (1.0 - 0.85 * grad)).astype(np.float32)


def _light_bilateral_u8(channel: np.ndarray, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(channel, d=5, sigmaColor=10.0, sigmaSpace=max(3.0, sigma_space * 0.5))


def _light_bilateral_float(channel: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(channel.astype(np.float32), d=5, sigmaColor=4.0, sigmaSpace=3.0)


def chroma_flatten_fast_bgr(
    bgr: np.ndarray,
    strength: float,
    edge_protect: bool,
    mean_shift_sp: int,
    mean_shift_sr: int,
) -> np.ndarray:
    _validate_bgr_image(bgr)
    strength = _validate_strength(strength)
    mean_shift_sp = _validate_positive_int("Mean Shift Spatial", mean_shift_sp)
    mean_shift_sr = _validate_positive_int("Mean Shift Color", mean_shift_sr)

    if strength == 0.0:
        return bgr.copy()

    source_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    shifted = cv2.pyrMeanShiftFiltering(bgr, sp=mean_shift_sp, sr=mean_shift_sr)
    shifted_lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)

    target_a = _light_bilateral_u8(shifted_lab[:, :, 1], mean_shift_sp)
    target_b = _light_bilateral_u8(shifted_lab[:, :, 2], mean_shift_sp)
    target_ab = np.stack([target_a, target_b], axis=2).astype(np.float32)
    source_ab = source_lab[:, :, 1:3].astype(np.float32)

    if edge_protect:
        blend = _edge_strength_map(source_lab[:, :, 0], strength)[:, :, None]
    else:
        blend = np.float32(strength)

    result_ab = source_ab + (target_ab - source_ab) * blend
    result_lab = source_lab.copy()
    result_lab[:, :, 1:3] = np.clip(np.rint(result_ab), 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def superpixel_chroma_flatten_bgr(
    bgr: np.ndarray,
    strength: float,
    edge_protect: bool,
    n_segments: int,
    compactness: float,
) -> np.ndarray:
    _validate_bgr_image(bgr)
    strength = _validate_strength(strength)
    n_segments = _validate_positive_int("SLIC Segments", n_segments)
    compactness = _validate_positive_float("SLIC Compactness", compactness)

    if strength == 0.0:
        return bgr.copy()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    labels = slic(
        rgb_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )
    lab = skimage_color.rgb2lab(rgb_float).astype(np.float32)

    labels_flat = labels.reshape(-1)
    order = np.argsort(labels_flat, kind="stable")
    ordered_labels = labels_flat[order]
    starts = np.empty(ordered_labels.shape[0], dtype=bool)
    starts[0] = True
    starts[1:] = ordered_labels[1:] != ordered_labels[:-1]
    starts = np.flatnonzero(starts)
    ends = np.concatenate([starts[1:], np.array([ordered_labels.shape[0]])])

    source_a = lab[:, :, 1].reshape(-1)
    source_b = lab[:, :, 2].reshape(-1)
    ordered_a = source_a[order]
    ordered_b = source_b[order]
    target_a = np.empty_like(source_a)
    target_b = np.empty_like(source_b)

    for start, end in zip(starts, ends):
        segment_indices = order[start:end]
        target_a[segment_indices] = np.median(ordered_a[start:end])
        target_b[segment_indices] = np.median(ordered_b[start:end])

    target_a = target_a.reshape(labels.shape)
    target_b = target_b.reshape(labels.shape)
    target_a = _light_bilateral_float(target_a)
    target_b = _light_bilateral_float(target_b)

    if edge_protect:
        blend = _edge_strength_map(lab[:, :, 0], strength)
    else:
        blend = np.float32(strength)

    result_lab = lab.copy()
    result_lab[:, :, 1] = lab[:, :, 1] + (target_a - lab[:, :, 1]) * blend
    result_lab[:, :, 2] = lab[:, :, 2] + (target_b - lab[:, :, 2]) * blend

    result_rgb = np.clip(skimage_color.lab2rgb(result_lab), 0.0, 1.0)
    result_rgb = np.rint(result_rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def color_flatten_pil(
    image: Image.Image,
    mode: str,
    strength: float,
    edge_protect: bool,
    mean_shift_sp: int,
    mean_shift_sr: int,
    n_segments: int,
    compactness: float,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Color Flatten expects a PIL.Image.Image input")

    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb_image = image.convert("RGB")
    elif image.mode == "RGB":
        alpha = None
        rgb_image = image
    else:
        raise ValueError("Color Flatten supports RGB and RGBA images")

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    if mode == FAST_MODE:
        result_bgr = chroma_flatten_fast_bgr(bgr, strength, edge_protect, mean_shift_sp, mean_shift_sr)
    elif mode == SUPERPIXEL_MODE:
        result_bgr = superpixel_chroma_flatten_bgr(bgr, strength, edge_protect, n_segments, compactness)
    else:
        raise ValueError(f"Unknown Color Flatten mode: {mode}")

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result_rgb, "RGB")

    if alpha is not None:
        result.putalpha(alpha)

    return result
