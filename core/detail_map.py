"""
Detail / complexity field from image structure tensor.
High detail → smaller circles. Low detail → larger circles.
"""

import numpy as np
import cv2


def compute_detail_map(
    rgb: np.ndarray,
    blur_sigma: float = 2.0,
    detail_boost: float = 1.0,
) -> np.ndarray:
    """
    Returns detail map (H, W) float32, range [0, 1].
    1.0 = maximum detail (use smallest circles).
    0.0 = smooth region (use largest circles).
    """
    # Structure tensor over LUMINANCE *and* COLOUR (a/b) channels.  A grayscale-only
    # detail map misses smooth colour gradients (hue shifting at similar brightness), so
    # those areas got the largest dots and couldn't dither — reading as solid blocks.
    # Including colour gradients gives colour-transition regions finer dots → finer stipple.
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    a_ch = lab[:, :, 1] / 255.0
    b_ch = lab[:, :, 2] / 255.0
    col_w = 0.6   # weight of colour gradients relative to luminance

    Jxx = Jxy = Jyy = 0.0
    for chan, wt in ((gray, 1.0), (a_ch, col_w), (b_ch, col_w)):
        cgx = cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=3)
        cgy = cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=3)
        Jxx = Jxx + wt * cgx * cgx
        Jxy = Jxy + wt * cgx * cgy
        Jyy = Jyy + wt * cgy * cgy

    ksize = max(3, int(blur_sigma * 4) | 1)
    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), blur_sigma)
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), blur_sigma)
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), blur_sigma)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    discriminant = np.sqrt(np.clip((trace ** 2) / 4 - det, 0, None))
    lambda_max = trace / 2 + discriminant

    p99 = np.percentile(lambda_max, 99)
    detail = np.clip(lambda_max / p99 if p99 > 0 else lambda_max, 0.0, 1.0)
    detail = np.power(detail, 1.0 / max(detail_boost, 0.1))

    return detail.astype(np.float32)


def build_radius_field(
    detail: np.ndarray,
    weight: np.ndarray,
    contrast: np.ndarray,
    r_min: float,
    r_max: float,
    luminosity: np.ndarray | None = None,
    lum_influence: float = 0.5,
    canvas_lightness: float = 100.0,
) -> np.ndarray:
    """
    Per-pixel target radius in working pixels.

    detail:           (H, W) 0–1, 1 = fine detail → smaller circles
    luminosity:       (H, W) L channel 0–100 from LAB.
    lum_influence:    0 = detail-only sizing, 1 = luminosity-only sizing.
    canvas_lightness: L value of canvas (0=black, 100=white). On a light canvas
                      bright pixels → small circles (canvas fills in light).
                      On a dark canvas the direction inverts: bright pixels need
                      large circles because the canvas contributes nothing there.
    """
    # Detail term: 0 = detailed (small), 1 = smooth (large)
    detail_size = 1.0 - detail

    if luminosity is not None and lum_influence > 0.0:
        lum_norm = np.clip(luminosity / 100.0, 0.0, 1.0)
        # On light canvas: bright → small (norm toward 0), dark → large (norm toward 1)
        # On dark canvas:  bright → large (norm toward 1), dark → small (norm toward 0)
        # Linearly interpolate the direction based on canvas lightness.
        canvas_t = np.clip(canvas_lightness / 100.0, 0.0, 1.0)  # 1=white, 0=black
        directed = canvas_t * (1.0 - lum_norm) + (1.0 - canvas_t) * lum_norm
        sigmoid_k = 8.0
        lum_size  = 1.0 / (1.0 + np.exp(-sigmoid_k * (directed - 0.5)))
        size_factor = ((1.0 - lum_influence) * detail_size
                       + lum_influence * lum_size).astype(np.float32)
    else:
        size_factor = detail_size

    # Contrast modulates the effective r_max: low contrast compresses range downward
    contrast_clamped = np.clip(contrast, 0.15, 1.0)
    effective_r_max = r_min + (r_max - r_min) * contrast_clamped

    r = r_min + (effective_r_max - r_min) * size_factor

    return np.clip(r, r_min, r_max).astype(np.float32)
