"""
Per-paint-color weight maps with canvas background compensation.

Design:
- Soft perceptual assignment in CIELAB. Every pixel gets assigned to
  every color with a soft weight. The dominant color wins territory.
- Canvas compensation shifts the effective target so the mix of
  paint + canvas gaps approximates the original pixel color.
- The contrast_map is returned for radius modulation only — it does NOT
  gate whether a circle is placed. Every pixel gets a circle.
"""

import numpy as np
from .image_processor import ImageData, hex_to_lab, rgb_to_lab, lab_distance


def build_weight_maps(
    img: ImageData,
    palette_hex: list[str],
    canvas_hex: str,
    softness: float = 30.0,
    canvas_comp: float = 0.35,
    chroma_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        weight_maps: (N, H, W) float32 — soft color assignment weights.
        contrast_map: (H, W) float32 [0–1] — perceptual contrast vs canvas,
                      used only to modulate circle size, never to skip circles.
        comp_lab:    (H, W, 3) float32 — the canvas-compensated target LAB the paint
                      should aim for (so downstream recolouring honours compensation).
    """
    n_colors = len(palette_hex)
    H, W = img.lab.shape[:2]

    palette_lab = np.array([hex_to_lab(h) for h in palette_hex])  # (N, 3)
    canvas_lab = hex_to_lab(canvas_hex)
    canvas_rgb = _hex_to_rgb(canvas_hex)  # (3,)

    # Canvas COLOUR compensation.  On the finished piece the paint dots sit on a
    # coloured canvas that shows through the gaps.  A viewer perceives, per region:
    #     perceived ≈ (1-s)·paint + s·canvas          (s = canvas show-through)
    # To reproduce the original image we must aim the paint at the compensated
    # target that cancels the canvas tint:
    #     paint_target = (image - s·canvas) / (1 - s)
    # canvas_comp (s) is the user's estimate of how much canvas shows through.
    #   s = 0   → no compensation, dots use the raw image colours.
    #   s ↑     → dots shift further to offset a non-neutral canvas colour.
    s = float(np.clip(canvas_comp, 0.0, 0.8))
    comp_rgb = (img.rgb_float - s * canvas_rgb) / max(1.0 - s, 0.2)
    comp_rgb = np.clip(comp_rgb, 0.0, 1.0)
    comp_lab = rgb_to_lab(comp_rgb)  # (H, W, 3)

    # Soft assignment: distance from compensated pixel to each paint color.
    # chroma_weight scales L* relative to a*,b* — lower = colors compete better
    # in shadow areas instead of black monopolizing all dark pixels.
    l_scale = 1.0 - chroma_weight * 0.85   # 0 chroma_weight → l_scale=1, 1 → l_scale=0.15
    dists = np.stack([
        _lab_distance_weighted(comp_lab, palette_lab[k], l_scale)
        for k in range(n_colors)
    ], axis=0)  # (N, H, W)

    # RBF weights: closer color = higher weight
    sigma = float(softness)
    weights = np.exp(-(dists ** 2) / (sigma ** 2))

    # Normalize across color axis
    weight_sum = weights.sum(axis=0, keepdims=True).clip(min=1e-8)
    weights = weights / weight_sum

    # Contrast map: how different is each pixel from the canvas in the ORIGINAL image?
    # High contrast → bigger circles OK. Low contrast → smaller circles (but still placed).
    canvas_dist = lab_distance(img.lab, canvas_lab)  # (H, W)
    contrast_map = np.clip(canvas_dist / 70.0, 0.0, 1.0).astype(np.float32)

    return weights.astype(np.float32), contrast_map, comp_lab.astype(np.float32)


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip('#')
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _lab_distance_weighted(lab_map: np.ndarray, palette_entry: np.ndarray, l_scale: float) -> np.ndarray:
    """LAB distance with reduced L* weight so chroma matters more in dark areas."""
    dl = (lab_map[:, :, 0] - palette_entry[0]) * l_scale
    da =  lab_map[:, :, 1] - palette_entry[1]
    db =  lab_map[:, :, 2] - palette_entry[2]
    return np.sqrt(dl * dl + da * da + db * db)
