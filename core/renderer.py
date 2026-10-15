"""
Fast mosaic renderer for live preview.
Draws filled circles onto a canvas using OpenCV (C++ backed, fast).
Returns a base64 JPEG data URL suitable for ui.image().
"""

import base64
import numpy as np
import cv2


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)  # OpenCV is BGR


def render_mosaic(
    xs: np.ndarray,
    ys: np.ndarray,
    rs: np.ndarray,
    color_indices: np.ndarray,
    palette_hex: list[str],
    canvas_hex: str,
    W: int,
    H: int,
    max_preview_px: int = 800,
    jpeg_quality: int = 82,
    source_rgb: np.ndarray | None = None,  # (H, W, 3) float32 — if set, shows split view
) -> str:
    """
    Render circles onto canvas, return as 'data:image/jpeg;base64,...' string.

    Uses a preview scale so it's fast even at 2000px working resolution.
    Circles are drawn in array order — later circles paint over earlier ones,
    which matches the airbrush spray order.
    """
    scale = min(1.0, max_preview_px / max(W, H))
    pw = max(1, int(W * scale))
    ph = max(1, int(H * scale))

    canvas_bgr = hex_to_bgr(canvas_hex)
    img = np.full((ph, pw, 3), canvas_bgr, dtype=np.uint8)

    palette_bgr = [hex_to_bgr(h) for h in palette_hex]

    for i in range(len(xs)):
        cx = int(round(xs[i] * scale))
        cy = int(round(ys[i] * scale))
        cr = max(1, int(round(rs[i] * scale)))
        if 0 <= cx < pw and 0 <= cy < ph:
            cv2.circle(img, (cx, cy), cr, palette_bgr[color_indices[i]], -1, cv2.LINE_AA)

    if source_rgb is not None:
        # Side-by-side: source (left) | mosaic (right) with thin divider
        src_small = cv2.resize(
            cv2.cvtColor((source_rgb * 255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            (pw, ph), interpolation=cv2.INTER_AREA,
        )
        divider = np.full((ph, 3, 3), 80, dtype=np.uint8)
        img = np.concatenate([src_small, divider, img], axis=1)

    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')


def render_error_map(
    err_map: np.ndarray,
    max_preview_px: int = 800,
) -> str:
    """
    Render a false-color ΔE error map for debugging.
    Blue = low error, Red = high error.
    """
    H, W = err_map.shape
    scale = min(1.0, max_preview_px / max(W, H))
    pw, ph = max(1, int(W * scale)), max(1, int(H * scale))

    norm = np.clip(err_map / (np.percentile(err_map, 99) + 1e-6), 0, 1)
    norm_small = cv2.resize(norm, (pw, ph), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((norm_small * 255).astype(np.uint8), cv2.COLORMAP_JET)

    ok, buf = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')
