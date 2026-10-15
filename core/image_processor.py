"""
Image loading and preprocessing.
Handles large (50MP+) images by working at a scaled-down resolution
while preserving physical coordinate mapping for final SVG output.
"""

import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass

# Registers HEIC/HEIF support with PIL's Image.open — without this, Pillow can't
# identify these files at all (raises UnidentifiedImageError immediately, not a
# thumbnail/quality issue). Import at module load so every Image.open() call in
# the app (not just this file) gets HEIC support transparently.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


@dataclass
class ImageData:
    rgb_float: np.ndarray          # (H, W, 3) float32, 0–1, sRGB
    lab: np.ndarray                # (H, W, 3) float32, CIELAB
    working_size: tuple[int, int]  # (W, H) of working image
    original_size: tuple[int, int] # (W, H) of original
    scale_factor: float
    source_path: str


def load_image(path: str, max_working_px: int = 2000) -> ImageData:
    img = Image.open(path)
    if img.mode not in ('RGB',):
        img = img.convert('RGB')

    original_size = img.size  # (W, H)
    longest = max(original_size)
    scale_factor = longest / max_working_px

    if scale_factor != 1.0:
        ww = int(round(original_size[0] / scale_factor))
        wh = int(round(original_size[1] / scale_factor))
        # Downscale: LANCZOS. Upscale: bicubic + post-blur to kill ringing.
        if scale_factor > 1.0:
            img = img.resize((ww, wh), Image.LANCZOS)
        else:
            img = img.resize((ww, wh), Image.BICUBIC)

    rgb_float = np.array(img, dtype=np.float32) / 255.0

    # When upscaling, interpolation creates periodic ringing that SIFT and the
    # detail map pick up as fake grid features — that periodic signal reads
    # as genuine (if faint) detail, so circles converge onto it exactly like
    # they would real high-frequency content, producing the source of a grid
    # pattern in what should be a smooth/flat region. The previous formula
    # (max(0.5, 0.4/scale_factor)) was too weak for a common case — a modest
    # upscale (say 1.3-1.5x) landed barely above the 0.5 floor, not nearly
    # enough to fully kill the ringing. Both the floor and the scaling are
    # stronger here.
    if scale_factor < 1.0:
        sigma = max(0.8, 0.9 / scale_factor)   # larger upscale → more blur needed
        ksize = int(sigma * 4) | 1              # odd kernel
        rgb_float = cv2.GaussianBlur(rgb_float, (ksize, ksize), sigma)

    lab = rgb_to_lab(rgb_float)

    return ImageData(
        rgb_float=rgb_float,
        lab=lab,
        working_size=(img.width, img.height),
        original_size=original_size,
        scale_factor=scale_factor,
        source_path=str(path),
    )


def load_image_array(rgb_uint8: np.ndarray, max_working_px: int = 2000) -> ImageData:
    """Build ImageData from an in-memory RGB uint8 array (H, W, 3) — e.g. a video
    frame — mirroring load_image's downscale/blur/LAB processing without touching disk."""
    from PIL import Image as _Image
    img = _Image.fromarray(rgb_uint8, 'RGB')
    original_size = img.size  # (W, H)
    longest = max(original_size)
    scale_factor = longest / max_working_px
    if scale_factor != 1.0:
        ww = int(round(original_size[0] / scale_factor))
        wh = int(round(original_size[1] / scale_factor))
        img = img.resize((ww, wh), Image.LANCZOS if scale_factor > 1.0 else Image.BICUBIC)

    rgb_float = np.array(img, dtype=np.float32) / 255.0
    # Same anti-ringing mitigation as load_image() — see the comment there.
    if scale_factor < 1.0:
        sigma = max(0.8, 0.9 / scale_factor)
        ksize = int(sigma * 4) | 1
        rgb_float = cv2.GaussianBlur(rgb_float, (ksize, ksize), sigma)

    lab = rgb_to_lab(rgb_float)
    return ImageData(
        rgb_float=rgb_float,
        lab=lab,
        working_size=(img.width, img.height),
        original_size=original_size,
        scale_factor=scale_factor,
        source_path='<frame>',
    )


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB float (0–1) → CIELAB. L in [0,100], a/b in [-128,127]."""
    rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def hex_to_lab(hex_color: str) -> np.ndarray:
    """#RRGGBB → CIELAB (3,)."""
    h = hex_color.lstrip('#')
    rgb = np.array([[[int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]]], dtype=np.float32)
    return rgb_to_lab(rgb)[0, 0]


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip('#')
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def lab_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))
