"""
Registration marks: small alignment targets drawn at all 4 canvas corners so
separately cut/printed layers (stencils, multi-pass plots, screen layers) can
be physically aligned against each other and against the source artwork.

Geometry is unit-agnostic — callers pass W/H/dist/size already converted into
whatever unit that export format works in (SVG px, raster px, or mm), and get
back primitives in the same unit.

Sign convention for `dist`: POSITIVE places the mark's centre OUTSIDE the
W x H artwork rectangle (the default, and how registration marks are used in
print/cut production — they sit in the margin around the artwork, never
overlapping it), NEGATIVE pulls it inward, onto the artwork itself. Because
positive marks fall outside [0,W]x[0,H], any caller that also NEEDS the
canvas/document it draws onto to actually contain them (raster/SVG/gcode/plot
output, as opposed to a UI overlay that already has margin around the
artwork) must pad its own canvas by `required_margin()` and translate the
artwork by the same amount — see that function's docstring.

Three shapes:
  'circle' — a circle outline, useful for aligning cut stencils by eye.
  'cross'  — a solid "+" crosshair (symmetric on all 4 corners).
  'lines'  — corner tick marks: at each corner, only the two arms pointing
             AWAY from the artwork (e.g. the top-left mark draws its top and
             left arms, not bottom/right) — the standard print crop-mark
             convention, forming an L bracket that hugs the corner instead
             of a full crosshair.
"""

import math

SHAPES = ('circle', 'cross', 'lines')

# Per corner-index (matching corner_points' own order), the (sx, sy) unit
# direction each corner's marks should extend AWAY from the artwork in.
_CORNER_DIRS = [
    (-1, -1),  # top-left:     arms go left and up
    (1, -1),   # top-right:    arms go right and up
    (-1, 1),   # bottom-left:  arms go left and down
    (1, 1),    # bottom-right: arms go right and down
]


def contrast_color(canvas_hex: str) -> str:
    """Mark colour for a COMBINED file, which has no single layer colour to take: black on a
    light canvas, white on a dark one. Plain black marks vanish on a dark canvas (black on
    near-black), which defeats their purpose. Per-layer files don't use this -- their marks
    take that layer's own colour."""
    h = (canvas_hex or '').lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return '#000000'
    return '#000000' if (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 > 0.5 else '#FFFFFF'


def corner_points(W: float, H: float, dist: float):
    """4 (cx, cy) anchor points, `dist` from each corner of a W x H canvas.
    Positive dist -> outside the canvas; negative -> inside. Order matches
    _CORNER_DIRS: top-left, top-right, bottom-left, bottom-right."""
    return [
        (-dist, -dist),
        (W + dist, -dist),
        (-dist, H + dist),
        (W + dist, H + dist),
    ]


def required_margin(dist: float, size: float, shape: str) -> float:
    """Extra space a caller must pad onto EACH side of its W x H canvas so
    outward (positive-dist) marks aren't clipped. 0 when dist<=0 (marks sit
    on or inside the artwork, no extra canvas needed). The artwork itself
    must then be translated by (margin, margin) so it still lands correctly
    inside the padded canvas, and marks drawn via corner_points/all_marks
    using the ORIGINAL (un-padded) W/H, likewise translated by (margin,
    margin) -- i.e. pad first, then draw everything (artwork + marks) at
    `coord + margin`."""
    if dist <= 0:
        return 0.0
    if shape == 'circle':
        return dist + size / 2.0
    return dist + size  # cross/lines: arms reach `size` from the mark centre


def mark_primitives(cx: float, cy: float, size: float, shape: str, corner_idx: int = 0):
    """Primitives for one mark at (cx, cy): ('circle', cx, cy, r) or
    ('line', x0, y0, x1, y1). `size` is the user-facing mark size — a DIAMETER
    for the circle shape (halved to a radius here), and an arm length (centre
    to tip) for cross/lines, which have no separate "diameter" concept.
    `corner_idx` (0=TL, 1=TR, 2=BL, 3=BR, matching corner_points' order)
    only affects 'lines': which two arms get drawn."""
    if shape == 'circle':
        return [('circle', cx, cy, size / 2.0)]
    if shape == 'cross':
        return [
            ('line', cx - size, cy, cx + size, cy),
            ('line', cx, cy - size, cx, cy + size),
        ]
    if shape == 'lines':
        sx, sy = _CORNER_DIRS[corner_idx % 4]
        gap = size * 0.35
        return [
            ('line', cx + sx * gap, cy, cx + sx * size, cy),
            ('line', cx, cy + sy * gap, cx, cy + sy * size),
        ]
    return []


def all_marks(W: float, H: float, dist: float, size: float, shape: str,
              offset=(0.0, 0.0)):
    """Flattened primitives for all 4 corners. `offset`: added to every
    resulting coordinate -- for a caller that pads its own canvas by
    `required_margin()` and translates the artwork by that same margin
    (positive-dist marks fall outside the original [0,W]x[0,H] rect, so they
    need the same translation the artwork got to land inside the padded
    canvas instead of at a negative/out-of-bounds coordinate)."""
    ox, oy = offset
    out = []
    for idx, (cx, cy) in enumerate(corner_points(W, H, dist)):
        for p in mark_primitives(cx, cy, size, shape, idx):
            if p[0] == 'circle':
                out.append(('circle', p[1] + ox, p[2] + oy, p[3]))
            else:
                out.append(('line', p[1] + ox, p[2] + oy, p[3] + ox, p[4] + oy))
    return out


def marks_svg_snippet(W: float, H: float, dist: float, size: float, shape: str,
                       stroke: str = '#000000', stroke_width: float = 1.0,
                       offset=(0.0, 0.0)) -> str:
    """Raw SVG markup (no wrapping <g>) for all 4 corner marks."""
    parts = []
    for p in all_marks(W, H, dist, size, shape, offset):
        if p[0] == 'circle':
            _, cx, cy, r = p
            parts.append(
                f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" fill="none" '
                f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
            )
        else:
            _, x0, y0, x1, y1 = p
            parts.append(
                f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
                f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
            )
    return ''.join(parts)


def draw_marks_cv(img, W: float, H: float, dist: float, size: float, shape: str,
                   color=(0, 0, 0), thickness: int = 1, offset=(0.0, 0.0)):
    """Draw all 4 corner marks directly onto a BGR or BGRA numpy array (mutated in place)."""
    import cv2
    col = (*color, 255) if img.shape[2] == 4 else color
    for p in all_marks(W, H, dist, size, shape, offset):
        if p[0] == 'circle':
            _, cx, cy, r = p
            cv2.circle(img, (int(round(cx)), int(round(cy))), max(1, int(round(r))),
                       col, thickness, cv2.LINE_AA)
        else:
            _, x0, y0, x1, y1 = p
            cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
                     col, thickness, cv2.LINE_AA)


def mark_paths(W: float, H: float, dist: float, size: float, shape: str, circle_segments: int = 24,
               offset=(0.0, 0.0)):
    """Point-paths for pen-plotter output: list of paths, each a list of (x, y)
    points to draw pen-down (a circle is approximated as a closed polygon)."""
    paths = []
    for p in all_marks(W, H, dist, size, shape, offset):
        if p[0] == 'circle':
            _, cx, cy, r = p
            n = circle_segments
            paths.append([
                (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
                for i in range(n + 1)
            ])
        else:
            _, x0, y0, x1, y1 = p
            paths.append([(x0, y0), (x1, y1)])
    return paths
