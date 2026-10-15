"""
Plotter-ready ("Plot SVG") export — stroke-only SVG for pen plotters.

Sites like plotterfiles.com distinguish a normal illustrative SVG (filled shapes,
meant for screens) from a "Plot SVG": no fills at all, only actual pen strokes,
since a physical plotter can only draw lines — it has no way to "fill" a region.

Smalti's mosaics are made of solid dots/cells, which a plotter obviously can't
draw as flat fills. The G-code exporter (gcode_export.py) already solves exactly
this — it fills each shape with a continuous spiral stroke close enough together
(pen-width spacing) to read as solid coverage once drawn. This module reuses that
SAME spiral geometry (compute_layer_spirals) and serializes it as SVG <path>
strokes instead of G-code, so the file can be opened directly in plotter software
(Inkscape + a plotter extension, vpype, etc.) rather than needing raw G-code.
"""

from pathlib import Path

from .gcode_export import _to_mm, compute_layer_spirals


def _path_d(shapes: list[list[tuple[float, float]]]) -> str:
    """One SVG path `d` attribute covering every shape in a layer — each shape is
    its own "M ... L L L ..." subpath, so plotter software treats the gap between
    shapes as a pen-up travel move, same as the G-code's G0 travel between spirals."""
    parts = []
    for path in shapes:
        if not path:
            continue
        x0, y0 = path[0]
        if len(path) == 1:
            # A target circle smaller than the pen's own width is a single touch-down
            # point (see compute_layer_spirals) — a zero-length "M x L x" segment with
            # round linecap renders as a solid dot of diameter = stroke-width, which is
            # exactly the physical mark a pen makes touching down without moving.
            parts.append(f"M {x0:.3f} {y0:.3f} L {x0:.3f} {y0:.3f}")
            continue
        parts.append(f"M {x0:.3f} {y0:.3f}")
        for (x, y) in path[1:]:
            parts.append(f"L {x:.3f} {y:.3f}")
    return " ".join(parts)


def write_plot_svg(
    layer_results,          # list of (circles, hex, name) — circles in working px
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    mode: str,               # 'circle' | 'polygon'
    pen_width_mm: float,
    poly_inset: float = 0.08,
    poly_smooth: float = 0.0,
    r_min: float = 1.0,
    r_max: float = 10.0,
    passes: int = 2,
    per_layer: bool = True,
    canvas_hex: str = None,  # a layer matching this colour is skipped (paper shows through)
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale = _to_mm(physical_long_edge, unit) / float(max(working_w, working_h))
    svg_w_mm = working_w * scale
    svg_h_mm = working_h * scale
    stroke_w = max(float(pen_width_mm), 0.05)
    reg_marks_svg = ''       # combined file: neutral black, no single layer colour applies
    dist_mm = size_mm = 0.0
    margin_mm = 0.0
    if reg_marks:
        from .registration_marks import marks_svg_snippet, required_margin
        dist_mm = _to_mm(reg_mark_dist, unit)
        size_mm = _to_mm(reg_mark_size, unit)
        margin_mm = required_margin(dist_mm, size_mm, reg_mark_shape)
        reg_marks_svg = marks_svg_snippet(
            svg_w_mm, svg_h_mm, dist_mm, size_mm, reg_mark_shape,
            stroke='#000000', stroke_width=stroke_w,
        )

    def reg_marks_svg_for(color_hex):
        # Per-layer files draw marks in THAT layer's own colour -- same
        # reasoning as write_layer_svg: if you're plotting/cutting each
        # colour as its own pass, the mark should come from that same pass.
        if not reg_marks:
            return ''
        from .registration_marks import marks_svg_snippet
        return marks_svg_snippet(
            svg_w_mm, svg_h_mm, dist_mm, size_mm, reg_mark_shape,
            stroke=color_hex, stroke_width=stroke_w,
        )

    layer_spirals = compute_layer_spirals(
        layer_results, mode, working_w, working_h, scale, pen_width_mm,
        poly_inset, poly_smooth, r_min, r_max, passes, canvas_hex=canvas_hex,
    )

    def svg_open():
        # SVG viewBoxes support a non-zero, negative origin natively, so
        # padding for outward (positive-dist) marks only needs the
        # viewBox/width/height widened -- every stroke coordinate below is
        # computed in the ORIGINAL (un-padded) mm frame and needs no
        # translation; the shifted viewBox does that for free.
        vb_w, vb_h = svg_w_mm + 2 * margin_mm, svg_h_mm + 2 * margin_mm
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{vb_w:.3f}mm" height="{vb_h:.3f}mm" '
            f'viewBox="{-margin_mm:.3f} {-margin_mm:.3f} {vb_w:.3f} {vb_h:.3f}">\n'
        )

    def path_elem(d, color):
        return (
            f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_w:.3f}" stroke-linecap="round" stroke-linejoin="round"/>\n'
        )

    paths_written = []

    if per_layer:
        for li, (_circles, hexc, _name) in enumerate(layer_results):
            shapes = layer_spirals[li]
            if not shapes:
                continue
            svg = svg_open() + path_elem(_path_d(shapes), hexc) + reg_marks_svg_for(hexc) + '</svg>\n'
            p = out / f"plot_layer_{li + 1:02d}_{hexc.lstrip('#').lower()}.svg"
            p.write_text(svg)
            paths_written.append(str(p))

    svg = svg_open()
    any_shapes = False
    for li, (_circles, hexc, _name) in enumerate(layer_results):
        shapes = layer_spirals[li]
        if not shapes:
            continue
        any_shapes = True
        svg += path_elem(_path_d(shapes), hexc)
    svg += reg_marks_svg + '</svg>\n'
    if any_shapes:
        combined_path = out / "plot_combined.svg"
        combined_path.write_text(svg)
        paths_written.append(str(combined_path))

    return paths_written
