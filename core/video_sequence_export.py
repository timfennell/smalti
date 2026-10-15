"""
Video → per-frame file sequence export.

An alternative to core.video_render's single-video output: instead of muxing frames
into an .mp4, each frame is exported through the SAME writer a still image would use
(SVG / PNG / JPEG / TIFF / G-code / Plot SVG), landing in its own frame_NNNNN/
subfolder. Meant for physical output — e.g. plotting or printing one page per frame
to build a flip-book from a processed video clip. 3D formats are deliberately not
offered here (a per-frame 3D export doesn't map onto "flip through pages").
"""

import json
from pathlib import Path

from .video_render import _frame_layers

_SUPPORTED_FORMATS = ('svg', 'png', 'jpeg', 'tiff', 'gcode', 'plot_svg')


def export_frame_sequence(
    json_path: str,
    output_dir: str,
    fmt: str,
    *,
    mode=None,
    eq_curve=None,
    poly_inset=None,
    poly_smooth=None,
    poly_gradient=None,
    visible_colors=None,
    physical_long_edge: float = 0.0,
    unit: str = 'cm',
    dpi: float = 96.0,
    long_px: int = 0,             # raster formats only; 0 = working size (1:1)
    pen_width_mm: float = 0.5,    # gcode / plot_svg
    passes: int = 2,              # gcode / plot_svg
    gcode_origin: str = 'bottom-left',
    per_layer: bool = False,
    jpeg_quality: int = 92,
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
    progress_cb=None,             # (frame_index, total, message) -> None
    stop_event=None,
) -> str:
    """Run the per-frame export. Returns output_dir; each frame lands in its own
    frame_NNNNN/ subfolder written by the same code path as still-image export."""
    if fmt not in _SUPPORTED_FORMATS:
        raise RuntimeError(f"unsupported sequence export format: {fmt}")

    data = json.loads(Path(json_path).read_text())
    palette = data['palette']
    canvas_hex = data.get('canvas', '#FFFFFF')
    mode = mode or data.get('mode', 'polygon')
    Wg, Hg = data.get('working_size', [0, 0])
    r_min = data.get('r_min', 1.0)
    r_max = data.get('r_max', 10.0)
    poly_inset = data.get('poly_inset', 0.08) if poly_inset is None else poly_inset
    poly_smooth = data.get('poly_smooth', 0.0) if poly_smooth is None else poly_smooth
    poly_gradient = data.get('poly_gradient', False) if poly_gradient is None else poly_gradient
    eq_curve = eq_curve if eq_curve is not None else data.get('airbrush_eq')
    frames = data['frames']
    visible = set(visible_colors) if visible_colors is not None else None

    if not frames or Wg <= 0 or Hg <= 0:
        raise RuntimeError("mosaic_video.json has no renderable frames")
    if fmt in ('gcode', 'plot_svg') and mode not in ('circle', 'polygon', 'tile'):
        raise RuntimeError("G-code / Plot SVG need Circle, Polygon, or Tile render mode "
                            "(Airbrush has no outline to fill)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(frames)

    for i, frame in enumerate(frames):
        if stop_event is not None and stop_event.is_set():
            break
        layers = _frame_layers(frame, palette, visible)
        frame_dir = str(out / f'frame_{i:05d}')

        if fmt == 'svg':
            from .svg_writer import write_all_outputs
            write_all_outputs(
                layers, canvas_hex, frame_dir, Wg, Hg, physical_long_edge, unit, dpi,
                mode, r_min=r_min, r_max=r_max, eq_curve=eq_curve,
                poly_inset=poly_inset, poly_gradient=poly_gradient, poly_smooth=poly_smooth,
                per_layer=per_layer,
                reg_marks=reg_marks, reg_mark_dist=reg_mark_dist,
                reg_mark_size=reg_mark_size, reg_mark_shape=reg_mark_shape,
            )
        elif fmt == 'gcode':
            from .gcode_export import write_gcode
            write_gcode(
                layers, canvas_hex, frame_dir, Wg, Hg, physical_long_edge, unit, dpi,
                mode, pen_width_mm, poly_inset=poly_inset, poly_smooth=poly_smooth,
                r_min=r_min, r_max=r_max, passes=passes, per_layer=per_layer,
                origin=gcode_origin,
                reg_marks=reg_marks, reg_mark_dist=reg_mark_dist,
                reg_mark_size=reg_mark_size, reg_mark_shape=reg_mark_shape,
            )
        elif fmt == 'plot_svg':
            from .plot_svg_export import write_plot_svg
            write_plot_svg(
                layers, frame_dir, Wg, Hg, physical_long_edge, unit, mode,
                pen_width_mm, poly_inset=poly_inset, poly_smooth=poly_smooth,
                r_min=r_min, r_max=r_max, passes=passes, per_layer=per_layer,
                canvas_hex=canvas_hex,
                reg_marks=reg_marks, reg_mark_dist=reg_mark_dist,
                reg_mark_size=reg_mark_size, reg_mark_shape=reg_mark_shape,
            )
        else:   # png / jpeg / tiff
            from .raster_export import write_raster_outputs
            lp = long_px or max(Wg, Hg)
            write_raster_outputs(
                layers, canvas_hex, frame_dir, Wg, Hg, lp, fmt, mode=mode,
                r_min=r_min, r_max=r_max, eq_curve=eq_curve,
                poly_inset=poly_inset, poly_smooth=poly_smooth, poly_gradient=poly_gradient,
                per_layer=per_layer, jpeg_quality=jpeg_quality,
                reg_marks=reg_marks,
                reg_mark_dist_px=(reg_mark_dist / physical_long_edge * lp) if (reg_marks and physical_long_edge) else 0.0,
                reg_mark_size_px=(reg_mark_size / physical_long_edge * lp) if (reg_marks and physical_long_edge) else 0.0,
                reg_mark_shape=reg_mark_shape,
            )

        if progress_cb:
            progress_cb(i + 1, n, f"Exported frame {i + 1}/{n}")

    return str(out)
