"""
Top-level processing pipeline.

Phases:
  1. Load + preprocess
  2. SIFT initialization (keypoints at natural scales)
  3. Poisson disk gap-fill (full coverage guarantee)
  4. Iterative Lloyd optimization (move/resize/recolor circles to minimize error)
  5. SVG output
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from .optimizer import (
    _delaunay_neighbors, _voronoi_polygons, _preview_subsample, compute_preview_tiles,
)

from .image_processor import load_image, load_image_array, hex_to_lab
from .color_decomp import build_weight_maps
from .detail_map import compute_detail_map, build_radius_field
from .poisson_sampler import sift_seeds
from .poisson_parallel import poisson_gap_fill_parallel as poisson_gap_fill
from .optimizer import CircleOptimizer, OptimizerConfig
from .svg_writer import write_all_outputs


@dataclass
class PipelineConfig:
    image_path: str
    max_working_px: int = 2000

    canvas_hex: str = "#FFFFFF"
    palette_hex: list = field(default_factory=lambda: ["#000000"])

    # Circle sizes in physical units
    r_min_physical: float = 0.1
    r_max_physical: float = 1.0
    unit: str = 'cm'
    physical_long_edge: float = 30.0
    dpi: float = 96.0

    gap_physical: float = 0.05
    color_softness: float = 30.0
    canvas_comp: float = 0.35   # canvas colour show-through compensation (0 = ignore canvas)
    chroma_weight: float = 0.0
    detail_boost: float = 1.2
    detail_blur_sigma: float = 2.0
    lum_influence: float = 0.5   # 0=detail-only, 1=luminosity-only circle sizing
    poisson_k: int = 30
    mode: str = 'circle'   # 'circle' | 'airbrush' | 'polygon'
    # Airbrush EQ: 5 scale multipliers for [smallest … largest] circle size bands.
    # Each value scales the Gaussian blob radius relative to the circle radius.
    # e.g. [1.5, 1.3, 1.2, 1.1, 1.0] = small circles spread more than large ones.
    airbrush_eq: list = None
    poly_inset: float = 0.08   # Voronoi cell inset ratio (fraction of each cell's equivalent radius)
    pre_quantize: bool = False

    # Optimization
    n_iterations: int = 8
    dither_strength: float = 0.7    # 0=pure closest-color, 1=full optical mixing
    blend_radius: float = 3.5       # neighbor search radius (× circle radius)
    dither_kernel: str = 'ball'      # 'sierra' | 'ball'

    output_dir: str = "./output"


@dataclass
class PipelineResult:
    layer_circles: list
    output_paths: list[str]
    working_size: tuple[int, int]
    original_size: tuple[int, int]
    total_circles: int
    circles_per_layer: list[int]
    r_min_px: float = 1.0   # working-pixel radius bounds — needed for re-export
    r_max_px: float = 10.0


def run_pipeline(
    config: PipelineConfig,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    preview_cb: Optional[Callable[[str], None]] = None,
    stop_event=None,
    image_array=None,
    init_circles=None,
) -> PipelineResult:
    """image_array: optional in-memory RGB uint8 frame to process instead of loading
    config.image_path (used for video frames).
    init_circles: optional list of (x, y, r) in working-pixel space to WARM-START the
    optimizer with — skips SIFT + Poisson placement so the dots evolve from the previous
    frame rather than being re-seeded (temporal coherence for video)."""

    def emit(msg: str, pct: float):
        if progress_cb:
            progress_cb(msg, pct)

    def check_stop():
        if stop_event and stop_event.is_set():
            raise InterruptedError("cancelled")

    emit("Loading image…", 0.0)
    if image_array is not None:
        img = load_image_array(image_array, config.max_working_px)
    else:
        img = load_image(config.image_path, config.max_working_px)
    W, H = img.working_size

    emit("Building color weight maps…", 0.05)
    weight_maps, contrast_map, comp_lab = build_weight_maps(
        img,
        config.palette_hex,
        config.canvas_hex,
        config.color_softness,
        config.canvas_comp,
        config.chroma_weight,
    )

    emit("Computing detail map…", 0.10)
    detail = compute_detail_map(
        img.rgb_float,
        blur_sigma=config.detail_blur_sigma,
        detail_boost=config.detail_boost,
    )

    # Physical → working-pixel conversions
    px_per_unit = _px_per_unit(config.unit, config.dpi)
    _, _, svg_scale = _svg_canvas(W, H, config.physical_long_edge, config.unit, config.dpi)
    work_scale = 1.0 / svg_scale

    r_min = max(config.r_min_physical * px_per_unit * work_scale, 1.0)
    # Only floor r_max up to r_min (never let max < min) -- do NOT force a
    # minimum +1px gap here. That forced gap silently broke an explicit
    # r_min == r_max request (e.g. both set to 0.25cm for uniform-size
    # circles): every circle still came out with up to 1px of size variation
    # regardless of the user's setting. r_max == r_min is safe all the way
    # through (build_radius_field's r_min + (r_max-r_min)*t collapses to a
    # constant r_min with no divide-by-zero).
    r_max = max(config.r_max_physical * px_per_unit * work_scale, r_min)
    gap   = config.gap_physical * px_per_unit * work_scale

    # Build radius field (detail + contrast → per-pixel target radius)
    max_weight = weight_maps.max(axis=0)
    canvas_lab  = hex_to_lab(config.canvas_hex)
    canvas_L = float(canvas_lab[0])
    radius_map = build_radius_field(
        detail, max_weight, contrast_map, r_min, r_max,
        luminosity=img.lab[:, :, 0],
        lum_influence=config.lum_influence,
        canvas_lightness=canvas_L,
    )
    bright_mask = img.lab[:, :, 0] > 60
    dark_mask   = img.lab[:, :, 0] < 20
    print(f"[pipeline] canvas_L={canvas_L:.1f}  r_min={r_min:.2f} r_max={r_max:.2f}", flush=True)
    print(f"[pipeline] radius bright_px mean={radius_map[bright_mask].mean():.2f}  dark_px mean={radius_map[dark_mask].mean():.2f}", flush=True)

    # ── Optional: pre-quantize source to palette colors ──────────────────────
    if config.pre_quantize:
        emit("Pre-quantizing image to palette…", 0.12)
        palette_lab = np.array([hex_to_lab(h) for h in config.palette_hex])
        src = img.lab.reshape(-1, 3).astype(np.float32)
        pal = palette_lab.astype(np.float32)
        # nearest palette color per pixel in LAB
        diffs = src[:, None, :] - pal[None, :, :]          # (N, C, 3)
        dists = (diffs ** 2).sum(axis=2)                    # (N, C)
        nearest = dists.argmin(axis=1)                      # (N,)
        img.lab = pal[nearest].reshape(img.lab.shape)

    if init_circles is not None:
        # ── Warm start (video): reuse the previous frame's dots, clamped to canvas ──
        emit("Warm-start from previous frame…", 0.15)
        all_circles = []
        for (x, y, r) in init_circles:
            x = min(max(float(x), 0.0), W - 1.0)
            y = min(max(float(y), 0.0), H - 1.0)
            all_circles.append((x, y, float(r)))
        emit(f"Warm-start: {len(all_circles):,} circles", 0.40)
    else:
        # ── Phase 1: SIFT initialization ─────────────────────────────────────────
        emit("SIFT keypoint detection…", 0.15)
        import cv2
        gray = cv2.cvtColor((img.rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        raw_sift = sift_seeds(gray_f, r_min, r_max)
        from .poisson_sampler import SpatialGrid
        placement_gap = max(gap, 0.0)  # negative gap = airbrush overlap; clamp for placement
        _sift_grid = SpatialGrid(W, H, max(r_min / 1.42, 0.5))
        sift_circles = []
        for (x, y, r) in raw_sift:
            if (x - r < placement_gap or x + r > W - 1 - placement_gap or
                    y - r < placement_gap or y + r > H - 1 - placement_gap):
                continue
            if _sift_grid.has_conflict(x, y, r, placement_gap):
                continue
            _sift_grid.insert(x, y, r)
            sift_circles.append((x, y, r))
        emit(f"SIFT: {len(sift_circles)} keypoints (filtered)", 0.20)

        # ── Phase 2: Poisson gap-fill ─────────────────────────────────────────────
        emit("Poisson gap-fill for full coverage…", 0.22)

        def _poisson_preview(circles_so_far):
            if not preview_cb:
                return
            import json as _json
            colors_tmp = _assign_colors(circles_so_far, weight_maps)
            xs_t = np.array([c[0] for c in circles_so_far], dtype=np.float32)
            ys_t = np.array([c[1] for c in circles_so_far], dtype=np.float32)
            rs_t = np.array([c[2] for c in circles_so_far], dtype=np.float32)
            # neighbors/polygons are only read by the JS preview's 'polygon' mode
            # branch — computing a full Delaunay+Voronoi structure here regardless of
            # mode is wasted work for circle/airbrush, and during gap-fill this can
            # fire repeatedly as the circle count climbs into the tens of thousands.
            # Same reasoning as optimizer.py's cleanup-pass fix; capped for the same
            # "many circles must stay slow, never fail" reason (see main.py's
            # websocket buffer-size fix + optimizer.py's _preview_subsample).
            if config.mode == 'polygon':
                nbrs, polys = _delaunay_neighbors(xs_t, ys_t, W, H), _voronoi_polygons(xs_t, ys_t, W, H)
                pxs, pys, prs, pcolors = xs_t, ys_t, rs_t, colors_tmp
            else:
                nbrs, polys = None, None
                pxs, pys, prs, pcolors = _preview_subsample(xs_t, ys_t, rs_t, colors_tmp)
            preview_cb(_json.dumps({
                'xs': pxs.tolist(), 'ys': pys.tolist(), 'rs': prs.tolist(),
                'colors': pcolors.tolist(),
                'palette': config.palette_hex, 'canvas_hex': config.canvas_hex,
                'W': W, 'H': H, 'mode': config.mode, 'phase': 'poisson',
                'r_min': float(r_min), 'r_max': float(r_max),
                'neighbors': nbrs,
                'polygons': polys,
            }))

        def _poisson_progress(n):
            check_stop()
            emit(f"Gap-fill… {n:,} circles", 0.22)

        new_circles = poisson_gap_fill(
            sift_circles,
            radius_map,
            gap=placement_gap,
            candidates_k=config.poisson_k,
            progress_callback=_poisson_progress,
            preview_callback=_poisson_preview,
            preview_every=2000,
        )
        all_circles = sift_circles + new_circles
        emit(f"Initial placement: {len(all_circles):,} circles", 0.40)
    check_stop()

    # ── Phase 3: Assign initial colors ───────────────────────────────────────
    color_indices = _assign_colors(all_circles, weight_maps)

    # ── Phase 4: Lloyd optimization ───────────────────────────────────────────
    if config.n_iterations > 0:
        palette_lab = np.array([hex_to_lab(h) for h in config.palette_hex])

        opt_cfg = OptimizerConfig(
            r_min=r_min,
            r_max=r_max,
            gap=gap,
            coverage_factor=0.65,   # fixed: circles auto-size to cover their cells efficiently
                                    # (coverage is handled by placement + this scaling, not a user slider)
            n_iterations=config.n_iterations,
            dither_strength=config.dither_strength,
            blend_radius=config.blend_radius,
            lum_influence=config.lum_influence,
            allow_restructure=(init_circles is None),
            dither_kernel=config.dither_kernel,
        )

        optimizer = CircleOptimizer(
            circles=all_circles,
            color_indices=color_indices,
            source_lab=img.lab,
            weight_maps=weight_maps,
            detail_map=detail,
            palette_lab=palette_lab,
            canvas_lab=canvas_lab,
            config=opt_cfg,
            target_lab=comp_lab,   # canvas-compensated colours for recolouring
        )

        opt_pct_start = 0.42
        opt_pct_range = 0.45
        total_iter = config.n_iterations

        def opt_cb(iter_n: int, err: float):
            check_stop()
            if iter_n > total_iter:
                emit(f"Cleanup pass…  (ΔE {err:.1f})", opt_pct_start + opt_pct_range)
            else:
                pct = opt_pct_start + (iter_n / total_iter) * opt_pct_range
                emit(f"Optimizing… iteration {iter_n}/{total_iter}  (ΔE {err:.1f})", pct)

        # Emit initial placement preview before optimization starts
        if preview_cb:
            import json as _json
            xs0 = np.array([c[0] for c in all_circles], dtype=np.float32)
            ys0 = np.array([c[1] for c in all_circles], dtype=np.float32)
            rs0 = np.array([c[2] for c in all_circles], dtype=np.float32)
            if config.mode == 'polygon':
                nbrs0, polys0 = _delaunay_neighbors(xs0, ys0, W, H), _voronoi_polygons(xs0, ys0, W, H)
                pxs0, pys0, prs0, pci0 = xs0, ys0, rs0, color_indices
            else:
                nbrs0, polys0 = None, None
                pxs0, pys0, prs0, pci0 = _preview_subsample(xs0, ys0, rs0, color_indices)
            preview_cb(_json.dumps({
                'xs': pxs0.tolist(), 'ys': pys0.tolist(), 'rs': prs0.tolist(),
                'colors': pci0.tolist(),
                'palette': config.palette_hex, 'canvas_hex': config.canvas_hex,
                'W': W, 'H': H, 'mode': config.mode, 'phase': 'initial',
                'r_min': float(r_min), 'r_max': float(r_max),
                'neighbors': nbrs0,
                'polygons': polys0,
            }))

        all_circles, color_indices = optimizer.run(
            opt_cb,
            preview_cb=preview_cb,
            palette_hex=config.palette_hex,
            canvas_hex=config.canvas_hex,
            mode=config.mode,
            stop_event=stop_event,
        )

    # Final preview (covers n_iterations=0 case too)
    if preview_cb:
        import json as _json
        xs_f = np.array([c[0] for c in all_circles], dtype=np.float32)
        ys_f = np.array([c[1] for c in all_circles], dtype=np.float32)
        rs_f = np.array([c[2] for c in all_circles], dtype=np.float32)
        ci_f = color_indices if isinstance(color_indices, np.ndarray) else np.array(color_indices)
        tiles_f, tile_color_idx_f = None, None
        if config.mode == 'polygon':
            nbrs_f, polys_f = _delaunay_neighbors(xs_f, ys_f, W, H), _voronoi_polygons(xs_f, ys_f, W, H)
            pxs_f, pys_f, prs_f, pci_f = xs_f, ys_f, rs_f, ci_f
        elif config.mode == 'tile' and len(xs_f) >= 3:
            # Full row-based tile packing is too slow (tens of seconds) to run on
            # every optimization iteration, so — like the polygon mode's Voronoi
            # cells above — it's only computed once here, for the finished result.
            # The live preview shows circles until this lands (see JS renderScene's
            # `mode==='tile' && !circleData.tiles` fallback). See
            # optimizer.compute_preview_tiles for the downscaled-canvas approach
            # and why tile SIZE (not just position) needs care when re-projecting
            # back to the full W x H frame — shared with the on-demand re-preview
            # main.py does when the render-mode toggle switches to 'tile' after a
            # completed run, so both paths get any future fix at once.
            tiles_f, tile_color_idx_f = compute_preview_tiles(
                xs_f, ys_f, rs_f, ci_f, config.palette_hex, W, H)
            nbrs_f, polys_f = None, None
            pxs_f, pys_f, prs_f, pci_f = _preview_subsample(xs_f, ys_f, rs_f, ci_f)
        else:
            nbrs_f, polys_f = None, None
            pxs_f, pys_f, prs_f, pci_f = _preview_subsample(xs_f, ys_f, rs_f, ci_f)
        preview_cb(_json.dumps({
            'xs': pxs_f.tolist(), 'ys': pys_f.tolist(), 'rs': prs_f.tolist(),
            'colors': pci_f.tolist(),
            'palette': config.palette_hex, 'canvas_hex': config.canvas_hex,
            'W': W, 'H': H, 'mode': config.mode, 'phase': 'final',
            'r_min': float(r_min), 'r_max': float(r_max),
            'neighbors': nbrs_f,
            'polygons': polys_f,
            'tiles': tiles_f,
            'tile_color_idx': tile_color_idx_f,
        }))

    emit("Finalizing…", 0.92)

    # Split circles by color layer.  NOTE: files are NOT written here — export is
    # explicit (the Export button).  Auto-writing on every generation also failed
    # on macOS when the output folder (e.g. ~/Desktop) wasn't user-granted, raising
    # PermissionError: Operation not permitted.  Choosing a folder via Browse grants
    # that access; generation itself just produces the in-memory result + preview.
    n_colors = len(config.palette_hex)
    layer_map: dict[int, list] = {k: [] for k in range(n_colors)}
    for i, (x, y, r) in enumerate(all_circles):
        layer_map[color_indices[i]].append((x, y, r))

    layer_results = []
    circles_per_layer = []
    for k, hex_color in enumerate(config.palette_hex):
        circles = layer_map[k]
        layer_results.append((circles, hex_color, hex_color))
        circles_per_layer.append(len(circles))

    emit("Done.", 1.0)

    return PipelineResult(
        layer_circles=layer_results,
        output_paths=[],
        working_size=img.working_size,
        original_size=img.original_size,
        total_circles=len(all_circles),
        circles_per_layer=circles_per_layer,
        r_min_px=r_min,
        r_max_px=r_max,
    )


def _assign_colors(
    circles: list[tuple[float, float, float]],
    weight_maps: np.ndarray,
) -> np.ndarray:
    H, W = weight_maps.shape[1], weight_maps.shape[2]
    indices = np.zeros(len(circles), dtype=np.int32)
    for i, (x, y, _) in enumerate(circles):
        ix = int(np.clip(x, 0, W - 1))
        iy = int(np.clip(y, 0, H - 1))
        indices[i] = int(np.argmax(weight_maps[:, iy, ix]))
    return indices



def _px_per_unit(unit: str, dpi: float) -> float:
    return dpi * {'in': 1.0, 'cm': 1.0 / 2.54, 'mm': 0.1 / 2.54, 'px': 1.0 / 96.0}[unit]


def _svg_canvas(W, H, physical_long_edge, unit, dpi):
    from .units import svg_canvas_size
    return svg_canvas_size(W, H, physical_long_edge, unit, dpi)
