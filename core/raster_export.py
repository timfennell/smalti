"""
Raster export: render the mosaic to PNG / JPEG / layered TIFF at a chosen
long-edge resolution.

Shapes are drawn directly with OpenCV (anti-aliased) rather than rasterising the
SVG, so no external SVG renderer is required.  Geometry matches the SVG/preview:
- circle mode  → filled discs
- polygon mode → shared-circumcenter Voronoi cells, rounded + inset (same
                 morphological-opening outline as the vector export), flattened
                 to points for fillPoly
- airbrush     → Gaussian blobs composited additively toward the paint colour

Layered TIFF = one page per palette colour (each colour on the canvas
background), LZW-compressed.  PNG/JPEG = the flattened composite.
"""

import math
import time
import numpy as np
import cv2

# Explicit GIL yield cadence for the heavy per-dot/per-cell Python loops below — these
# run on a background export thread and can otherwise dominate a core for a long stretch
# on a large mosaic or a video's many frames, making the server feel unresponsive.
_YIELD_EVERY = 1000
from pathlib import Path
from PIL import Image

from .optimizer import _voronoi_polygons, _delaunay_neighbors, _flow_tiles


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _tile_inset_points(poly_pts, inset, smooth=0.0):
    """Grout inset for a flow-tile (rectangle/triangle from `_flow_tiles`, NOT a
    Voronoi cell). `_rounded_cell_points`'s erosion math assumes a roughly
    star-convex cell centred on its own site — flow tiles are often thin/elongated
    (row-packed rectangles, tapered triangle caps), and that erosion degenerates
    badly on shapes like that (self-intersecting slivers). Shapely's `buffer()` is
    a robust, shape-agnostic erosion that handles thin/concave polygons correctly,
    so use that instead. Returns None if the tile erodes away completely."""
    if inset <= 0:
        return np.asarray(poly_pts, dtype=np.float32)
    from shapely.geometry import Polygon
    try:
        poly = Polygon(poly_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # `inset` is one GLOBAL value (derived from the whole image's overall
        # circle-radius range) applied to every tile — flow-tile sizes are now
        # clamped to a much narrower, more uniform band than that range (real
        # tesserae don't vary 5x in size), so the same absolute inset can end up
        # eating a much larger FRACTION of a small tile than intended, eroding
        # most of them away entirely. Cap it to a fraction of this tile's own
        # size so grout thickness scales with the tile it's actually cutting into.
        # Erosion eats into a shape from every side, so a cap of fraction f of the
        # extent can remove up to 2f of it — for a roughly square tile, area drops
        # to (1-2f)². At f=0.3 that leaves only ~16% of the tile's area, which is
        # exactly what "cap over-erosion" was supposed to prevent. Cap at a much
        # smaller fraction that reads as a fine grout line, not a shrink-to-nothing.
        minx, miny, maxx, maxy = poly.bounds
        tile_extent = min(maxx - minx, maxy - miny)
        eff_inset = min(inset, tile_extent * 0.05)
        join_style = 1 if smooth > 0 else 2   # 1=round, 2=mitre (sharp corners)
        eroded = poly.buffer(-eff_inset, join_style=join_style)
    except Exception:
        return None
    if eroded.is_empty:
        return None
    # A thin tile can erode into multiple pieces or lose area to a hole — take the
    # largest single piece's outer boundary (holes/extra pieces are exactly the
    # kind of degenerate result that's not worth rendering as a separate shape).
    if eroded.geom_type == 'MultiPolygon':
        eroded = max(eroded.geoms, key=lambda g: g.area)
    if eroded.geom_type != 'Polygon' or eroded.exterior is None:
        return None
    coords = list(eroded.exterior.coords)[:-1]
    if len(coords) < 3:
        return None
    return np.asarray(coords, dtype=np.float32)


def _rounded_cell_points(pts, smooth, inset):
    """Flatten one rounded+inset Voronoi cell (already in output-pixel coords) to a
    list of (x,y) points.  Mirrors svg_writer._opening_rounded_path but samples the
    arcs into line segments.  Returns np.ndarray (M,2) float32, or None if the cell
    vanishes under the grout inset."""
    n = len(pts)
    if n < 3:
        return None
    area2 = 0.0
    cxg = cyg = 0.0
    for i in range(n):
        a = pts[i]; b = pts[(i + 1) % n]
        area2 += a[0] * b[1] - b[0] * a[1]
        cxg += a[0]; cyg += a[1]
    cxg /= n; cyg /= n
    cell_scale = math.sqrt(abs(area2) * 0.5)
    r = smooth * cell_scale * 0.5
    E = r + inset
    if E < 0.01:
        return np.asarray(pts, dtype=np.float32)

    def build_edges(idxs):
        out = []
        for i in idxs:
            a = pts[i]; b = pts[(i + 1) % n]
            dx = b[0] - a[0]; dy = b[1] - a[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                continue
            dx /= ln; dy /= ln
            nx = -dy; ny = dx
            if nx * (cxg - a[0]) + ny * (cyg - a[1]) < 0:
                nx = -nx; ny = -ny
            out.append({'dx': dx, 'dy': dy, 'nx': nx, 'ny': ny, 'len': ln,
                        'px': a[0] + nx * E, 'py': a[1] + ny * E})
        return out

    def isect(e1, e2):
        den = e1['dx'] * e2['dy'] - e1['dy'] * e2['dx']
        if abs(den) < 1e-9:
            return None
        t = ((e2['px'] - e1['px']) * e2['dy'] - (e2['py'] - e1['py']) * e2['dx']) / den
        return (e1['px'] + t * e1['dx'], e1['py'] + t * e1['dy'])

    idxs = list(range(n))
    centres = None
    guard = 0
    while len(idxs) >= 3 and guard < n + 2:
        guard += 1
        Es = build_edges(idxs)
        m = len(Es)
        if m < 3:
            idxs = []
            break
        C = [isect(Es[(k - 1) % m], Es[k]) for k in range(m)]
        ok = all(c is not None for c in C)
        worst = -1; worst_val = 0.0
        if ok:
            for k in range(m):
                cS = C[k]; cE = C[(k + 1) % m]
                span = (cE[0] - cS[0]) * Es[k]['dx'] + (cE[1] - cS[1]) * Es[k]['dy']
                if span < -1e-6 and span < worst_val:
                    worst_val = span; worst = k
        if ok and worst < 0:
            centres = C
            break
        if worst < 0:
            worst = min(range(len(idxs)), key=lambda k: Es[k]['len'] if k < len(Es) else 1e18)
        idxs.pop(worst)

    def arc_pts(cx, cy, rad, a0, a1, out):
        d = a1 - a0
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        steps = max(2, int(abs(d) * rad / 3) + 1)
        for k in range(1, steps + 1):
            a = a0 + d * k / steps
            out.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))

    # Fully collapsed → circle of radius (inradius − inset).
    if not centres or len(centres) < 3 or len(idxs) < 3:
        rin = math.inf
        for i in range(n):
            a = pts[i]; b = pts[(i + 1) % n]
            dx = b[0] - a[0]; dy = b[1] - a[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                continue
            rin = min(rin, abs(dx * (cyg - a[1]) - dy * (cxg - a[0])) / ln)
        rin = (rin - inset) if math.isfinite(rin) else 0.0
        if rin < 0.5:
            return None
        out = []
        arc_pts(cxg, cyg, rin, 0.0, math.pi, out)
        arc_pts(cxg, cyg, rin, math.pi, 2 * math.pi, out)
        return np.asarray(out, dtype=np.float32)

    Efin = build_edges(idxs)
    m = len(Efin)
    if r < 0.01:
        return np.asarray(centres, dtype=np.float32)

    def tan_start(k):
        c = centres[k]; e = Efin[k]
        return (c[0] - e['nx'] * r, c[1] - e['ny'] * r)

    def tan_end(k):
        c = centres[(k + 1) % m]; e = Efin[k]
        return (c[0] - e['nx'] * r, c[1] - e['ny'] * r)

    out = [tan_start(0)]
    for k in range(m):
        te = tan_end(k)
        out.append(te)
        c = centres[(k + 1) % m]
        ts = tan_start((k + 1) % m)
        a0 = math.atan2(te[1] - c[1], te[0] - c[0])
        a1 = math.atan2(ts[1] - c[1], ts[0] - c[0])
        arc_pts(c[0], c[1], r, a0, a1, out)
    return np.asarray(out, dtype=np.float32)


def _gaussian_stamp(rad: int) -> np.ndarray:
    """Normalized (0..1) radial falloff stamp of size (2*rad+1)."""
    if rad < 1:
        rad = 1
    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / rad
    a = np.clip(1.0 - d, 0.0, 1.0)
    return a * a * (3 - 2 * a)   # smoothstep falloff


def _mix_bgr(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _fill_gradient_triangle(img, cx, cy, v0, v1, center_bgr, edge_bgr):
    """Rasterize triangle (cx,cy)-v0-v1 with colour interpolated from center_bgr at the
    apex to edge_bgr along the v0-v1 edge (mirrors the SVG/JS pie-slice gradient: colour
    depends only on the apex's barycentric weight, not on position along the edge)."""
    H, W = img.shape[:2]
    x0 = max(int(np.floor(min(cx, v0[0], v1[0]))), 0)
    x1 = min(int(np.ceil(max(cx, v0[0], v1[0]))), W - 1)
    y0 = max(int(np.floor(min(cy, v0[1], v1[1]))), 0)
    y1 = min(int(np.ceil(max(cy, v0[1], v1[1]))), H - 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys_g, xs_g = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    x1p, y1p = v0; x2p, y2p = v1
    denom = (y1p - y2p) * (cx - x2p) + (x2p - x1p) * (cy - y2p)
    if abs(denom) < 1e-9:
        return
    w0 = ((y1p - y2p) * (xs_g - x2p) + (x2p - x1p) * (ys_g - y2p)) / denom
    w1 = ((y2p - cy) * (xs_g - x2p) + (cx - x2p) * (ys_g - y2p)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not inside.any():
        return
    edge_w = 1.0 - w0   # combined weight of v0+v1, which share edge_bgr
    region = img[y0:y1 + 1, x0:x1 + 1]
    for c in range(3):
        chan = center_bgr[c] * w0 + edge_bgr[c] * edge_w
        region[..., c][inside] = np.clip(chan[inside], 0, 255).astype(np.uint8)
    if img.shape[2] == 4:
        region[..., 3][inside] = 255


def _draw_gradient_polygon_layer(img, sel, all_xs, all_ys, cells, smooth, inset_px,
                                 point_bgr, neighbors):
    """Gradient variant of the polygon fill: fan-triangulate each cell's rounded+inset
    outline from its own site to every edge, colouring each triangle from the site's own
    colour out to the halfway colour with whichever neighbour cell is nearest that edge —
    the same construction as the SVG/live-preview gradient mode."""
    for step_i, i in enumerate(sel):
        if step_i % _YIELD_EVERY == 0:
            time.sleep(0)
        poly = cells[i]
        if poly is None:
            continue
        pchunk = _rounded_cell_points(poly, smooth, inset_px)
        if pchunk is None or len(pchunk) < 3:
            continue
        cx, cy = float(all_xs[i]), float(all_ys[i])
        center_bgr = point_bgr[i]
        nbrs = neighbors[i] if i < len(neighbors) else []
        pts = pchunk if not np.array_equal(pchunk[0], pchunk[-1]) else pchunk[:-1]
        n = len(pts)
        for e in range(n):
            v0 = pts[e]; v1 = pts[(e + 1) % n]
            mx = (v0[0] + v1[0]) * 0.5; my = (v0[1] + v1[1]) * 0.5
            best_j, best_d2 = None, float('inf')
            for j in nbrs:
                d2 = (all_xs[j] - mx) ** 2 + (all_ys[j] - my) ** 2
                if d2 < best_d2:
                    best_d2 = d2; best_j = j
            edge_bgr = _mix_bgr(center_bgr, point_bgr[best_j]) if best_j is not None else center_bgr
            _fill_gradient_triangle(img, cx, cy, v0, v1, center_bgr, edge_bgr)


def _reg_margin_px(reg_marks, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape):
    """Whole-pixel canvas margin needed on each side so outward (positive-
    dist) registration marks aren't clipped. 0 when marks are off, or dist
    <=0 (inside the artwork -- no extra canvas needed)."""
    if not reg_marks:
        return 0
    from .registration_marks import required_margin
    return int(round(required_margin(reg_mark_dist_px, reg_mark_size_px, reg_mark_shape)))


def _pad_and_mark(img, W, H, margin_px, canvas_bgr, alpha,
                   reg_marks, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape,
                   mark_color=(0, 0, 0)):
    """Paste an already-rendered W x H artwork image into a canvas padded by
    `margin_px` on every side (so it stays centred at the same position it
    would occupy with no padding), then draw registration marks on TOP of
    the padded canvas using that same offset -- marks computed in the
    original (un-padded) W x H frame land outside [0,W]x[0,H] for a
    positive dist, which is exactly the padded margin this just added.
    Returns the padded image unchanged if there's no margin to add.
    `mark_color`: BGR -- black for the combined/multi-colour render, or a
    single layer's own colour for that layer's own file."""
    if margin_px <= 0:
        if reg_marks:
            from .registration_marks import draw_marks_cv
            draw_marks_cv(img, W, H, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape,
                          color=mark_color)
        return img
    ch = img.shape[2]
    Wp, Hp = W + 2 * margin_px, H + 2 * margin_px
    padded = np.zeros((Hp, Wp, ch), dtype=np.uint8)
    if alpha:
        pass  # transparent margin — leave at 0
    else:
        padded[:, :] = canvas_bgr if ch == 3 else (*canvas_bgr, 255)
    padded[margin_px:margin_px + H, margin_px:margin_px + W] = img
    if reg_marks:
        from .registration_marks import draw_marks_cv
        draw_marks_cv(padded, W, H, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape,
                      offset=(margin_px, margin_px), color=mark_color)
    return padded


def _draw_layer(img, layer_idx, all_xs, all_ys, all_rs, all_ci, cells,
                color_bgr, mode, eq_scale_fn, smooth, inset_px,
                gradient=False, point_bgr=None, neighbors=None,
                tiles=None, tile_layer_idx=None, canvas_bgr=None, grout_width=1):
    """Draw every shape whose colour == layer_idx onto image `img`.
    `img` may be 3-channel BGR (opaque) or 4-channel BGRA (transparent background —
    composited premultiplied; caller un-premultiplies at the end)."""
    ch = img.shape[2]
    col = (color_bgr + (255,)) if ch == 4 else color_bgr
    sel = np.where(all_ci == layer_idx)[0]
    if mode == 'tile' and tiles is not None:
        # Tiles aren't 1:1 with points (row-packed, not per-circle) — select by
        # tile_layer_idx instead of `sel`.
        grout_col = (canvas_bgr + (255,)) if (ch == 4 and canvas_bgr is not None) else canvas_bgr
        # Row-packed tile width varies a lot locally (deliberately shrinking toward
        # zero near row convergence points), so the geometric inset gap between two
        # adjacent tiles can end up sub-pixel there even though it's never actually
        # zero — a real but visually imperceptible grout gap reads as "this tile is
        # cutting into its neighbour" when two similarly coloured tiles sit right up
        # against each other. Stroking each tile's OWN (already inset) boundary in
        # the canvas colour, on top of its fill, guarantees a minimum ~1px visible
        # seam around every tile regardless of how thin the true geometric gap is —
        # the same way real mosaic grout lines are always visible even between
        # same-coloured tiles.
        for i in np.where(tile_layer_idx == layer_idx)[0]:
            pchunk = _tile_inset_points(tiles[i], inset_px, smooth)
            if pchunk is None or len(pchunk) < 3:
                continue
            pts_i = np.round(pchunk).astype(np.int32)
            cv2.fillPoly(img, [pts_i], col, cv2.LINE_AA)
            if grout_col is not None:
                cv2.polylines(img, [pts_i], True, grout_col, max(1, int(round(grout_width))), cv2.LINE_AA)
        return
    if mode == 'polygon' and cells is not None:
        if gradient and point_bgr is not None and neighbors is not None:
            _draw_gradient_polygon_layer(img, sel, all_xs, all_ys, cells, smooth, inset_px,
                                         point_bgr, neighbors)
            return
        for i in sel:
            poly = cells[i]
            if poly is None:
                continue
            pchunk = _rounded_cell_points(poly, smooth, inset_px)
            if pchunk is None or len(pchunk) < 3:
                continue
            cv2.fillPoly(img, [np.round(pchunk).astype(np.int32)], col, cv2.LINE_AA)
    elif mode == 'airbrush':
        H, W = img.shape[:2]
        acc = np.zeros((H, W), dtype=np.float32)   # coverage for this layer
        for step_i, i in enumerate(sel):
            if step_i % _YIELD_EVERY == 0:
                time.sleep(0)
            br = int(round(eq_scale_fn(all_rs[i])))
            if br < 1:
                br = 1
            st = _gaussian_stamp(br)
            x = int(round(all_xs[i])); y = int(round(all_ys[i]))
            x0, x1 = x - br, x + br + 1
            y0, y1 = y - br, y + br + 1
            sx0 = max(0, -x0); sy0 = max(0, -y0)
            ix0, iy0 = max(0, x0), max(0, y0)
            ix1, iy1 = min(W, x1), min(H, y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            sub = st[sy0:sy0 + (iy1 - iy0), sx0:sx0 + (ix1 - ix0)]
            # Screen/OR-combine overlapping splats (1-(1-a)(1-b)) instead of max().  max()
            # leaves a visible dip in coverage exactly between two dot centres — neither
            # Gaussian has decayed to zero there, but neither has reached full strength
            # either, so the boundary between splats never fills in — producing a
            # honeycomb/cell pattern instead of smooth continuous airbrush coverage.
            region = acc[iy0:iy1, ix0:ix1]
            region[:] = 1.0 - (1.0 - region) * (1.0 - sub)
        a = acc[:, :, None]
        colf = np.array(color_bgr, dtype=np.float32)
        f = img.astype(np.float32)
        f[:, :, :3] = f[:, :, :3] * (1 - a) + colf * a
        if ch == 4:
            f[:, :, 3:4] = f[:, :, 3:4] * (1 - a) + 255.0 * a
        img[:] = f.astype(np.uint8)
    else:  # circle
        for i in sel:
            cv2.circle(img, (int(round(all_xs[i])), int(round(all_ys[i]))),
                       max(1, int(round(all_rs[i]))), col, -1, cv2.LINE_AA)


def _draw_airbrush_composite(img, all_xs, all_ys, all_rs, all_ci, layer_bgr, eq_scale_fn,
                             layer_ids=None):
    """Airbrush composite in one size-ordered pass: largest splats drawn first (bottom),
    smallest last (top).  Drawing strictly by palette/layer index (as _draw_layer does
    per-colour) means a later-indexed colour's big wash can fully bury an earlier-indexed
    colour's small detail dot wherever they overlap, regardless of which one should read
    as "on top" — size order fixes that: a small dot's colour is never swallowed by a
    larger splat from another colour.  `layer_ids`, if given, restricts to those colours
    (used for isolated per-layer renders, where it's just a same-colour size sort)."""
    H, W = img.shape[:2]
    ch = img.shape[2]
    idxs = np.arange(len(all_xs)) if layer_ids is None else np.where(np.isin(all_ci, list(layer_ids)))[0]
    order = idxs[np.argsort(-all_rs[idxs])]  # largest → smallest
    for step_i, i in enumerate(order):
        if step_i % _YIELD_EVERY == 0:
            time.sleep(0)
        br = int(round(eq_scale_fn(all_rs[i])))
        if br < 1:
            br = 1
        st = _gaussian_stamp(br)
        x = int(round(all_xs[i])); y = int(round(all_ys[i]))
        x0, x1 = x - br, x + br + 1
        y0, y1 = y - br, y + br + 1
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(W, x1), min(H, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        sub = st[sy0:sy0 + (iy1 - iy0), sx0:sx0 + (ix1 - ix0), None]
        col = np.asarray(layer_bgr[all_ci[i]], dtype=np.float32)
        region = img[iy0:iy1, ix0:ix1, :3].astype(np.float32)
        region[:] = region * (1 - sub) + col * sub
        img[iy0:iy1, ix0:ix1, :3] = region.astype(np.uint8)
        if ch == 4:
            alpha_region = img[iy0:iy1, ix0:ix1, 3:4].astype(np.float32)
            alpha_region[:] = alpha_region * (1 - sub) + 255.0 * sub
            img[iy0:iy1, ix0:ix1, 3:4] = alpha_region.astype(np.uint8)


def render_mosaic_bgr(
    layer_results,          # list of (circles, hex, name) — circles in working px
    canvas_hex: str,
    working_w: int,
    working_h: int,
    out_w: int,
    out_h: int,
    mode: str = 'circle',
    r_min: float = 1.0,
    r_max: float = 10.0,
    eq_curve=None,
    poly_inset: float = 0.08,
    poly_smooth: float = 0.0,
    poly_gradient: bool = False,
    alpha: bool = False,
    reg_marks: bool = False,
    reg_mark_dist_px: float = 0.0,   # already scaled to this frame's own out_w/out_h
    reg_mark_size_px: float = 0.0,
    reg_mark_shape: str = 'circle',
) -> np.ndarray:
    """Render one mosaic to a uint8 array (for video frames).  Scales working-pixel
    geometry to the exact output resolution.  alpha=False → (H, W, 3) BGR on the canvas
    colour; alpha=True → (H, W, 4) BGRA with a transparent background.

    KNOWN GAP: unlike write_raster_outputs, this does NOT pad the canvas for
    outward (positive-dist) registration marks -- every frame in a video
    sequence must share one fixed resolution, decided by the caller before
    encoding starts, so padding would need to happen at the video pipeline
    level (video_render.py), not silently here per-frame. A positive-dist
    mark on a video frame will currently draw partially or fully off-canvas
    and get clipped."""
    sx = out_w / float(working_w)
    sy = out_h / float(working_h)
    scale = (sx + sy) * 0.5   # radii scale by the average (dims are kept in aspect upstream)

    xs, ys, rs, ci = [], [], [], []
    for li, (circles, _hexc, _name) in enumerate(layer_results):
        for (x, y, r) in circles:
            xs.append(x * sx); ys.append(y * sy); rs.append(r * scale); ci.append(li)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    ci = np.asarray(ci, dtype=np.int32)

    cells = None
    point_bgr = neighbors = None
    tiles = tile_layer_idx = None
    if mode == 'polygon' and len(xs) >= 3:
        cells = _voronoi_polygons(xs, ys, out_w, out_h)
        if poly_gradient:
            layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
            point_bgr = [layer_bgr[c] for c in ci]
            neighbors = _delaunay_neighbors(xs, ys, out_w, out_h)
    elif mode == 'tile' and len(xs) >= 3:
        layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
        point_bgr_arr = np.asarray([layer_bgr[c] for c in ci], dtype=np.float32)
        tiles, tile_pt_idx = _flow_tiles(xs, ys, rs, point_bgr_arr, out_w, out_h)
        tile_layer_idx = ci[np.asarray(tile_pt_idx, dtype=np.int64)] if tiles else np.array([], dtype=np.int32)
    inset_px = poly_inset * (r_min + r_max) * 0.5 * scale
    eqc = eq_curve or [1.2] * 5

    def eq_scale_fn(r_px):
        r_phys = r_px / scale
        rng = max(r_max - r_min, 1e-6)
        t = (r_phys - r_min) / rng
        n = len(eqc)
        pos = t * (n - 1)
        lo = int(np.clip(np.floor(pos), 0, n - 1))
        hi = int(np.clip(lo + 1, 0, n - 1))
        f = pos - lo
        return r_px * (eqc[lo] * (1 - f) + eqc[hi] * f)

    canvas_bgr = tuple(reversed(_hex_rgb(canvas_hex)))
    if alpha:
        img = np.zeros((out_h, out_w, 4), dtype=np.uint8)   # transparent background
    else:
        img = np.empty((out_h, out_w, 3), dtype=np.uint8)
        img[:] = canvas_bgr
    if mode == 'airbrush':
        # Composite ALL colours together in one SIZE-ordered pass (largest first, smallest
        # last/on top) — drawing strictly by palette/layer index would let a later colour's
        # large splat fully bury an earlier colour's small detail dot wherever they overlap.
        layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
        _draw_airbrush_composite(img, xs, ys, rs, ci, layer_bgr, eq_scale_fn)
    else:
        for li, (_c, hexc, _n) in enumerate(layer_results):
            _draw_layer(img, li, xs, ys, rs, ci, cells,
                        tuple(reversed(_hex_rgb(hexc))), mode, eq_scale_fn, poly_smooth, inset_px,
                        gradient=poly_gradient, point_bgr=point_bgr, neighbors=neighbors,
                        tiles=tiles, tile_layer_idx=tile_layer_idx, canvas_bgr=canvas_bgr)
    if alpha:
        # Un-premultiply so straight-alpha PNGs don't get dark edge fringing.
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            rgb = np.where(a > 0, np.clip(rgb / np.maximum(a, 1e-6), 0, 255), 0)
        img[:, :, :3] = rgb.astype(np.uint8)
    if reg_marks:
        from .registration_marks import draw_marks_cv
        draw_marks_cv(img, out_w, out_h, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape)
    return img


def write_raster_outputs(
    layer_results,          # list of (circles, hex, name)
    canvas_hex: str,
    output_dir: str,
    working_w: int,
    working_h: int,
    long_px: int,
    fmt: str,               # 'png' | 'jpeg' | 'tiff'
    mode: str = 'circle',
    r_min: float = 1.0,
    r_max: float = 10.0,
    eq_curve=None,
    poly_inset: float = 0.08,
    poly_smooth: float = 0.0,
    poly_gradient: bool = False,
    per_layer: bool = True,
    jpeg_quality: int = 92,
    reg_marks: bool = False,
    reg_mark_dist_px: float = 0.0,   # already scaled to this export's own output resolution
    reg_mark_size_px: float = 0.0,
    reg_mark_shape: str = 'circle',
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    long_px = int(max(64, min(20000, long_px)))
    scale = long_px / float(max(working_w, working_h))
    W = max(1, int(round(working_w * scale)))
    H = max(1, int(round(working_h * scale)))
    margin_px = _reg_margin_px(reg_marks, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape)

    # Flatten all layers → one point set (colour index = layer index).
    xs, ys, rs, ci = [], [], [], []
    for li, (circles, hexc, _name) in enumerate(layer_results):
        for (x, y, r) in circles:
            xs.append(x * scale); ys.append(y * scale); rs.append(r * scale); ci.append(li)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    ci = np.asarray(ci, dtype=np.int32)

    cells = None
    point_bgr = neighbors = None
    tiles = tile_layer_idx = None
    if mode == 'polygon' and len(xs) >= 3:
        cells = _voronoi_polygons(xs, ys, W, H)   # list of [[x,y],...] in output px
        if poly_gradient:
            # Neighbour RELATIONS are scale-independent, so this is computed once at
            # base scale and reused for every supersampled render() call below.
            layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
            point_bgr = [layer_bgr[c] for c in ci]
            neighbors = _delaunay_neighbors(xs, ys, W, H)
    elif mode == 'tile' and len(xs) >= 3:
        layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
        point_bgr_arr = np.asarray([layer_bgr[c] for c in ci], dtype=np.float32)
        tiles, tile_pt_idx = _flow_tiles(xs, ys, rs, point_bgr_arr, W, H)
        tile_layer_idx = ci[np.asarray(tile_pt_idx, dtype=np.int64)] if tiles else np.array([], dtype=np.int32)

    inset_px = poly_inset * (r_min + r_max) * 0.5 * scale
    eqc = eq_curve or [1.2] * 5
    canvas_bgr = tuple(reversed(_hex_rgb(canvas_hex)))
    paths = []

    # Supersample then downscale with INTER_AREA for smooth, round edges.  cv2's own
    # filled-circle antialiasing leaves flat spots at the top/bottom/sides; rendering at
    # SS× and area-averaging down removes them.
    SS = 3 if long_px <= 3000 else 2

    def _eq_fn(eff_scale):
        def fn(r_px):
            r_phys = r_px / eff_scale
            rng = max(r_max - r_min, 1e-6)
            t = (r_phys - r_min) / rng
            n = len(eqc)
            pos = t * (n - 1)
            lo = int(np.clip(np.floor(pos), 0, n - 1))
            hi = int(np.clip(lo + 1, 0, n - 1))
            f = pos - lo
            return r_px * (eqc[lo] * (1 - f) + eqc[hi] * f)
        return fn

    def render(layer_ids, alpha, use_gradient=False, mark_color=(0, 0, 0)):
        q = SS
        Wq, Hq = W * q, H * q
        xsq, ysq, rsq = xs * q, ys * q, rs * q
        cellsq = None
        if cells is not None:
            cellsq = [([(px * q, py * q) for px, py in c] if c is not None else None) for c in cells]
        tilesq = None
        if tiles is not None:
            tilesq = [[[px * q, py * q] for px, py in t] for t in tiles]
        insetq = inset_px * q
        eqfn = _eq_fn(scale * q)
        if alpha:
            img = np.zeros((Hq, Wq, 4), dtype=np.uint8)
        else:
            img = np.empty((Hq, Wq, 3), dtype=np.uint8)
            img[:] = canvas_bgr
        if mode == 'airbrush':
            layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
            _draw_airbrush_composite(img, xsq, ysq, rsq, ci, layer_bgr, eqfn, layer_ids=layer_ids)
        else:
            for li in layer_ids:
                hexc = layer_results[li][1]
                _draw_layer(img, li, xsq, ysq, rsq, ci, cellsq,
                            tuple(reversed(_hex_rgb(hexc))), mode, eqfn, poly_smooth, insetq,
                            gradient=(use_gradient and poly_gradient),
                            point_bgr=point_bgr, neighbors=neighbors,
                            tiles=tilesq, tile_layer_idx=tile_layer_idx,
                            canvas_bgr=canvas_bgr, grout_width=q)
        if q > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if alpha:
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = np.where(a > 0, np.clip(img[:, :, :3].astype(np.float32) / np.maximum(a, 1e-6), 0, 255), 0)
            img[:, :, :3] = rgb.astype(np.uint8)
        return _pad_and_mark(img, W, H, margin_px, canvas_bgr, alpha,
                             reg_marks, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape,
                             mark_color=mark_color)

    # Combined/base marks: black on a light canvas, white on a dark one (no single layer
    # colour applies to a multi-colour composite, and black vanishes on a dark canvas).
    if reg_marks:
        from .registration_marks import contrast_color
        combined_mark = tuple(reversed(_hex_rgb(contrast_color(canvas_hex))))
    else:
        combined_mark = (0, 0, 0)

    def render_full():
        # Gradient only makes sense on the full composite — a single-colour layer/page
        # can't meaningfully gradient into a colour it doesn't contain (same reasoning
        # the SVG per-layer export already follows).
        return render(range(len(layer_results)), alpha=False, use_gradient=True,
                      mark_color=combined_mark)

    def render_layer(li, hexc=None):
        # Marks on a per-layer file are drawn in THAT layer's own colour (not
        # the combined file's neutral black) — see write_layer_svg's matching
        # reasoning: for cutting/printing each colour from its own material
        # pass, the mark should come from that same pass.
        mark_color = tuple(reversed(_hex_rgb(hexc))) if hexc else (0, 0, 0)
        return render([li], alpha=False, mark_color=mark_color)

    def bgr_to_pil(img_bgr):
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def bgra_to_pil(img_bgra):
        return Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))

    if fmt == 'tiff':
        # Layered TIFF: an opaque canvas base + one TRANSPARENT layer per colour, so the
        # layers composite (stack) correctly in an editor instead of occluding each other
        # (that was why layers/dots looked "missing").
        base = np.empty((H + 2 * margin_px, W + 2 * margin_px, 4), np.uint8)
        base[:, :, :3] = canvas_bgr
        base[:, :, 3] = 255
        if reg_marks:
            from .registration_marks import draw_marks_cv
            draw_marks_cv(base, W, H, reg_mark_dist_px, reg_mark_size_px, reg_mark_shape,
                          offset=(margin_px, margin_px), color=combined_mark)
        pages = [bgra_to_pil(base)]
        for li, (_c, hexc, _n) in enumerate(layer_results):
            mark_color = tuple(reversed(_hex_rgb(hexc)))
            pages.append(bgra_to_pil(render([li], alpha=True, mark_color=mark_color)))
        tiff_path = str(out / 'mosaic.tiff')
        pages[0].save(tiff_path, format='TIFF', compression='tiff_lzw',
                      save_all=True, append_images=pages[1:])
        paths.append(tiff_path)
        return paths

    # PNG / JPEG → flattened composite (+ optional per-layer files).
    ext = 'png' if fmt == 'png' else 'jpg'
    full = bgr_to_pil(render_full())
    full_path = str(out / f'mosaic.{ext}')
    if fmt == 'png':
        full.save(full_path, format='PNG', optimize=True)
    else:
        full.convert('RGB').save(full_path, format='JPEG', quality=jpeg_quality, subsampling=0)
    paths.append(full_path)

    if per_layer:
        for li, (_c, hexc, _n) in enumerate(layer_results):
            lp = str(out / f"layer_{li+1:02d}_{hexc.lstrip('#').lower()}.{ext}")
            im = bgr_to_pil(render_layer(li, hexc))
            if fmt == 'png':
                im.save(lp, format='PNG', optimize=True)
            else:
                im.convert('RGB').save(lp, format='JPEG', quality=jpeg_quality, subsampling=0)
            paths.append(lp)
    return paths
