"""
Iterative circle optimizer — the "training loop" for the mosaic.

Concept (analogous to 2D Gaussian Splatting):
  - We have a set of circles. Each renders as a solid-color disk on the canvas.
  - The composited result should perceptually match the source image.
  - We iteratively move, resize, and recolor circles to minimize LAB error.

Algorithm: Weighted Voronoi Stippling (Secord 2002) extended for:
  - Multi-color palettes
  - Canvas background compensation
  - Non-overlap enforcement via radius rescaling
  - Densification (split large circles in high-error regions)
  - Pruning (remove circles in zero-contribution regions)

Each "iteration" = one Lloyd step:
  1. Voronoi: assign each pixel to nearest circle center (KD-tree)
  2. Update: move center to weighted centroid of its cell
  3. Resize: r ← √(cell_area × coverage_factor / π), clamped to [r_min, r_max]
  4. Recolor: reassign to dominant paint color in cell
  5. Resolve: push apart any circles that now overlap
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

# Caps the LIVE PREVIEW payload independent of the true circle count — the full,
# uncapped circle set always drives the real computation and the final export; this
# only limits what gets drawn during processing. Without it, a dense mosaic (tens or
# hundreds of thousands of circles) can produce preview JSON payloads tens of MB in
# size on every iteration, which is slow to build/serialize/render even with the
# websocket transport's message-size limit raised (see main.py) to actually allow it
# through — capping the preview keeps the UI responsive regardless of how large the
# real job gets, so "many circles" stays merely slow rather than a failure mode.
#
# Set generously (not tightly, like the original 40,000): the websocket transport can
# now comfortably carry a much larger payload (main.py raises the buffer limit to
# 200 MB), so the real remaining constraint is just "don't make the preview visibly
# gappy" — a uniform random subsample below true density reads as missing coverage,
# especially in already-sparser regions of the image, and was mistaken for an actual
# placement regression. 300k points keeps the preview visually complete for the vast
# majority of jobs while still bounding the truly extreme tail (500k+ circles).
_PREVIEW_MAX_POINTS = 300000


def _preview_subsample(xs, ys, rs, colors, max_points=_PREVIEW_MAX_POINTS):
    n = len(xs)
    if n <= max_points:
        return xs, ys, rs, colors
    idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    return xs[idx], ys[idx], rs[idx], colors[idx]


def _voronoi_polygons(xs, ys, W, H):
    """
    Compute Voronoi cell polygons for each point using Delaunay circumcenters.

    Each cell's polygon vertices are the circumcenters of the Delaunay triangles
    that contain that point, sorted by angle and clipped to [0,W]×[0,H].

    Because circumcenters are computed once and shared between adjacent cells,
    all polygon vertices at triple-junctions are mathematically identical —
    eliminating the floating-point gaps that arise from independent cell computation.

    Returns list of [[x,y],...] per point, or None for degenerate cells.
    """
    from scipy.spatial import Delaunay
    n = len(xs)
    if n < 3:
        return [None] * n

    pts = np.column_stack([xs, ys]).astype(float)

    # Far-field sentinels force all original cells to be bounded.
    margin = max(W, H) * 4.0
    sentinels = np.array([
        [-margin, H / 2], [W + margin, H / 2],
        [W / 2, -margin], [W / 2, H + margin],
        [-margin, -margin], [W + margin, -margin],
        [-margin, H + margin], [W + margin, H + margin],
    ], dtype=float)
    pts_aug = np.vstack([pts, sentinels])

    tri = Delaunay(pts_aug)

    # Circumcenter of triangle with vertices A, B, C.
    def circumcenter(A, B, C):
        ax, ay = A;  bx, by = B;  cx, cy = C
        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            return None
        a2 = ax*ax + ay*ay;  b2 = bx*bx + by*by;  c2 = cx*cx + cy*cy
        ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / D
        uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / D
        return (ux, uy)

    # Precompute circumcenter for every triangle.
    ccs = [circumcenter(pts_aug[s[0]], pts_aug[s[1]], pts_aug[s[2]])
           for s in tri.simplices]

    # Map each original point → its triangle indices.
    pt2tri = [[] for _ in range(n)]
    for ti, s in enumerate(tri.simplices):
        for v in s:
            if v < n:
                pt2tri[v].append(ti)

    # Sutherland-Hodgman clip against one half-plane nx*x + ny*y ≤ c (inside).
    def clip_hp(poly, nx, ny, c):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        pd = nx * prev[0] + ny * prev[1] - c
        for curr in poly:
            cd = nx * curr[0] + ny * curr[1] - c
            if (pd > 1e-9) != (cd > 1e-9):
                t = pd / (pd - cd)
                out.append((prev[0] + t * (curr[0] - prev[0]),
                             prev[1] + t * (curr[1] - prev[1])))
            if cd <= 1e-9:
                out.append(curr)
            prev, pd = curr, cd
        return out

    def clip_box(poly):
        poly = clip_hp(poly, -1,  0,  0)   # x >= 0
        poly = clip_hp(poly,  1,  0,  W)   # x <= W
        poly = clip_hp(poly,  0, -1,  0)   # y >= 0
        poly = clip_hp(poly,  0,  1,  H)   # y <= H
        return poly

    polygons = []
    for i in range(n):
        xi, yi = float(xs[i]), float(ys[i])
        t_idxs = pt2tri[i]
        if len(t_idxs) < 2:
            polygons.append(None)
            continue

        verts = [ccs[ti] for ti in t_idxs if ccs[ti] is not None]
        if len(verts) < 3:
            polygons.append(None)
            continue

        # Sort circumcenters by angle around circle center → Voronoi polygon in order.
        verts.sort(key=lambda p: np.arctan2(p[1] - yi, p[0] - xi))

        poly = clip_box(verts)
        if len(poly) >= 3:
            polygons.append([[float(p[0]), float(p[1])] for p in poly])
        else:
            polygons.append(None)

    return polygons


def _delaunay_neighbors(xs, ys, W=None, H=None):
    """Kept for gradient-mode edge color lookup (neighbor indices per circle)."""
    from scipy.spatial import Delaunay
    n = len(xs)
    if n < 3:
        return [[] for _ in range(n)]
    pts = np.column_stack([xs, ys]).astype(float)
    if W is not None and H is not None:
        margin = max(W, H) * 4.0
        extras = np.array([
            [-margin, H/2], [W+margin, H/2],
            [W/2, -margin], [W/2, H+margin],
        ], dtype=float)
        pts = np.vstack([pts, extras])
    tri = Delaunay(pts)
    nbrs = [set() for _ in range(n)]
    for s in tri.simplices:
        vs = [int(v) for v in s if int(v) < n]
        for v in vs:
            for u in vs:
                if u != v:
                    nbrs[v].add(u)
    return [sorted(s) for s in nbrs]

def _build_topography(xs, ys, rs, W, H, analysis_res=256):
    """The single source of truth for turning a point arrangement's circle
    size/distribution into a topographic elevation field — used by every
    tile-shaped consumer (contour tracer, mesh relaxation, anything else) so
    they're all reading the same terrain, not subtly-different reimplementations
    that could drift apart.

    Returns a dict: g, sx, sy, size_grid, elev, grad_mag, dir_x, dir_y (the
    gap-filled doubled-angle gradient direction components), plus the
    `sample_size`/`local_scale`/`contour_angle` closures bound to this field.
    """
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    g = max(16, int(analysis_res))
    sx, sy = g / float(W), g / float(H)
    tree = KDTree(np.column_stack([xs * sx, ys * sy]))
    gy, gx = np.mgrid[0:g, 0:g]
    grid_pts = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
    _, nearest = tree.query(grid_pts, k=1)
    # `rs[nearest]` is already in world-space units (real circle radius, in
    # pixels) — no rescaling needed. A stray `/ ((sx+sy)*0.5)` division here
    # previously inflated these values well past the clip bounds below
    # (which stayed in true world units), collapsing the ENTIRE grid to one
    # flat clip-ceiling value whenever the analysis grid was much coarser
    # than the canvas (any real, non-tiny image) — invisible in the old
    # row-tracing code (a constant target size just meant no detail-adaptive
    # sizing) but fatal for anything topography-based: a flat elevation
    # field has no gradient, so there's nothing to trace or relax against.
    size_grid = rs[nearest].reshape(g, g)
    _median_r = float(np.median(rs))
    # The biggest tile is never more than 3x the smallest — clip bounds
    # centred (geometrically) on the median so that ratio is exactly 3,
    # rather than the far wider ~10x range this used to allow.
    size_grid = np.clip(size_grid, _median_r / math.sqrt(3.0), _median_r * math.sqrt(3.0))
    # Smooth the size field itself (not just the derived elevation) so tile
    # size actually GRADATES across an area — without this, two neighbouring
    # grid cells backed by very different nearest-circle radii could hand out
    # a max-size tile right next to a min-size one, which reads as noisy
    # rather than the deliberate small-to-large sizing progression a real
    # mosaic has.
    size_grid = gaussian_filter(size_grid, sigma=max(2.0, g / 50.0))

    # A pure per-cell circle-DENSITY splat has too much high-frequency noise
    # at real (thousands-of-circles) scale — it over-segments into countless
    # tiny features. The circle-size signal (the same one that already
    # drives detail-adaptive tile sizing) is smoother and reflects the SAME
    # underlying distribution, so it's kept as the elevation source — but
    # smoothed at a much broader scale than a single-cell lookup, so nearby
    # detail merges into contiguous hills long enough to trace or relax a
    # mesh across, rather than a fragmented patchwork.
    _elev_sigma = max(2.0, g / 40.0)
    elev = gaussian_filter(1.0 / size_grid, sigma=_elev_sigma)
    gy_e, gx_e = np.gradient(elev)
    grad_mag = np.clip(gaussian_filter(np.hypot(gx_e, gy_e), 2.0), 1e-6, None)

    # In a genuinely flat region the gradient is ~zero, so atan2(0,0) picks
    # the SAME fixed axis everywhere — which reads as a rigid x/y grid
    # instead of following anything. Flat cells inherit the nearest REAL
    # (non-flat) direction instead (doubled-angle so the mod-pi axis
    # ambiguity averages/fills correctly).
    _flat = grad_mag < (grad_mag.max() * 0.03 if grad_mag.max() > 0 else 1.0)
    _doubled = np.exp(2j * np.arctan2(gy_e, gx_e))
    if _flat.any() and not _flat.all():
        _, (_iy, _ix) = distance_transform_edt(_flat, return_indices=True)
        _doubled = _doubled[_iy, _ix]
    dir_y, dir_x = np.imag(_doubled), np.real(_doubled)

    def sample_size(x, y):
        return max(1.5, float(size_grid[min(g - 1, max(0, int(y * sy))), min(g - 1, max(0, int(x * sx)))]))

    def contour_angle(x, y):
        gxp = min(max(int(round(x * sx)), 0), g - 1)
        gyp = min(max(int(round(y * sy)), 0), g - 1)
        grad_ang = 0.5 * math.atan2(dir_y[gyp, gxp], dir_x[gyp, gxp])
        return grad_ang + math.pi / 2   # perpendicular to gradient == along the contour

    def local_scale(x, y, curvature_gain=6.0, curv=0.0, min_tile_px=None, max_tile_px=None):
        gxp = min(max(int(round(x * sx)), 0), g - 1)
        gyp = min(max(int(round(y * sy)), 0), g - 1)
        grade_norm = grad_mag[gyp, gxp] / (grad_mag.max() + 1e-9)
        val = sample_size(x, y) * float(np.clip(1.15 - grade_norm, 0.3, 1.15))
        if curv > 0.0:
            curv_factor = float(np.clip(1.0 - curvature_gain * curv, 0.25, 1.0))
            val *= curv_factor
        if min_tile_px is not None:
            val = max(val, min_tile_px)
        if max_tile_px is not None:
            val = min(val, max_tile_px)
        return val

    return {
        'g': g, 'sx': sx, 'sy': sy, 'size_grid': size_grid, 'elev': elev,
        'grad_mag': grad_mag, 'dir_x': dir_x, 'dir_y': dir_y,
        'sample_size': sample_size, 'contour_angle': contour_angle, 'local_scale': local_scale,
    }


def _tile_make_rect(cx, cy, ang, half_w, half_h):
    tx, ty = math.cos(ang), math.sin(ang)
    nx, ny = -ty, tx
    return [
        [cx + tx * half_w + nx * half_h, cy + ty * half_w + ny * half_h],
        [cx - tx * half_w + nx * half_h, cy - ty * half_w + ny * half_h],
        [cx - tx * half_w - nx * half_h, cy - ty * half_w - ny * half_h],
        [cx + tx * half_w - nx * half_h, cy + ty * half_w - ny * half_h],
    ]


def _tile_fill_gaps(tiles, canvas, local_scale, contour_angle, color_tree, min_area_factor=0.0):
    """Compute whatever's currently uncovered and fill it with tiles, sized
    off the SURROUNDING already-placed tiles rather than the hyper-local
    target alone (a gap in an otherwise flat area was getting shredded into
    many tiny tiles whenever the hyper-local value happened to read small at
    that exact point — prefer matching what's already around a gap, only go
    small if the surroundings themselves are already small, i.e. real
    detail). Mutates `tiles` in place (appends new alive entries); returns
    how many were added, so a caller can loop until there's nothing left.

    `min_area_factor`: skip leftover pieces smaller than this fraction of
    the local tile's own area — independently-rotated neighbouring tiles
    always leave thin sliver gaps between them (they don't share edges the
    way a mesh would), and chasing every sliver with a new tile explodes
    tile count for almost no real coverage gain once de-overlap resolves
    the resulting pile of thousands of tiny mutually-conflicting slivers.
    Pass >0 once slivers have already been closed by growing the existing
    tiles, so this only fires for genuinely open areas."""
    from scipy.spatial import KDTree
    from shapely.geometry import Polygon, Point as ShPoint
    from shapely.ops import unary_union

    live = [t for t in tiles if t['alive']]
    polys = [Polygon(t['poly']) for t in live]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    covered = unary_union(polys) if polys else None
    leftover = canvas.difference(covered) if covered is not None else canvas
    pieces = list(leftover.geoms) if hasattr(leftover, 'geoms') else [leftover]

    live_centers = np.array([[t['cx'], t['cy']] for t in live]) if live else None
    live_sizes = np.array([t['half'] * 2 for t in live]) if live else None
    size_tree = KDTree(live_centers) if live_centers is not None and len(live_centers) > 0 else None

    n_added = 0
    for lp in pieces:
        if lp.geom_type != 'Polygon' or lp.is_empty or lp.area < 4:
            continue
        minx, miny, maxx, maxy = lp.bounds
        cx0, cy0 = (minx + maxx) / 2, (miny + maxy) / 2
        if min_area_factor > 0:
            local_area = local_scale(cx0, cy0) ** 2
            if lp.area < local_area * min_area_factor:
                continue
        gap_extent = max(maxx - minx, maxy - miny)
        if size_tree is not None:
            k = min(8, len(live_centers))
            d, idx = size_tree.query([cx0, cy0], k=k)
            idx = np.atleast_1d(idx)
            nearby_avg = float(np.mean(live_sizes[idx]))
        else:
            nearby_avg = local_scale(cx0, cy0)
        cell = max(local_scale(cx0, cy0), nearby_avg * 0.7)
        cell = max(min(cell, gap_extent), 2.0)
        gxw = minx
        while gxw < maxx:
            gyw = miny
            while gyw < maxy:
                ccx, ccy = gxw + cell / 2, gyw + cell / 2
                if lp.contains(ShPoint(ccx, ccy)):
                    ang = contour_angle(ccx, ccy)
                    half = cell / 2.0
                    _, ci_ = color_tree.query([ccx, ccy], k=1)
                    tiles.append({'poly': _tile_make_rect(ccx, ccy, ang, half, half),
                                  'cx': ccx, 'cy': ccy, 'ang': ang, 'half': half,
                                  'ci': int(ci_), 'alive': True})
                    n_added += 1
                gyw += cell
            gxw += cell
    return n_added


def _flow_tiles(xs, ys, rs, colors_bgr, W, H, analysis_res=256,
                 min_tile_px=None, max_tile_px=None, curvature_gain=6.0,
                 n_iters=20):
    """
    Place single-size-varying SQUARE tiles, rotated to follow the local
    contour direction, packed via a continuous physics-style dynamic
    (surface-tension-like attraction to close gaps + continuous resize
    toward the same detail-adaptive target circle sizing uses, sampled at
    each tile's CURRENT position + exact SAT/MTV repulsion for zero overlap)
    with a final corner-growth pass that stretches settled tiles into any
    remaining void. See core/tile_square.py for the full implementation —
    this function is a thin adapter preserving the original signature/return
    contract every caller (pipeline preview, raster export) depends on.

    Derived entirely from the existing point arrangement (positions, sizes,
    colours) — no separate source image needed, so this works identically for
    still images and video frames, using the same data every other mode
    already has by this point in the pipeline.

    Returns (tiles, tile_color_idx):
      - tiles: list of polygons ([[x,y],...], 3+ points each)
      - tile_color_idx: parallel list — the ORIGINAL point index each tile
        takes its colour from (nearest point to the tile's centre)
    """
    from .tile_cvd import flow_square_tiles_cvd as flow_square_tiles

    n = len(xs)
    if n == 0:
        return [], []

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)

    if n < 3:
        tiles = [[[xs[i] - rs[i], ys[i] - rs[i]], [xs[i] + rs[i], ys[i] - rs[i]],
                  [xs[i] + rs[i], ys[i] + rs[i]], [xs[i] - rs[i], ys[i] + rs[i]]]
                 for i in range(n)]
        return tiles, list(range(n))

    topo = _build_topography(xs, ys, rs, W, H, analysis_res)
    r_min = float(min_tile_px) if min_tile_px is not None else float(rs.min())
    r_max = float(max_tile_px) if max_tile_px is not None else float(rs.max())
    if r_max <= r_min:
        r_max = r_min + 1e-3

    # n_iters here is real Lloyd relaxation iteration count (CVD is the
    # active tile algorithm), NOT the old row-tracing-era "iterations" that
    # this `max(50, n_iters * 10)` scaling was written for -- row-tracing
    # ignored n_iters entirely (it's deterministic), so that x10 multiplier
    # was silently dead code the whole time it was in effect. Once CVD
    # became active again it started being taken literally: with the
    # default n_iters=20 that's max(50, 200) = 200 real Lloyd iterations,
    # and since edge-avoidance masking (70% of iterations, see
    # `_lloyd_relax`) blocks the early-convergence check from even
    # activating until it lifts, that floor forced at least 140 full
    # relaxation passes regardless of how quickly the layout had actually
    # converged -- measured directly: _lloyd_relax alone took 77s at
    # n_iters=200 on a 10k-site case that converges in 12s at n_iters=30,
    # and was the dominant cost of a run reported as still showing the
    # stale circle preview after "running in the background". 30 is what
    # this session's own CVD testing/tuning was actually done against.
    polys, src_idx = flow_square_tiles(xs, ys, rs, W, H, topo, r_min, r_max,
                                        n_iters=30,
                                        colors_bgr=colors_bgr)
    if not polys:
        return [], []
    # flow_square_tiles_cvd already returns plain point lists; the older
    # (still-present, fallback-only) row-tracing path returns shapely
    # Polygons -- accept either.
    if hasattr(polys[0], 'exterior'):
        tiles = [[[float(x), float(y)] for x, y in p.exterior.coords[:-1]] for p in polys]
    else:
        tiles = polys
    return tiles, list(src_idx)


def compute_preview_tiles(xs, ys, rs, ci, palette_hex, W, H):
    """Tile geometry for the LIVE PREVIEW (pipeline.py's final-phase callback,
    and the on-demand re-preview when the render-mode toggle is switched to
    'tile' after a completed run) -- and, deliberately, identical to what
    export computes (raster_export.py/svg_writer.py/gcode_export.py all call
    `_flow_tiles` the same way). Two earlier attempts at a cheaper
    approximation -- downscaling the canvas and either (a) uniformly
    rescaling tiles back up, which inflated their size ~3x past the real
    export size, or (b) rescaling only their centroid, which left the
    full-canvas preview covering ~6% of the area instead of the real ~65% --
    both produced a preview that visibly disagreed with the export. Given
    that choice, correctness was picked over speed: this is the same
    (currently uncached, tens of seconds to minutes on a large piece) full
    computation export uses, run once per completed generation or mode
    switch rather than per iteration. Returns (tiles, tile_color_idx) in the
    W x H coordinate frame -- same contract as `_flow_tiles`.
    """
    from .raster_export import _hex_rgb
    if len(xs) < 3:
        return [], []
    colors_bgr = np.array([tuple(reversed(_hex_rgb(palette_hex[c]))) for c in ci], dtype=np.float32)
    tiles, tile_pt_idx = _flow_tiles(xs, ys, rs, colors_bgr, W, H)
    tile_color_idx = [int(ci[i]) for i in tile_pt_idx]
    return tiles, tile_color_idx


def _clip_halfplane(poly, origin, normal):
    """Keep only the part of `poly` on the side of the line through `origin`
    perpendicular to `normal` where dot(p - origin, normal) <= 0. A single
    straight-line (Sutherland-Hodgman one-edge) clip of a simple polygon can
    only ever trim it along that one line — it can't carve a concave notch
    into it the way subtracting an arbitrarily-shaped neighbour can."""
    from shapely.geometry import Polygon
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 3:
        return None
    ox, oy = origin
    nx_, ny_ = normal

    def side(p):
        return (p[0] - ox) * nx_ + (p[1] - oy) * ny_

    out = []
    for i in range(n):
        cur = coords[i]
        nxt = coords[(i + 1) % n]
        cur_s, nxt_s = side(cur), side(nxt)
        if cur_s <= 0:
            out.append(cur)
        if (cur_s <= 0) != (nxt_s <= 0):
            t = cur_s / (cur_s - nxt_s)
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    if len(out) < 3:
        return None
    try:
        p = Polygon(out)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            return None
        if p.geom_type != 'Polygon':
            parts = [g for g in p.geoms if g.geom_type == 'Polygon']
            if not parts:
                return None
            p = max(parts, key=lambda g: g.area)
        return p
    except Exception:
        return None


def _deoverlap_tiles(candidates, min_area, seed_polys=None):
    """Enforce strict non-overlap on flow-row tile candidates: process in order,
    and drop (never reshape) any candidate that meaningfully overlaps an
    already-placed tile. A real tessera has 3 or 4 straight sides and no
    concave bite — so a conflicting tile is rejected whole rather than clipped
    or carved into some other shape. Instead, a conflicting tile is SPLIT —
    pulled and replaced with two smaller pieces (cut at the midpoints of a pair
    of opposite edges, so a quad becomes two quads and a triangle becomes two
    triangles — never more than 4 sides) — and each half is retried the same
    way, recursively, until every piece either fits cleanly or is too small to
    be worth placing. This is what an actual mosaicist does with a tile that
    doesn't fit a gap: cut it down, not force it in. The independent per-row
    height calculation in `_flow_tiles` is only an ESTIMATE of the gap to the
    nearest other row — real rows curve and don't stay perfectly parallel, so
    that estimate alone still lets candidates overlap in practice; this is the
    actual guarantee.

    `seed_polys`: already-placed tiles from an earlier pass to check candidates
    against (and never re-emit) — lets a later batch (e.g. gap-filler pieces)
    respect an earlier batch's tiles without redoing that earlier batch's work."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    placed_polys = list(seed_polys) if seed_polys else []
    tree = None
    indexed_count = 0     # how many of placed_polys[:indexed_count] the tree covers
    rebuild_every = 40
    MAX_SPLIT_DEPTH = 4

    out_tiles, out_colors = [], []

    def _split_piece(poly_pts):
        n = len(poly_pts)
        if n == 3:
            edges = [(poly_pts[0], poly_pts[1]), (poly_pts[1], poly_pts[2]), (poly_pts[2], poly_pts[0])]
            lens = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in edges]
            li = lens.index(max(lens))
            p0, p1 = edges[li]
            opp = poly_pts[(li + 2) % 3]
            mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
            return [[list(p0), mid, list(opp)], [mid, list(p1), list(opp)]]
        elif n == 4:
            p0, p1, p2, p3 = poly_pts
            len_a = math.hypot(p0[0] - p1[0], p0[1] - p1[1]) + math.hypot(p2[0] - p3[0], p2[1] - p3[1])
            len_b = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) + math.hypot(p3[0] - p0[0], p3[1] - p0[1])
            if len_a >= len_b:
                m01 = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
                m23 = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2]
                return [[list(p0), m01, m23, list(p3)], [m01, list(p1), list(p2), m23]]
            else:
                m12 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
                m30 = [(p3[0] + p0[0]) / 2, (p3[1] + p0[1]) / 2]
                return [[list(p0), list(p1), m12, m30], [m30, m12, list(p2), list(p3)]]
        return None

    def _place(poly_pts, ci_, depth):
        nonlocal tree, indexed_count
        try:
            cand = Polygon(poly_pts)
        except Exception:
            return
        if not cand.is_valid:
            cand = cand.buffer(0)
        if cand.is_empty or cand.geom_type != 'Polygon' or cand.area < min_area:
            return

        nearby = []
        if placed_polys:
            if tree is None or len(placed_polys) - indexed_count >= rebuild_every:
                tree = STRtree(placed_polys)
                indexed_count = len(placed_polys)
            nearby = [placed_polys[i] for i in tree.query(cand)]
            # The STRtree only covers placed_polys[:indexed_count] — anything
            # placed since the last rebuild wouldn't be checked without this,
            # silently letting a tile overlap its most recent neighbours (the
            # rebuild batching is a performance optimisation; correctness can't
            # depend on how often it happens to trigger).
            nearby.extend(placed_polys[indexed_count:])

        # If cand conflicts with exactly ONE neighbour, a single straight cut
        # through the midpoint of their centres (keeping cand's own side)
        # resolves it while losing far less area than dropping/splitting the
        # whole tile — and since it's one straight line through a quad or
        # triangle, the result never grows past 4 sides. Multi-neighbour
        # conflicts fall straight to recursive splitting instead of chasing a
        # multi-cut resolution (that path is a lot of extra work for a case
        # that's already the exception, not the common one).
        # Threshold scales with the CANDIDATE's own area, not just the fixed
        # (tiny, for small fill squares) min_area floor — a small filler tile
        # touching a diagonal row-tile edge routinely has a hairline numerical
        # sliver of "overlap" from floating point alone, which a fixed tiny
        # threshold was flagging as a real conflict and rejecting/splitting
        # over, when it wasn't visually or geometrically meaningful.
        conflict_tol = max(min_area * 0.1, cand.area * 0.02)
        conflicts = [_np for _np in nearby if cand.intersection(_np).area > conflict_tol]
        if not conflicts:
            coords = list(cand.exterior.coords)[:-1]
            if len(coords) < 3 or not _is_convex(coords):
                return
            out_tiles.append([[float(x), float(y)] for x, y in coords])
            out_colors.append(ci_)
            placed_polys.append(cand)
            return

        if 1 <= len(conflicts) <= 6:
            # A small filler square dropped into a leftover gap commonly
            # touches several differently-angled neighbours at once (they
            # aren't all parallel like a row's own tiles are) — resolving
            # only a handful of conflict cases here was rejecting/splitting
            # the large majority of candidates outright at real (thousands
            # of tiles) density, losing far more true area than necessary
            # (a raw union covering ~97% of the canvas was coming out only
            # ~64% covered after this pass). Cut against each conflict in
            # turn (worst overlap first); each cut is one straight line, so
            # it can only ever remove a vertex or add one — bail to
            # splitting instead of dropping the whole tile if a cut would
            # ever push past 4 sides.
            cur = cand
            ordered = sorted(conflicts, key=lambda _np: cand.intersection(_np).area, reverse=True)
            ok = True
            for worst in ordered:
                wc = worst.centroid
                cc = cur.centroid
                dx, dy = wc.x - cc.x, wc.y - cc.y
                dlen = math.hypot(dx, dy)
                if dlen <= 1e-9:
                    ok = False
                    break
                mid = (cc.x + dx * 0.5, cc.y + dy * 0.5)
                cut = _clip_halfplane(cur, mid, (dx / dlen, dy / dlen))
                if cut is None or len(list(cut.exterior.coords)) - 1 > 4 or cut.area < min_area:
                    ok = False
                    break
                cur = cut
            coords = list(cur.exterior.coords)[:-1]
            if ok and _is_convex(coords) and not any(cur.intersection(_np).area > min_area * 0.1 for _np in nearby):
                out_tiles.append([[float(x), float(y)] for x, y in coords])
                out_colors.append(ci_)
                placed_polys.append(cur)
                return

        if depth >= MAX_SPLIT_DEPTH:
            return   # too small to keep cutting down — leave it as a gap
        parts = _split_piece(poly_pts)
        if not parts:
            return
        for part in parts:
            _place(part, ci_, depth + 1)

    for poly_pts, ci_ in candidates:
        _place(poly_pts, ci_, 0)

    return out_tiles, out_colors


def _is_convex(poly):
    """True if every turn around `poly` has the same sign (no reflex/concave
    vertex) — a real cut tessera has straight, uncarved sides, so any piece
    this pipeline places should pass this."""
    n = len(poly)
    if n < 3:
        return False
    sign = 0.0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cx, cy = poly[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            continue
        s = 1.0 if cross > 0 else -1.0
        if sign == 0.0:
            sign = s
        elif s != sign:
            return False
    return True


def _clip_convex(subject, clip_poly):
    """Sutherland-Hodgman: clip `subject` (any simple polygon) against `clip_poly`
    (must be CONVEX, e.g. a Voronoi cell — every cell from `_voronoi_polygons` is
    convex by construction). Returns the clipped point list, possibly empty."""
    out = subject
    n = len(clip_poly)
    for k in range(n):
        if not out:
            return out
        ax, ay = clip_poly[k]
        bx, by = clip_poly[(k + 1) % n]
        # Inward normal for a counter/clockwise-agnostic convex polygon: points with
        # a negative cross product relative to this edge are "inside" — consistent
        # regardless of winding as long as it's used the same way for every edge.
        ex, ey = bx - ax, by - ay
        nx, ny = -ey, ex   # left-hand normal of the edge direction

        def inside(p, ax=ax, ay=ay, nx=nx, ny=ny):
            return (p[0] - ax) * nx + (p[1] - ay) * ny >= -1e-9

        new_out = []
        prev = out[-1]
        prev_in = inside(prev)
        for curr in out:
            curr_in = inside(curr)
            if curr_in != prev_in:
                px, py = prev; cx2, cy2 = curr
                dx, dy = cx2 - px, cy2 - py
                denom = dx * nx + dy * ny
                t = ((ax - px) * nx + (ay - py) * ny) / denom if abs(denom) > 1e-12 else 0.0
                new_out.append([px + t * dx, py + t * dy])
            if curr_in:
                new_out.append(curr)
            prev, prev_in = curr, curr_in
        out = new_out
    return out


try:
    from scipy.spatial import KDTree
    from scipy.ndimage import label as ndlabel
except ImportError:
    raise ImportError("scipy is required for the optimizer")

try:
    import torch
    _TORCH_DEVICE = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )
    _USE_GPU = True
    print(f"[optimizer] GPU acceleration active ({_TORCH_DEVICE}) — error map + bincount on GPU", flush=True)
except ImportError:
    _USE_GPU = False
    _TORCH_DEVICE = None




@dataclass
class OptimizerConfig:
    r_min: float             # working pixels
    r_max: float
    gap: float               # min gap between circle edges
    coverage_factor: float = 0.65   # target fraction of area covered (affects radius scaling)
    n_iterations: int = 10
    densify_error_pct: float = 85.0  # percentile threshold for splitting
    prune_min_weight: float = 0.01   # remove circles whose cell has < this mean weight
    lloyd_smoothing: float = 0.6     # blend factor: new_pos = α*centroid + (1-α)*old_pos
    dither_strength: float = 0.7     # 0=no optical mixing, 1=full error diffusion
    blend_radius: float = 3.5        # neighbor search radius as multiple of circle radius
    lum_influence: float = 0.5       # 0=detail-only sizing, 1=luminosity-only sizing
    allow_restructure: bool = True   # densify/prune between Lloyd steps.  Set False for
                                     # video warm-start so the dot COUNT stays fixed and
                                     # each dot forms a continuous per-frame trajectory.
    dither_kernel: str = 'ball'      # error-diffusion kernel: 'sierra' | 'ball'


class CircleOptimizer:

    def __init__(
        self,
        circles: list[tuple[float, float, float]],
        color_indices: np.ndarray,
        source_lab: np.ndarray,            # (H, W, 3)
        weight_maps: np.ndarray,           # (N_colors, H, W)
        detail_map: np.ndarray,            # (H, W)
        palette_lab: np.ndarray,           # (N_colors, 3)
        canvas_lab: np.ndarray,            # (3,)
        config: OptimizerConfig,
        target_lab: np.ndarray = None,     # canvas-compensated LAB to recolour toward
    ):
        self.xs = np.array([c[0] for c in circles], dtype=np.float32)
        self.ys = np.array([c[1] for c in circles], dtype=np.float32)
        self.rs = np.array([c[2] for c in circles], dtype=np.float32)
        self.colors = color_indices.astype(np.int32)

        self.src_lab = source_lab.astype(np.float32)
        # Colours are chosen to match this target; canvas compensation shifts it away from
        # the raw image so paint offsets the canvas tint.  Falls back to the raw image.
        self.target_lab = (target_lab.astype(np.float32)
                           if target_lab is not None else self.src_lab)
        # Keep source RGB for side-by-side preview (convert from LAB)
        import cv2 as _cv2
        lab_u8 = source_lab.copy()
        lab_u8[:,:,0] = (lab_u8[:,:,0] * 255.0 / 100.0).clip(0,255)
        lab_u8[:,:,1] = (lab_u8[:,:,1] + 128.0).clip(0,255)
        lab_u8[:,:,2] = (lab_u8[:,:,2] + 128.0).clip(0,255)
        rgb_u8 = _cv2.cvtColor(lab_u8.astype(np.uint8), _cv2.COLOR_LAB2RGB)
        self._source_rgb = rgb_u8.astype(np.float32) / 255.0
        self.weight_maps = weight_maps
        self.detail = detail_map.astype(np.float32)
        self.palette_lab = palette_lab.astype(np.float32)
        self.canvas_lab = canvas_lab.astype(np.float32)
        self.cfg = config

        self.H, self.W = source_lab.shape[:2]

        # Pixel coordinate grids (float32, reused every iteration)
        ys_g, xs_g = np.mgrid[0:self.H, 0:self.W]
        self.px_grid = xs_g.astype(np.float32)   # (H, W)
        self.py_grid = ys_g.astype(np.float32)

        # Pre-upload static arrays to GPU once — avoids per-iteration transfers
        self._dev = _TORCH_DEVICE if _USE_GPU else None
        if _USE_GPU:
            import torch as _torch
            dev = self._dev
            self._t_src_lab   = _torch.from_numpy(self.src_lab).to(dev)          # (H,W,3)
            self._t_palette   = _torch.from_numpy(self.palette_lab).to(dev)       # (C,3)
            self._t_canvas    = _torch.from_numpy(self.canvas_lab).to(dev)        # (3,)
            self._t_wt_max    = _torch.from_numpy(weight_maps.max(axis=0).astype(np.float32)).to(dev)  # (H,W)
            self._t_detail    = _torch.from_numpy(detail_map.astype(np.float32)).to(dev)               # (H,W)
            self._t_flat_px   = _torch.from_numpy(self.px_grid.ravel()).to(dev)   # (H*W,)
            self._t_flat_py   = _torch.from_numpy(self.py_grid.ravel()).to(dev)   # (H*W,)
            # Dominant per-pixel color is weight_maps.argmax(axis=0) — constant
            n_c = weight_maps.shape[0]
            dom_px = weight_maps.reshape(n_c, -1).argmax(axis=0).astype(np.int64)
            self._t_dominant_px = _torch.from_numpy(dom_px).to(dev)              # (H*W,)
            self._t_ones = _torch.ones(self.H * self.W, device=dev)              # for scatter counts

    # ── Public ──────────────────────────────────────────────────────────────

    def run(
        self,
        progress_cb: Optional[Callable[[int, float], None]] = None,
        preview_cb: Optional[Callable[[str], None]] = None,
        palette_hex: Optional[list[str]] = None,
        canvas_hex: str = '#FFFFFF',
        mode: str = 'circle',
        stop_event=None,
    ):
        """
        Run all configured iterations.
        progress_cb(iter_number, error_score)
        preview_cb(data_url_string)  — called after each iteration with a JPEG preview
        """
        # NOTE: earlier versions snapshotted whatever mid-loop iteration had the lowest
        # raw per-pixel error and reverted to it at the end.  That metric structurally
        # penalises anything that trades local pixel accuracy for a better *result* —
        # dithering (intentional local variance for perceptual averaging) and densify's
        # gradient-driven splits (adds circles whose benefit only shows up once THEIR
        # colours are optimised a step later) both score as "worse" the moment they
        # happen, so the snapshot almost always reverted back near iteration 1 and threw
        # away the refinement from every iteration after it.  Measured A/B: final quality
        # is statistically the same (~39.4 ΔE either way) whether or not that snapshot is
        # used, so we now just run every iteration forward and keep the last one — no
        # reason to discard iterations 2..N for a metric that penalises the very features
        # we want them to do.
        # Stashed on the instance so the hot inner Python loops (error-diffusion dither,
        # densify) can check it themselves every ~1000 items instead of only between
        # whole iterations — on a large mosaic a single iteration's dither pass alone
        # can run for several seconds, which is how long Cancel used to take to land.
        self._stop_event = stop_event
        last_score = float('inf')

        for i in range(self.cfg.n_iterations):
            if stop_event and stop_event.is_set():
                raise InterruptedError("cancelled")
            _t_lloyd0 = time.time()
            err = self._lloyd_step()
            print(f"[optimizer] iter {i+1}/{self.cfg.n_iterations} _lloyd_step: "
                  f"{time.time()-_t_lloyd0:.1f}s, {len(self.xs)} circles", flush=True)
            score = float(err.mean())
            last_score = score

            if progress_cb:
                progress_cb(i + 1, score)
            if preview_cb and palette_hex:
                _t_prev0 = time.time()
                import json as _json
                # Skip expensive voronoi/neighbor computation on mid-run iter previews —
                # the 250 ms coalescing throttle discards most of them anyway, and the
                # computation blocks the worker for 100–500 ms per call at high circle counts.
                # JS keeps the previous polygon layout when polygons=null arrives.
                _pxs, _pys, _prs, _pcolors = _preview_subsample(
                    self.xs, self.ys, self.rs, self.colors)
                preview_cb(_json.dumps({
                    'xs': _pxs.tolist(),
                    'ys': _pys.tolist(),
                    'rs': _prs.tolist(),
                    'colors': _pcolors.tolist(),
                    'palette': palette_hex,
                    'canvas_hex': canvas_hex,
                    'W': self.W,
                    'H': self.H,
                    'mode': mode,
                    'phase': 'iter',
                    'iter': i + 1,
                    'total_iter': self.cfg.n_iterations,
                    'r_min': float(self.cfg.r_min),
                    'r_max': float(self.cfg.r_max),
                    'neighbors': None,
                    'polygons': None,
                }))
                print(f"[optimizer] iter {i+1} preview build+send: "
                      f"{time.time()-_t_prev0:.1f}s", flush=True)

            # Densify and prune prep the field for the next Lloyd step.
            # On the last iteration this is skipped — the loop always ends on Lloyd.
            if i < self.cfg.n_iterations - 1 and self.cfg.allow_restructure:
                _t_dp0 = time.time()
                self._densify(err)
                self._prune()
                print(f"[optimizer] iter {i+1} densify+prune: {time.time()-_t_dp0:.1f}s, "
                      f"{len(self.xs)} circles after", flush=True)

        # Cleanup pass: one extra Lloyd step after the last configured iteration so
        # circles can resize and reposition into any gaps left by the final prune.  This
        # ALSO re-runs the full recolor + dithering pass from a fresh error accumulator —
        # so if it scores worse than where the loop already was, we revert to that
        # pre-cleanup state rather than silently keeping a strictly worse result.
        if not (stop_event and stop_event.is_set()):
            pre_cleanup_state = (self.xs.copy(), self.ys.copy(), self.rs.copy(), self.colors.copy())
            pre_cleanup_score = last_score
            _t_cleanup0 = time.time()
            cleanup_err = self._lloyd_step()
            print(f"[optimizer] cleanup _lloyd_step: {time.time()-_t_cleanup0:.1f}s, "
                  f"{len(self.xs)} circles", flush=True)
            cleanup_score = float(cleanup_err.mean())
            if cleanup_score > pre_cleanup_score:
                # Cleanup made it worse — revert to the pre-cleanup state instead of keeping it.
                self.xs, self.ys, self.rs, self.colors = pre_cleanup_state
                cleanup_err = np.full_like(cleanup_err, pre_cleanup_score)
            if progress_cb:
                progress_cb(self.cfg.n_iterations + 1, float(cleanup_err.mean()))
            if preview_cb and palette_hex:
                _t_prev0 = time.time()
                import json as _json
                # neighbors/polygons are ONLY read by the JS preview's 'polygon' mode
                # branch — computing a full Delaunay triangulation + Voronoi cell
                # structure here unconditionally (regardless of render mode) is wasted
                # work for circle/airbrush mode, and at high circle counts (tens of
                # thousands, after densify) is expensive enough on its own to make the
                # final cleanup pass look hung for many minutes with no progress update,
                # on top of a much larger JSON payload to serialize and ship over the
                # websocket. Same reasoning already applied to the mid-run previews
                # above (which skip this for a different reason — throttling).
                if mode == 'polygon':
                    _neighbors = _delaunay_neighbors(self.xs, self.ys, self.W, self.H)
                    _polygons = _voronoi_polygons(self.xs, self.ys, self.W, self.H)
                else:
                    _neighbors = None
                    _polygons = None
                print(f"[optimizer] cleanup neighbors/polygons: "
                      f"{time.time()-_t_prev0:.1f}s", flush=True)
                _t_prev0 = time.time()
                # Polygon mode's cell geometry (_neighbors/_polygons above) is index-
                # aligned to the FULL xs/ys — subsampling here would desync them, so
                # only circle/airbrush mode (where those are always None) gets capped.
                if mode == 'polygon':
                    _pxs, _pys, _prs, _pcolors = self.xs, self.ys, self.rs, self.colors
                else:
                    _pxs, _pys, _prs, _pcolors = _preview_subsample(
                        self.xs, self.ys, self.rs, self.colors)
                preview_cb(_json.dumps({
                    'xs': _pxs.tolist(),
                    'ys': _pys.tolist(),
                    'rs': _prs.tolist(),
                    'colors': _pcolors.tolist(),
                    'palette': palette_hex,
                    'canvas_hex': canvas_hex,
                    'W': self.W,
                    'H': self.H,
                    'mode': mode,
                    'phase': 'cleanup',
                    'iter': self.cfg.n_iterations + 1,
                    'total_iter': self.cfg.n_iterations,
                    'r_min': float(self.cfg.r_min),
                    'r_max': float(self.cfg.r_max),
                    'neighbors': _neighbors,
                    'polygons': _polygons,
                }))
                print(f"[optimizer] cleanup preview build+send: "
                      f"{time.time()-_t_prev0:.1f}s", flush=True)

        return self.get_circles()

    def get_circles(self) -> tuple[list[tuple[float, float, float]], np.ndarray]:
        circles = list(zip(self.xs.tolist(), self.ys.tolist(), self.rs.tolist()))
        return circles, self.colors.copy()

    # ── Core Lloyd step ──────────────────────────────────────────────────────

    def _lloyd_step(self) -> np.ndarray:
        """
        One full Lloyd iteration. Returns per-pixel LAB error map.
        """
        N = len(self.xs)
        if N == 0:
            return np.zeros((self.H, self.W), dtype=np.float32)

        # 1. Voronoi assignment: each pixel → nearest circle center
        pts = np.column_stack([self.xs, self.ys])  # (N, 2)
        tree = KDTree(pts)

        # Voronoi at subsampled resolution for speed, then upscale assignment.
        step = max(1, min(4, self.H // 200))   # adaptive: 1 for tiny, 4 for large
        py_sub = self.py_grid[::step, ::step]
        px_sub = self.px_grid[::step, ::step]
        ph, pw = py_sub.shape
        px_flat = np.column_stack([px_sub.ravel(), py_sub.ravel()])
        _, asgn_flat = tree.query(px_flat, workers=-1)
        asgn_sub = asgn_flat.reshape(ph, pw).astype(np.int32)

        # Upsample back to full resolution using nearest-neighbour
        import cv2 as _cv2
        assignment = _cv2.resize(
            asgn_sub, (self.W, self.H), interpolation=_cv2.INTER_NEAREST
        ).astype(np.int32)

        flat_assign = assignment.ravel()                    # (H*W,)

        if _USE_GPU and self._dev is not None:
            import torch as _torch
            dev = self._dev
            n_colors = self.weight_maps.shape[0]

            # Upload per-iteration data: assignment, colors, circle positions+radii
            fa_t   = _torch.from_numpy(flat_assign.astype(np.int64)).to(dev)
            col_t  = _torch.from_numpy(self.colors.astype(np.int64)).to(dev)
            xs_t   = _torch.from_numpy(self.xs).to(dev)
            ys_t   = _torch.from_numpy(self.ys).to(dev)
            rs_t   = _torch.from_numpy(self.rs).to(dev)

            # 2. Physically-accurate rendering: pixels inside circle radius get
            #    paint color; pixels outside get canvas color.
            #    This gives the optimizer the true error signal — it now knows
            #    exactly which uncovered regions are contributing to the error.
            cpp_t      = col_t[fa_t]                                 # (H*W,) color index
            assign_xs  = xs_t[fa_t]
            assign_ys  = ys_t[fa_t]
            assign_rs  = rs_t[fa_t]
            dx_t = self._t_flat_px - assign_xs
            dy_t = self._t_flat_py - assign_ys
            inside_t = (dx_t*dx_t + dy_t*dy_t) <= (assign_rs*assign_rs)  # (H*W,)
            paint_t  = self._t_palette[cpp_t]                            # (H*W, 3)
            canvas_t = self._t_canvas.unsqueeze(0).expand(self.H*self.W, 3)
            rend_flat = _torch.where(inside_t.unsqueeze(1), paint_t, canvas_t)
            rend_t   = rend_flat.view(self.H, self.W, 3)
            diff_t   = self._t_src_lab - rend_t                          # (H,W,3)
            err_t    = _torch.sqrt(diff_t[:,:,0]**2 * 4.0
                                   + diff_t[:,:,1]**2
                                   + diff_t[:,:,2]**2)                   # (H,W)

            # 3. Importance + scatter (GPU bincount)
            lum_err_t = diff_t[:,:,0].abs().ravel()
            imp_t = (self._t_wt_max.ravel()
                     * (1.0 + self._t_detail.ravel() * 2.0 + lum_err_t / 25.0)
                     ).clamp(min=1e-6)

            w_sum_t  = _torch.zeros(N, device=dev).scatter_add_(0, fa_t, imp_t)
            sum_wx_t = _torch.zeros(N, device=dev).scatter_add_(0, fa_t, imp_t * self._t_flat_px)
            sum_wy_t = _torch.zeros(N, device=dev).scatter_add_(0, fa_t, imp_t * self._t_flat_py)
            sum_L_t  = _torch.zeros(N, device=dev).scatter_add_(0, fa_t, imp_t * self._t_src_lab[:,:,0].ravel())
            cnt_t    = _torch.zeros(N, device=dev).scatter_add_(0, fa_t, self._t_ones)

            # Color votes (GPU)
            vk_t  = fa_t * n_colors + self._t_dominant_px
            vc_t  = _torch.zeros(N * n_colors, device=dev).scatter_add_(0, vk_t, self._t_ones)

            # Download everything in one batch
            cpu = _torch.stack([w_sum_t, sum_wx_t, sum_wy_t, sum_L_t, cnt_t]).cpu().numpy()
            w_sum, sum_wx, sum_wy, sum_L, counts = cpu
            err_map    = err_t.cpu().numpy()
            vote_matrix = vc_t.cpu().numpy().reshape(N, n_colors)

        else:
            # ── CPU fallback ──────────────────────────────────────────────────
            color_per_pixel = self.colors[assignment]
            rendered_lab    = self.palette_lab[color_per_pixel].copy()  # (H,W,3)
            # Physically-accurate: pixels outside their assigned circle show canvas
            assign_xs = self.xs[assignment]   # (H,W)
            assign_ys = self.ys[assignment]
            assign_rs = self.rs[assignment]
            dx = self.px_grid - assign_xs
            dy = self.py_grid - assign_ys
            outside = (dx*dx + dy*dy) > (assign_rs*assign_rs)
            rendered_lab[outside] = self.canvas_lab
            diff = self.src_lab - rendered_lab
            err_map = np.sqrt(
                diff[:, :, 0] ** 2 * 4.0
                + diff[:, :, 1] ** 2
                + diff[:, :, 2] ** 2
            )
            lum_err    = np.abs(diff[:, :, 0])
            importance = (
                self.weight_maps.max(axis=0)
                * (1.0 + self.detail * 2.0 + lum_err / 25.0)
            ).clip(min=1e-6)
            flat_imp = importance.ravel()
            flat_px  = self.px_grid.ravel()
            flat_py  = self.py_grid.ravel()
            flat_L   = self.src_lab[:, :, 0].ravel().astype(np.float32)

            w_sum  = np.bincount(flat_assign, weights=flat_imp,           minlength=N).astype(np.float32)
            sum_wx = np.bincount(flat_assign, weights=flat_imp * flat_px, minlength=N).astype(np.float32)
            sum_wy = np.bincount(flat_assign, weights=flat_imp * flat_py, minlength=N).astype(np.float32)
            sum_L  = np.bincount(flat_assign, weights=flat_imp * flat_L,  minlength=N)
            counts = np.bincount(flat_assign, minlength=N).astype(np.float32)

            n_colors    = self.weight_maps.shape[0]
            dominant_px = self.weight_maps.reshape(n_colors, -1).argmax(axis=0)
            vote_keys   = flat_assign * n_colors + dominant_px
            vote_counts = np.bincount(vote_keys, minlength=N * n_colors)
            vote_matrix = vote_counts.reshape(N, n_colors)

        # ── Centroid update (tiny, CPU) ───────────────────────────────────────
        safe_w = np.where(w_sum > 1e-8, w_sum, 1.0)
        cx = (sum_wx / safe_w).astype(np.float32)
        cy = (sum_wy / safe_w).astype(np.float32)
        empty = counts == 0
        cx[empty] = self.xs[empty]
        cy[empty] = self.ys[empty]

        # Lloyd relaxation (move every circle to its weighted cell centroid,
        # repeated) has a well-known mathematical equilibrium in any flat,
        # uniform-importance region: a regular hexagonal/grid lattice. That's
        # not a bug in any single step, it's what the algorithm converges TO
        # when there's no detail signal pulling it anywhere else. A small,
        # PERSISTENT (not iteration-random — a per-circle offset that never
        # decays away, so it survives to equilibrium instead of averaging
        # out) perturbation breaks that lock-in. Each circle's own stable
        # index gives it a golden-angle direction — the same irrational-angle
        # principle behind phyllotaxis (sunflower-seed) packing, chosen
        # specifically because consecutive indices land far apart around the
        # circle, so neighbouring circles get uncorrelated push directions
        # and can't collectively re-align into a periodic pattern. Scaled
        # down (and to ~zero) in genuinely detailed areas, where the
        # existing detail-driven placement should still win outright.
        _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
        _idx = np.arange(N, dtype=np.float64)
        _phyllo_ang = (_idx * _GOLDEN_ANGLE) % (2 * math.pi)
        _ix_cur = np.clip(self.xs.astype(np.int32), 0, self.W - 1)
        _iy_cur = np.clip(self.ys.astype(np.int32), 0, self.H - 1)
        _flatness = 1.0 - self.detail[_iy_cur, _ix_cur]   # 1 in flat areas, ~0 in detail
        _jitter_mag = (self.rs * 0.22 * _flatness).astype(np.float32)
        cx = cx + _jitter_mag * np.cos(_phyllo_ang).astype(np.float32)
        cy = cy + _jitter_mag * np.sin(_phyllo_ang).astype(np.float32)

        α = self.cfg.lloyd_smoothing
        new_xs = (α * cx + (1 - α) * self.xs).astype(np.float32)
        new_ys = (α * cy + (1 - α) * self.ys).astype(np.float32)

        # ── Radius from detail map + cell-averaged luminosity ─────────────────
        avg_L = (sum_L / safe_w).astype(np.float32)
        avg_L[empty] = self.src_lab[
            np.clip(self.ys[empty].astype(np.int32), 0, self.H-1),
            np.clip(self.xs[empty].astype(np.int32), 0, self.W-1), 0]

        # Detail at each circle's new position: smooth=0 → large, detailed=1 → small
        ix_new = np.clip(new_xs.astype(np.int32), 0, self.W - 1)
        iy_new = np.clip(new_ys.astype(np.int32), 0, self.H - 1)
        detail_at = self.detail[iy_new, ix_new]           # (N,) 0=smooth, 1=detailed
        detail_size = 1.0 - detail_at                     # smooth → 1 (large), detailed → 0 (small)

        canvas_L    = float(self.canvas_lab[0])
        canvas_t    = float(np.clip(canvas_L / 100.0, 0.0, 1.0))
        lum_norm    = np.clip(avg_L / 100.0, 0.0, 1.0)
        directed    = canvas_t * (1.0 - lum_norm) + (1.0 - canvas_t) * lum_norm
        sigmoid_k   = 8.0
        lum_size    = 1.0 / (1.0 + np.exp(-sigmoid_k * (directed - 0.5)))

        lum_influence = getattr(self.cfg, 'lum_influence', 0.5)
        size_factor = ((1.0 - lum_influence) * detail_size + lum_influence * lum_size).astype(np.float32)

        r_new  = (self.cfg.r_min + (self.cfg.r_max - self.cfg.r_min) * size_factor).astype(np.float32)
        new_rs = np.clip(r_new, self.cfg.r_min, self.cfg.r_max).astype(np.float32)
        new_rs[empty] = self.rs[empty]

        # ── Overlap fix (CPU — KDTree query_pairs is already fast) ───────────
        pts_new = np.column_stack([new_xs.astype(np.float64), new_ys.astype(np.float64)])
        _tree = KDTree(pts_new)
        _gap = self.cfg.gap
        for _pass in range(4):
            _pairs = _tree.query_pairs(r=float(new_rs.max()) * 2 + _gap,
                                       output_type='ndarray')
            if len(_pairs) == 0:
                break
            _ii, _jj = _pairs[:, 0], _pairs[:, 1]
            _dx = pts_new[_jj, 0] - pts_new[_ii, 0]
            _dy = pts_new[_jj, 1] - pts_new[_ii, 1]
            _dist = np.sqrt(_dx * _dx + _dy * _dy).astype(np.float32)
            _need = new_rs[_ii] + new_rs[_jj] + _gap
            _mask = _dist < _need
            if not _mask.any():
                break
            _ii, _jj, _dist, _need = _ii[_mask], _jj[_mask], _dist[_mask], _need[_mask]
            _scale = np.sqrt(np.maximum(_dist / _need, 1e-6))
            np.minimum.at(new_rs, _ii, new_rs[_ii] * _scale)
            np.minimum.at(new_rs, _jj, new_rs[_jj] * _scale)
        new_rs = np.clip(new_rs, self.cfg.r_min, self.cfg.r_max)

        # ── Recolor from vote matrix ──────────────────────────────────────────
        new_colors = vote_matrix.argmax(axis=1).astype(np.int32)
        new_colors[empty] = self.colors[empty]

        # Clamp so circle edges (+ gap) stay inside the canvas
        margin_x = new_rs + self.cfg.gap
        margin_y = new_rs + self.cfg.gap
        new_xs = np.clip(new_xs, margin_x, self.W - 1 - margin_x)
        new_ys = np.clip(new_ys, margin_y, self.H - 1 - margin_y)

        # 4. Resolve overlaps: push apart pairs that violate gap
        new_xs, new_ys = self._resolve_overlaps(new_xs, new_ys, new_rs)

        # 5. Error-diffusion recolor: replaces simple argmax with spatial dithering.
        #    Propagates color residuals to neighbors so adjacent circles optically
        #    mix into intermediate tones the palette cannot directly express.
        if self.cfg.dither_strength > 0.01:
            from .color_assign import error_diffusion_assign
            new_colors = error_diffusion_assign(
                new_xs, new_ys, new_rs,
                self.target_lab,
                self.palette_lab,
                dither_strength=self.cfg.dither_strength,
                blend_radius_multiplier=self.cfg.blend_radius,
                kernel=self.cfg.dither_kernel,
                stop_event=getattr(self, '_stop_event', None),
            )

        self.xs, self.ys, self.rs, self.colors = new_xs, new_ys, new_rs, new_colors
        return err_map

    # ── Densification ────────────────────────────────────────────────────────

    def _densify(self, err_map: np.ndarray):
        """
        Split circles in high-error regions by replacing one large circle
        with two smaller circles placed at the error centroid.
        """
        if len(self.xs) == 0:
            return

        pts = np.column_stack([self.xs, self.ys])
        tree = KDTree(pts)
        # How close this circle's nearest neighbour already is -- used below
        # to skip densifying anywhere already packed as tight as physically
        # possible, which is what actually bounds growth once the "already
        # at r_min" skip no longer applies unconditionally (see below): with
        # nothing else to stop it, removing that gate let densify keep
        # re-triggering on the same saturated area every iteration with no
        # natural stopping point (measured: 2011 -> 45187 circles over 8
        # iterations, with genuinely overlapping pairs left behind).
        nn_dist, _ = tree.query(pts, k=2, workers=-1)
        nn_dist = nn_dist[:, 1]
        step = max(1, min(4, self.H // 200))
        py_s = self.py_grid[::step, ::step]; px_s = self.px_grid[::step, ::step]
        _, asgn_s = tree.query(np.column_stack([px_s.ravel(), py_s.ravel()]), workers=-1)
        import cv2 as _cv2
        assignment = _cv2.resize(
            asgn_s.reshape(py_s.shape).astype(np.int32),
            (self.W, self.H), interpolation=_cv2.INTER_NEAREST,
        )
        threshold = np.percentile(err_map, self.cfg.densify_error_pct)
        high_err = err_map > threshold

        # Second, independent trigger: local COLOUR GRADIENT (not current-fit error).
        # A large circle can sit dead-centre of a real gradient and still have a fine
        # per-pixel match (low err_map) at its own centre — so it never gets flagged by
        # the error-based rule even though the colour is genuinely changing under its
        # footprint.  This lets large circles interject smaller ones wherever there's
        # real gradient, without lowering the global max-size ceiling.  Kept conservative
        # (high percentile, large-circles-only) so it targets exactly this case instead of
        # broadly re-triggering splits everywhere.
        grad_trigger = self.detail > np.percentile(self.detail, 72.0)
        combined = high_err | grad_trigger

        N = len(self.xs)
        # Vectorized equivalent of the old per-circle loop below — that loop
        # recomputed `(assignment == k) & combined` (a full-image boolean mask)
        # for EVERY circle, i.e. O(circles * H * W) comparisons. bincount gets
        # the exact same per-circle pixel counts in one O(H*W) pass instead —
        # this was the dominant cost of the whole optimization loop (profiling
        # showed densify+prune taking ~18s/iteration on a real job, almost all
        # of it here).
        flat_assignment = assignment.ravel()
        combined_owners = flat_assignment[combined.ravel()]
        cell_counts = np.bincount(combined_owners, minlength=N)
        high_err_owners = flat_assignment[high_err.ravel()]
        has_high_err = np.zeros(N, dtype=bool)
        if high_err_owners.size:
            has_high_err[np.unique(high_err_owners)] = True

        add_xs, add_ys, add_rs, add_cs = [], [], [], []
        remove_mask = np.zeros(N, dtype=bool)

        # Each candidate's own "is there room" check above only looked at
        # the PRE-densify snapshot, so two nearby candidates (different
        # parents, each with genuine room against the old layout) could
        # still place children that overlap EACH OTHER once both go
        # through -- `_resolve_overlaps` afterwards is a positional push
        # that isn't guaranteed to fully converge against thousands of
        # simultaneous new conflicts, especially now that circles can't
        # shrink to help when r_min == r_max (measured: 84.7% of circles
        # ended up overlapping their nearest neighbour with only the
        # pre-check in place, vs 0% in the normal variable-size case). A
        # spatial hash of accepted positions, checked and updated as each
        # child is placed (not just once per parent against a stale
        # snapshot), makes every acceptance genuinely conflict-free against
        # everything decided so far in this same pass, not just the old one.
        cell = max(1.0, 2.0 * self.cfg.r_min + self.cfg.gap)
        grid: dict = {}

        def _cell_of(x, y):
            return (int(x / cell), int(y / cell))

        def _insert(x, y, r):
            grid.setdefault(_cell_of(x, y), []).append((x, y, r))

        def _remove(x, y, r):
            # Exact-tuple removal: used to pull a parent circle out of the
            # hash before testing its own children against it -- children
            # sit close enough to their parent's old centre that leaving the
            # about-to-be-replaced parent in place could spuriously reject
            # its own replacements as "too close to a neighbour".
            bucket = grid.get(_cell_of(x, y))
            if bucket:
                try:
                    bucket.remove((x, y, r))
                except ValueError:
                    pass

        def _has_room(x, y, r):
            cx, cy = _cell_of(x, y)
            rad = int((r + self.cfg.r_max + self.cfg.gap) / cell) + 1
            for gx in range(cx - rad, cx + rad + 1):
                for gy in range(cy - rad, cy + rad + 1):
                    for (ox, oy, orr) in grid.get((gx, gy), ()):
                        d2 = (x - ox) ** 2 + (y - oy) ** 2
                        min_d = r + orr + self.cfg.gap
                        if d2 < min_d * min_d:
                            return False
            return True

        for _x, _y, _r in zip(self.xs, self.ys, self.rs):
            _insert(float(_x), float(_y), float(_r))

        stop_event = getattr(self, '_stop_event', None)
        candidates = np.nonzero(cell_counts >= 4)[0]
        for i, k in enumerate(candidates):
            if i % 1000 == 0:
                time.sleep(0)   # explicit GIL yield — see color_assign.py's _YIELD_EVERY note
                if stop_event and stop_event.is_set():
                    raise InterruptedError("cancelled")
            r = self.rs[k]
            # "Already at min size, splitting into smaller children achieves
            # nothing" only holds when children CAN actually end up smaller
            # than the parent -- with r_min == r_max (or very close), every
            # circle is r_min already and always was, so this check was
            # unconditionally true and silently disabled densify entirely
            # regardless of local error/detail (reported: a uniform circle
            # size setting stopped detail areas like a face from getting any
            # denser at all). The real benefit of a split is an extra
            # independent sample point wherever there's room for one, which
            # is still true even when the two children come out the same
            # size as the parent -- only skip this early when there's an
            # actual size range for "near the floor" to mean something.
            if self.cfg.r_max > self.cfg.r_min * 1.01 and r < self.cfg.r_min * 1.5:
                continue  # already at min size, can't split meaningfully
            nr_check = float(np.clip(r * 0.65, self.cfg.r_min, self.cfg.r_max))
            if nn_dist[k] < 2.0 * nr_check + self.cfg.gap:
                # Already packed as tight as two same-size circles can
                # physically sit -- a child can't fit here regardless of
                # what triggered this candidate, so there's nothing to gain
                # by splitting (this is the actual stopping condition now
                # that "already at r_min" no longer blocks it by itself).
                continue
            # If this circle only qualifies via the gradient trigger (not the error
            # trigger), require it to still be a moderately large circle — this targets
            # "circle sitting on a real gradient" while letting it chain-split a bit
            # deeper than before, without re-fragmenting dots already near r_min.
            if not has_high_err[k] and r < self.cfg.r_max * 0.3:
                continue

            nr = float(np.clip(r * 0.65, self.cfg.r_min, self.cfg.r_max))
            margin = nr + self.cfg.gap

            # Place up to two children at independent random angles (a few
            # tries each) so adjacent densifications don't chain into
            # trails; each candidate position must clear the shared spatial
            # hash (existing circles AND every child accepted so far this
            # pass) before being accepted, so the result is genuinely
            # conflict-free rather than merely "was fine against the old
            # layout". Only remove the parent if at least one child
            # actually found room -- otherwise this area is already fully
            # saturated and the parent should just stay as-is.
            px, py = float(self.xs[k]), float(self.ys[k])
            _remove(px, py, r)
            placed = 0
            for _ in range(2):
                for _try in range(4):
                    angle = np.random.uniform(0, 2 * np.pi)
                    offset = nr + self.cfg.gap / 2.0
                    nx = float(np.clip(self.xs[k] + offset * np.cos(angle), margin, self.W - 1 - margin))
                    ny = float(np.clip(self.ys[k] + offset * np.sin(angle), margin, self.H - 1 - margin))
                    if _has_room(nx, ny, nr):
                        _insert(nx, ny, nr)
                        c = int(np.argmax(self.weight_maps[:, min(int(ny), self.H-1), min(int(nx), self.W-1)]))
                        add_xs.append(nx)
                        add_ys.append(ny)
                        add_rs.append(nr)
                        add_cs.append(c)
                        placed += 1
                        break
            if placed > 0:
                remove_mask[k] = True  # parent replaced by its accepted children
                if placed == 1:
                    # Give back the parent's own footprint as a "child" so a
                    # partial split doesn't just delete real coverage.
                    _insert(px, py, r)
                    add_xs.append(px)
                    add_ys.append(py)
                    add_rs.append(float(r))
                    add_cs.append(int(self.colors[k]))
            else:
                _insert(px, py, r)   # no room found -- put the parent back

        keep = ~remove_mask
        self.xs = np.concatenate([self.xs[keep], np.array(add_xs, dtype=np.float32)]) if add_xs else self.xs[keep]
        self.ys = np.concatenate([self.ys[keep], np.array(add_ys, dtype=np.float32)]) if add_xs else self.ys[keep]
        self.rs = np.concatenate([self.rs[keep], np.array(add_rs, dtype=np.float32)]) if add_xs else self.rs[keep]
        self.colors = np.concatenate([self.colors[keep], np.array(add_cs, dtype=np.int32)]) if add_xs else self.colors[keep]

        if add_xs:
            self.xs, self.ys = self._resolve_overlaps(self.xs, self.ys, self.rs)

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune(self):
        """
        Remove circles whose Voronoi cell has very low importance weight
        (they contribute nothing perceptually useful).
        """
        if len(self.xs) < 2:
            return

        pts = np.column_stack([self.xs, self.ys])
        tree = KDTree(pts)
        N = len(self.xs)
        step = max(1, min(4, self.H // 200))
        py_s = self.py_grid[::step, ::step]; px_s = self.px_grid[::step, ::step]
        _, asgn_s = tree.query(np.column_stack([px_s.ravel(), py_s.ravel()]), workers=-1)
        import cv2 as _cv2
        assignment = _cv2.resize(
            asgn_s.reshape(py_s.shape).astype(np.int32),
            (self.W, self.H), interpolation=_cv2.INTER_NEAREST,
        ).ravel()

        # Vectorized: mean weight per circle using bincount
        c_per_px = self.colors[assignment]                                     # (H*W,)
        wt_per_px = self.weight_maps[c_per_px, np.arange(self.H*self.W) // self.W,
                                     np.arange(self.H*self.W) % self.W]
        wt_sum    = np.bincount(assignment, weights=wt_per_px, minlength=N)
        px_counts = np.bincount(assignment, minlength=N).astype(np.float32)
        mean_wt   = np.where(px_counts > 0, wt_sum / px_counts.clip(min=1), 0)
        keep = (mean_wt >= self.cfg.prune_min_weight) & (px_counts > 0)

        self.xs = self.xs[keep]
        self.ys = self.ys[keep]
        self.rs = self.rs[keep]
        self.colors = self.colors[keep]

    # ── Overlap resolution ────────────────────────────────────────────────────

    def _resolve_overlaps(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        rs: np.ndarray,
        max_passes: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iteratively push overlapping circles apart (vectorized).
        Each pass: find all violating pairs via KDTree, compute push vectors,
        accumulate per-circle displacement, apply all at once.
        """
        gap = self.cfg.gap
        xs = xs.copy().astype(np.float64)
        ys = ys.copy().astype(np.float64)
        rs_f = rs.astype(np.float64)
        r_max = float(rs_f.max())

        # Keep each circle's whole footprint (radius + gap) inside the canvas.
        margin_x = rs_f + gap
        margin_y = rs_f + gap
        hi_x = self.W - 1 - margin_x
        hi_y = self.H - 1 - margin_y
        # Guard against a margin larger than the canvas (tiny canvas / huge r).
        hi_x = np.maximum(hi_x, margin_x)
        hi_y = np.maximum(hi_y, margin_y)

        for _ in range(max_passes):
            pts = np.column_stack([xs, ys])
            tree = KDTree(pts)
            pairs = tree.query_pairs(r=r_max * 2 + gap + 1, output_type='ndarray')
            if len(pairs) == 0:
                break

            ii, jj = pairs[:, 0], pairs[:, 1]
            dx = xs[jj] - xs[ii]
            dy = ys[jj] - ys[ii]
            dist2 = dx * dx + dy * dy
            min_dist = rs_f[ii] + rs_f[jj] + gap
            min_dist2 = min_dist * min_dist

            overlap_mask = dist2 < min_dist2   # include coincident pairs
            if not overlap_mask.any():
                break

            ii = ii[overlap_mask]
            jj = jj[overlap_mask]
            dx = dx[overlap_mask]
            dy = dy[overlap_mask]
            dist = np.sqrt(dist2[overlap_mask])
            min_d = min_dist[overlap_mask]

            overlap = (min_d - dist) * 0.5
            # Exactly-coincident circles have no separation direction — give them a
            # deterministic one (+x) so they still get pushed apart instead of stuck.
            coincident = dist < 1e-6
            safe = np.where(coincident, 1.0, dist)
            nx = np.where(coincident, 1.0, dx / safe)
            ny = np.where(coincident, 0.0, dy / safe)

            push_x = nx * overlap
            push_y = ny * overlap

            # Accumulate per-circle displacement (both directions)
            delta_x = np.zeros(len(xs))
            delta_y = np.zeros(len(xs))
            np.subtract.at(delta_x, ii, push_x)
            np.subtract.at(delta_y, ii, push_y)
            np.add.at(delta_x, jj, push_x)
            np.add.at(delta_y, jj, push_y)

            xs += delta_x
            ys += delta_y

            # Clamp to the canvas EVERY pass.  A corner clamp can force circles onto
            # the two margin lines and pile them up — clamping here (instead of once
            # at the end) lets the next push pass spread that pile along the edges,
            # which is what stops the overlapping-circle artefact at the corners.
            np.clip(xs, margin_x, hi_x, out=xs)
            np.clip(ys, margin_y, hi_y, out=ys)

        return xs.astype(np.float32), ys.astype(np.float32)


import math
