"""
Single-square-tile mosaic placement, built around ROW TRACING (andamento).

Real tessera mosaics aren't independently-scattered rotated squares -- the
defining visual signature is continuous FLOWING ROWS of tiles that trace
curved paths across the image (following an eye's curve, hair strands, leaf
veins), packed edge-to-edge with near-zero grout. An earlier version of this
module scattered independent seed points and relied on a physics-style
attract/repel dynamic to pack them -- that produced correctly-SIZED tiles
with correct per-tile rotation, but no row coherence, which reads as
"scattered squares" rather than a mosaic (this was shipped, tested, and
rejected on exactly that basis).

This version traces continuous paths along the local contour direction
field and grows new rows by OFFSETTING an already-accepted row's own path
by a fixed perpendicular separation distance (the classic "evenly spaced
streamlines" technique -- Jobard & Lefer 1997 -- adapted from flow
visualization to tile placement). Each row, while being traced, continuously
checks its distance to every other already-accepted row and stops the
instant it would violate the minimum separation, so two rows can never
converge into overlap by construction -- no destructive post-hoc cleanup
needed. Validated in isolation before this port: 0.1% of tiles need pruning
to guarantee zero overlap (vs 48-75% for the scattered-seed approach), and
it's roughly an order of magnitude faster.
"""

import math
import cv2
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box as ShBox, Point as ShPoint
from shapely.strtree import STRtree

from .optimizer import _is_convex

# Small rotations a gap-filler may try before giving up on a size, in
# radians. 0 first so the field angle always wins when it fits; the rest
# are deliberately small (a square repeats every 90 degrees, so +/-12 deg
# covers a lot of near-fits) to close one-tile holes without leaving a
# visibly crooked tile in an otherwise aligned run.
_FILL_ANGLE_TRIES = (0.0, 0.10, -0.10, 0.21, -0.21)

# Hard ceiling on gap-raster cells per leftover piece, so resolving down to
# tile scale can't blow up on a huge region.
_FILL_MAX_CELLS = 4_000_000


def _build_color_direction_field(xs, ys, colors_bgr, W, H, analysis_res=256, rs=None):
    """Direction field driven by actual image STRUCTURE (colour boundaries),
    not circle-size variation. Circle density only tells you how much detail
    is nearby, not which way any real edge/feature runs -- rows traced from
    a size-based field read as arbitrary swirls instead of following the
    source image's actual contours (a face's eye/jaw/hair boundaries, a
    leaf's veins, etc). This builds a low-res raster of the actual per-point
    colours, takes its luminance gradient, and returns a contour_angle
    closure (perpendicular to the gradient, i.e. tangent to colour edges) —
    the same interface `_build_topography`'s contour_angle has, so it's a
    drop-in replacement for row *direction* while `local_scale` (tile size)
    still comes from the existing detail-based topography.

    A version of this was swapped for `tile_cvd.py`'s distance-transform-
    to-strong-edges field mid-session, on the theory that it'd be strictly
    more robust to photo noise. It was NOT a good fit here: that field only
    keeps the top ~20% of pixels by gradient magnitude as real "edges" and
    has every other pixel copy its nearest one's EXACT tangent verbatim.
    For CVD (each tile sampling the field independently) that reads as
    mild extra coherence between nearby tiles. For row-tracing -- which
    walks a literal connected PATH along this field for potentially
    hundreds of tiles -- a large, fairly flat-contrast region (a bright
    face highlight, say) has no qualifying edges at all, so the entire
    region gets remapped to one distant reference edge's exact angle: a
    dead-flat plateau that traces out as a rigid straight bar cutting
    across real curved structure (reported directly, and clearly visible:
    straight horizontal rows slicing through a face while the surrounding
    hair still swirled normally). Using each pixel's OWN local gradient
    almost everywhere, with only a gentle nearest-fill for genuinely
    zero-signal pixels, keeps the field continuously varying instead of
    forming those plateaus -- this is what actually produced the liked
    concentric-contour screenshots."""
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from scipy.spatial import KDTree

    g = max(16, int(analysis_res))
    sx, sy = g / float(W), g / float(H)
    tree = KDTree(np.column_stack([xs * sx, ys * sy]))
    gy, gx = np.mgrid[0:g, 0:g]
    grid_pts = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
    _, nearest = tree.query(grid_pts, k=1)

    cb = np.asarray(colors_bgr, dtype=np.float32)
    color_grid = cb[nearest].reshape(g, g, 3)
    blur_sigma = max(1.0, g / 60.0)
    color_grid = gaussian_filter(color_grid, sigma=(blur_sigma, blur_sigma, 0))

    # BGR -> approximate luma for a single gradient-friendly scalar field.
    luma = 0.114 * color_grid[..., 0] + 0.587 * color_grid[..., 1] + 0.299 * color_grid[..., 2]
    gy_e, gx_e = np.gradient(luma)
    grad_mag = np.clip(gaussian_filter(np.hypot(gx_e, gy_e), 1.5), 1e-6, None)

    if rs is not None and len(rs) == len(xs):
        # Blend in a genuine elevation surface (detail density -> hills),
        # whose smooth, hierarchical contour lines are a second real source
        # of the liked topographic flavour, orthogonal to colour edges --
        # same gentle own-gradient style, no percentile-threshold plateaus.
        # Blended as RAW (undoubled) gradient vectors: both are genuine
        # single-valued vectors at this stage (a scalar field's own
        # gradient has a real sign, unlike the TANGENT direction derived
        # from it later) so a plain weighted vector sum is correct here --
        # doubling is only needed afterward, for the flat-region nearest-
        # fill step's mod-pi disambiguation, not for combining two sources.
        from .tile_cvd import _build_detail_topography_field
        topo_field = _build_detail_topography_field(xs, ys, rs, W, H, g)
        # topo_field's theta_grid already has the +pi/2 tangent rotation
        # baked in (see _build_detail_topography_field) -- undo it here so
        # both fields are combined in the same raw-gradient convention.
        topo_raw_ang = topo_field['theta_grid'] - math.pi / 2
        # Elevation-gradient magnitude and luma-gradient magnitude are
        # unrelated units -- normalize each to its own 90th percentile
        # before using them as blend weights, so one source can't
        # arbitrarily dominate just because its raw numbers happen to be
        # bigger.
        color_w = grad_mag / (np.percentile(grad_mag, 90) + 1e-9)
        topo_w = topo_field['grad_mag'] / (np.percentile(topo_field['grad_mag'], 90) + 1e-9)
        vec = color_w * np.exp(1j * np.arctan2(gy_e, gx_e)) + topo_w * np.exp(1j * topo_raw_ang)
        mag = np.abs(vec)
        gy_e = np.where(mag > 1e-9, np.imag(vec), gy_e)
        gx_e = np.where(mag > 1e-9, np.real(vec), gx_e)
        grad_mag = np.maximum(grad_mag, topo_field['grad_mag'])

    _doubled = np.exp(2j * np.arctan2(gy_e, gx_e))

    # Flat/uniform-colour cells have ~zero gradient, which would all pick the
    # same fixed axis (atan2(0,0)) -- inherit the nearest REAL edge direction
    # instead (doubled-angle so the mod-pi axis ambiguity fills correctly).
    _flat = grad_mag < (grad_mag.max() * 0.03 if grad_mag.max() > 0 else 1.0)
    if _flat.any() and not _flat.all():
        _, (_iy, _ix) = distance_transform_edt(_flat, return_indices=True)
        _doubled = _doubled[_iy, _ix]
    dir_y, dir_x = np.imag(_doubled), np.real(_doubled)

    def contour_angle(x, y):
        gxp = min(max(int(round(x * sx)), 0), g - 1)
        gyp = min(max(int(round(y * sy)), 0), g - 1)
        grad_ang = 0.5 * math.atan2(dir_y[gyp, gxp], dir_x[gyp, gxp])
        return grad_ang + math.pi / 2   # perpendicular to gradient == along the edge

    return contour_angle


# ── Geometry ─────────────────────────────────────────────────────────────

def make_square(cx, cy, ang, half):
    tx, ty = math.cos(ang), math.sin(ang)
    nx, ny = -ty, tx
    return [
        [cx + tx * half + nx * half, cy + ty * half + ny * half],
        [cx - tx * half + nx * half, cy - ty * half + ny * half],
        [cx - tx * half - nx * half, cy - ty * half - ny * half],
        [cx + tx * half - nx * half, cy + ty * half - ny * half],
    ]


def make_squares_array(xs, ys, angles, halves):
    """Vectorized make_square: builds ALL tiles' 4 corners at once as an
    (n, 4, 2) array, same corner order as make_square."""
    tx = np.cos(angles); ty = np.sin(angles)
    nx = -ty; ny = tx
    h = halves
    p0 = np.stack([xs + tx * h + nx * h, ys + ty * h + ny * h], axis=-1)
    p1 = np.stack([xs - tx * h + nx * h, ys - ty * h + ny * h], axis=-1)
    p2 = np.stack([xs - tx * h - nx * h, ys - ty * h - ny * h], axis=-1)
    p3 = np.stack([xs + tx * h - nx * h, ys + ty * h - ny * h], axis=-1)
    return np.stack([p0, p1, p2, p3], axis=1)


# ── Seeding: density-adaptive, following local_scale() like circle mode ────

def variable_density_seed(W, H, local_side_fn, min_side, spacing_mult=1.0,
                           gap=0.0, oversample=4, rng=None):
    """Greedy Poisson-disk-style seeding with LOCALLY VARYING spacing --
    denser where local_side_fn returns a smaller target size (near image
    detail), sparser where it's larger (flat areas) -- the same
    density-follows-detail behaviour circle mode's own Poisson placement has.
    A uniform spatial hash (cell sized to the smallest possible tile) keeps
    each candidate's neighbour check to a small bounded radius."""
    rng = rng or np.random.default_rng(0)
    cell = max(1.0, min_side * spacing_mult / 1.5)
    grid: dict[tuple[int, int], list] = {}

    def cell_of(x, y):
        return int(x / cell), int(y / cell)

    def far_enough(x, y, side):
        target = (side + gap) * spacing_mult
        cxg, cyg = cell_of(x, y)
        rad = int(target / cell) + 2
        for gx in range(cxg - rad, cxg + rad + 1):
            for gy in range(cyg - rad, cyg + rad + 1):
                for (ox, oy, oside) in grid.get((gx, gy), ()):
                    d = math.hypot(x - ox, y - oy)
                    if d < max(target, oside * spacing_mult) * 0.95:
                        return False
        return True

    n_candidates = int(oversample * W * H / (min_side * spacing_mult) ** 2)
    accepted = []
    for _ in range(n_candidates):
        x = rng.uniform(0, W); y = rng.uniform(0, H)
        side = local_side_fn(x, y)
        if not far_enough(x, y, side):
            continue
        accepted.append((x, y, side))
        grid.setdefault(cell_of(x, y), []).append((x, y, side))
    return accepted


def grid_neighbors(points, k=8):
    """k-nearest-neighbour graph for angle-consistency BFS."""
    from scipy.spatial import KDTree
    pts = np.array(points)
    tree = KDTree(pts)
    _, idx = tree.query(pts, k=min(k + 1, len(pts)))
    return [[j for j in idx[i] if j != i] for i in range(len(pts))]


def propagate_consistent_angles(points, neighbors, topo, W, H):
    """Each cell's raw contour_angle is only defined mod 90 deg (a square
    looks the same rotated a quarter turn) -- deciding which of the 4
    equivalent choices to snap to INDEPENDENTLY per point lets neighbouring
    tiles pick different branches even when the underlying angle is nearly
    identical, breaking visual continuity. Propagate one consistent choice
    outward via BFS over a neighbour graph instead."""
    contour_angle = topo['contour_angle']
    n = len(points)
    raw_ang = np.array([contour_angle(min(max(x, 0), W), min(max(y, 0), H)) for x, y in points])

    resolved = np.full(n, np.nan)
    if n == 0:
        return resolved
    visited = np.zeros(n, dtype=bool)
    resolved[0] = raw_ang[0]
    visited[0] = True
    frontier = [0]
    while frontier:
        nxt = []
        for pi in frontier:
            for pj in neighbors[pi]:
                if visited[pj]:
                    continue
                base = raw_ang[pj]
                parent = resolved[pi]
                best_k, best_d = 0, 1e9
                for k in range(4):
                    cand = base + k * (math.pi / 2)
                    d = abs(((cand - parent) + math.pi) % (2 * math.pi) - math.pi)
                    if d < best_d:
                        best_d, best_k = d, k
                resolved[pj] = base + best_k * (math.pi / 2)
                visited[pj] = True
                nxt.append(pj)
        frontier = nxt
    resolved[np.isnan(resolved)] = raw_ang[np.isnan(resolved)]
    return resolved


# ── Row tracing (andamento): the actual seeding/placement algorithm ────────

class RowTracer:
    """Evenly-spaced streamline tracing, adapted from flow visualization
    (Jobard & Lefer 1997) to tile-row placement. See module docstring."""

    def __init__(self, topo, W, H, r_min, r_max, sep_mult=1.05, test_mult=1.02,
                 step_frac=1.02, min_row_tiles=2):
        self.contour_angle = topo['contour_angle']
        self.local_scale = topo['local_scale']
        self.W, self.H = W, H
        self.r_min, self.r_max = r_min, r_max
        # d_test is a MINIMUM centre-to-centre distance, and tiles have real
        # width: two squares' centres must be at least the sum of their own
        # half-widths apart to avoid overlap (side for axis-aligned squares,
        # growing toward side*sqrt(2)~=1.41 when both are rotated ~45deg
        # toward each other) — test_mult/sep_mult must stay above 1.0, unlike
        # the classic streamline convention (for infinitesimally thin lines)
        # this was adapted from. Kept as CLOSE to 1.0 as the finalize/grow
        # cleanup can safely absorb -- rows should pack against each other
        # like courses of masonry (near-zero grout), not float apart with
        # visible gaps between them. 1.4/1.25 (the first working values) were
        # deliberately generous to guarantee no overlap-explosion while the
        # algorithm was still being debugged; once verified correct, tightening
        # toward 1.0 recovered real masonry packing at only ~3% extra pruning.
        self.sep_mult = sep_mult
        self.test_mult = test_mult
        self.step_frac = step_frac
        self.min_row_tiles = min_row_tiles

        self._pts = np.zeros((0, 2))
        self._tree = None
        self._dirty = 0
        self._rebuild_every = 40
        self._pending = []
        self.rows = []

    def local_side(self, x, y):
        return 2.0 * float(np.clip(self.local_scale(x, y), self.r_min, self.r_max))

    def _rebuild_tree(self):
        if self._pending:
            self._pts = np.vstack([self._pts, np.array(self._pending)]) if len(self._pts) else np.array(self._pending)
            self._pending = []
        self._tree = cKDTree(self._pts) if len(self._pts) else None
        self._dirty = 0

    def _add_point(self, x, y):
        self._pending.append((x, y))
        self._dirty += 1
        if self._dirty >= self._rebuild_every:
            self._rebuild_tree()

    def _dist_to_other_rows(self, x, y):
        best = math.inf
        if self._tree is not None:
            d, _ = self._tree.query([x, y], k=1)
            best = min(best, float(d))
        for (px, py) in self._pending:
            d = math.hypot(x - px, y - py)
            if d < best:
                best = d
        return best

    def _fits_canvas(self, x, y, ang, side):
        """Full rotated footprint check, not just the centre point -- a tile
        whose centre is safely inside can still have a corner poking past
        the edge once rotated, especially near the boundary."""
        half = side / 2.0
        for (cx, cy) in make_square(x, y, ang, half):
            if cx < -1e-6 or cx > self.W + 1e-6 or cy < -1e-6 or cy > self.H + 1e-6:
                return False
        return True

    def _trace_direction(self, x0, y0, ang0, sign, own_recent):
        pts = []
        x, y, prev_ang = x0, y0, ang0
        max_steps = int(3 * (self.W + self.H) / max(1.0, self.r_min))
        # Floor for TAPERED end tiles -- deliberately smaller than a normal
        # tile's r_min ever goes, so a row can pinch down further than usual
        # right at its own end, the same way a real mosaicist cuts
        # progressively smaller pieces to finish a course into a narrowing
        # gap instead of just stopping short and leaving it empty.
        min_taper_side = 2.0 * self.r_min * 0.35
        for _ in range(max_steps):
            raw = self.contour_angle(x, y)   # only defined mod pi
            # pick whichever of the two equivalent directions continues
            # forward from the previous step, so the row doesn't zigzag
            cand_ang = raw if abs(((raw - prev_ang) + math.pi) % (2 * math.pi) - math.pi) < math.pi / 2 \
                else raw + math.pi
            base_side = self.local_side(x, y)

            placed = None
            for shrink in (1.0, 0.8, 0.62, 0.48, 0.36, 0.27, 0.2):
                side = base_side * shrink
                if side < min_taper_side:
                    break
                step = side * self.step_frac * sign
                nx = x + step * math.cos(cand_ang)
                ny = y + step * math.sin(cand_ang)
                if not self._fits_canvas(nx, ny, cand_ang, side):
                    continue
                d_test = side * self.test_mult
                if self._dist_to_other_rows(nx, ny) < d_test:
                    continue
                # self-avoidance: skip the last few own points (that's just
                # the tile behind us) but DO stop if the row curls back on
                # itself.
                too_close_to_self = False
                for (ox, oy) in list(own_recent)[:-3]:
                    if math.hypot(nx - ox, ny - oy) < d_test:
                        too_close_to_self = True
                        break
                if too_close_to_self:
                    continue
                placed = (nx, ny, cand_ang, side)
                break

            if placed is None:
                break
            nx, ny, cand_ang, side = placed
            pts.append((nx, ny, cand_ang, side))
            own_recent.append((nx, ny))
            x, y, prev_ang = nx, ny, cand_ang
        return pts

    def trace_row(self, x0, y0):
        side0 = self.local_side(x0, y0)
        ang0 = self.contour_angle(x0, y0)
        if not self._fits_canvas(x0, y0, ang0, side0):
            return None
        d_test0 = side0 * self.test_mult
        if self._dist_to_other_rows(x0, y0) < d_test0:
            return None
        own_recent = deque(maxlen=64)
        own_recent.append((x0, y0))
        fwd = self._trace_direction(x0, y0, ang0, +1, own_recent)
        own_recent2 = deque(maxlen=64)
        own_recent2.append((x0, y0))
        for p in fwd[:8]:
            own_recent2.append((p[0], p[1]))
        bwd = self._trace_direction(x0, y0, ang0, -1, own_recent2)
        row = list(reversed(bwd)) + [(x0, y0, ang0, side0)] + fwd
        if len(row) < self.min_row_tiles:
            return None
        for (x, y, ang, side) in row:
            self._add_point(x, y)
        self._rebuild_tree()
        return row

    def _offsets_of(self, row):
        """Candidate new-row seed points offset perpendicular from `row`'s
        own path by d_sep, sampled every few tiles (not every single one --
        that would explode the queue for no benefit)."""
        out = []
        for (x, y, ang, side) in row[::3]:
            d_sep = side * self.sep_mult
            perp = ang + math.pi / 2
            for s in (+1, -1):
                nx = x + d_sep * s * math.cos(perp)
                ny = y + d_sep * s * math.sin(perp)
                if 0 <= nx <= self.W and 0 <= ny <= self.H:
                    out.append((nx, ny))
        return out

    def _grow_from_queue(self, queue, max_rows):
        while queue and len(self.rows) < max_rows:
            sx, sy = queue.popleft()
            row = self.trace_row(sx, sy)
            if row is None:
                continue
            self.rows.append(row)
            queue.extend(self._offsets_of(row))

    def _start_row_family(self, x0, y0, max_rows):
        row = self.trace_row(x0, y0)
        if row is None:
            return
        self.rows.append(row)
        self._grow_from_queue(deque(self._offsets_of(row)), max_rows)

    def run(self, seed, extra_seed_grid=None):
        """`extra_seed_grid`: spacing (world units) for a fallback coarse
        grid scan that starts NEW row families in any region the first
        seed's outward-offset growth never reached (isolated flat regions,
        far corners) -- growth only ever expands by offsetting an existing
        row's OWN path, so a single seed can leave large parts of the canvas
        uncovered entirely."""
        max_rows = int(4 * self.W * self.H / max(1.0, (self.r_min * self.sep_mult) ** 2))

        self._start_row_family(seed[0], seed[1], max_rows)

        spacing = extra_seed_grid or max(4.0, self.r_min * 1.5)
        sy = spacing * 0.5
        while sy < self.H and len(self.rows) < max_rows:
            sx = spacing * 0.5
            while sx < self.W and len(self.rows) < max_rows:
                self._start_row_family(sx, sy, max_rows)
                sx += spacing
            sy += spacing

        return self.rows


# ── Exact geometry for the dynamic (no shapely in the hot loop) ────────────

def _point_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _point_seg_dist_batch(P, A, B):
    AB = B - A
    AP = P - A
    l2 = np.sum(AB * AB, axis=-1)
    l2_safe = np.where(l2 < 1e-12, 1.0, l2)
    t = np.clip(np.sum(AP * AB, axis=-1) / l2_safe, 0.0, 1.0)
    t = np.where(l2 < 1e-12, 0.0, t)
    closest = A + t[:, None] * AB
    return np.linalg.norm(P - closest, axis=-1)


def _convex_poly_dist_batch(sq_a, sq_b):
    """Exact minimum distance between two convex polygons, for an entire
    array of (A, B) pairs at once: sq_a, sq_b are (N, 4, 2). Used for the
    attraction phase's narrow-phase gap check -- batching this is what makes
    the dynamic fast at real tile counts (millions of individual Python-level
    distance calls otherwise)."""
    N = sq_a.shape[0]
    best = np.full(N, np.inf)
    for k in range(4):
        A = sq_a[:, k, :]; B = sq_a[:, (k + 1) % 4, :]
        for m in range(4):
            d = _point_seg_dist_batch(sq_b[:, m, :], A, B)
            best = np.minimum(best, d)
    for k in range(4):
        A = sq_b[:, k, :]; B = sq_b[:, (k + 1) % 4, :]
        for m in range(4):
            d = _point_seg_dist_batch(sq_a[:, m, :], A, B)
            best = np.minimum(best, d)
    return best


def _sat_mtv(coords_a, coords_b):
    """Exact minimum translation vector to separate two convex polygons via
    the Separating Axis Theorem. Returns (dx, dy) to move A away from B by
    exactly enough to just touch (zero overlap), or None if not overlapping."""
    def axes(coords):
        out = []
        n = len(coords)
        for i in range(n):
            x1, y1 = coords[i]; x2, y2 = coords[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            l = math.hypot(nx, ny) or 1.0
            out.append((nx / l, ny / l))
        return out

    def project(coords, ax, ay):
        vals = [px * ax + py * ay for px, py in coords]
        return min(vals), max(vals)

    min_overlap = None
    min_axis = None
    for (ax, ay) in axes(coords_a)[:2] + axes(coords_b)[:2]:
        amin, amax = project(coords_a, ax, ay)
        bmin, bmax = project(coords_b, ax, ay)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap <= 0:
            return None
        if min_overlap is None or overlap < min_overlap:
            min_overlap, min_axis = overlap, (ax, ay)

    cax = sum(p[0] for p in coords_a) / len(coords_a)
    cay = sum(p[1] for p in coords_a) / len(coords_a)
    cbx = sum(p[0] for p in coords_b) / len(coords_b)
    cby = sum(p[1] for p in coords_b) / len(coords_b)
    ddx, ddy = cax - cbx, cay - cby
    if ddx * min_axis[0] + ddy * min_axis[1] < 0:
        min_axis = (-min_axis[0], -min_axis[1])
    return (min_axis[0] * min_overlap, min_axis[1] * min_overlap)


def _local_scale_batch(xs_arr, ys_arr, topo, r_min, r_max):
    """Vectorized local_scale() sampling against the topography's own grids
    directly, instead of calling the per-point closure in a Python loop."""
    g = topo['g']; sx = topo['sx']; sy = topo['sy']
    size_grid = topo['size_grid']; grad_mag = topo['grad_mag']
    gxp = np.clip(np.round(xs_arr * sx).astype(np.int64), 0, g - 1)
    gyp = np.clip(np.round(ys_arr * sy).astype(np.int64), 0, g - 1)
    samp = np.maximum(1.5, size_grid[gyp, gxp])
    grade_norm = grad_mag[gyp, gxp] / (grad_mag.max() + 1e-9)
    val = samp * np.clip(1.15 - grade_norm, 0.3, 1.15)
    return np.clip(val, r_min, r_max)


# ── The dynamic: attract (surface tension) + resize + repel (exact SAT) ───

def relax_dynamic(tiles, angles, W, H, topo, r_min, r_max, gap,
                   n_iters=200, repulse_correction=0.6, attract_rate=0.04,
                   resize_rate=0.15, attract_radius_mult=0.4,
                   early_stop_patience=100, early_stop_window=50):
    from scipy.spatial import KDTree
    n = len(tiles)
    xs = np.array([t[0] for t in tiles], dtype=np.float64)
    ys = np.array([t[1] for t in tiles], dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    halves = _local_scale_batch(xs, ys, topo, r_min, r_max)

    conflict_history = []
    for it in range(n_iters):
        # 1. Resize toward THIS iteration's local target (damped).
        target_halves = _local_scale_batch(xs, ys, topo, r_min, r_max)
        halves += (target_halves - halves) * resize_rate

        # 2. Attraction: pull tiles whose gap exceeds the target grout width.
        squares_arr = make_squares_array(xs, ys, angles, halves)
        pts = np.column_stack([xs, ys])
        tree_c = KDTree(pts)
        search_r = halves * (2 + attract_radius_mult)
        neighbor_lists = tree_c.query_ball_point(pts, search_r) if n > 1 else []
        pair_i, pair_j, seen = [], [], set()
        for i in range(n):
            for j in neighbor_lists[i] if n > 1 else ():
                if j == i:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                pair_i.append(key[0]); pair_j.append(key[1])

        attract_push = np.zeros((n, 2))
        if pair_i:
            pi = np.asarray(pair_i, dtype=np.int64)
            pj = np.asarray(pair_j, dtype=np.int64)
            d = _convex_poly_dist_batch(squares_arr[pi], squares_arr[pj])
            extra = d - gap
            mask = extra > 0
            pi, pj, extra = pi[mask], pj[mask], extra[mask]
            ddx, ddy = xs[pj] - xs[pi], ys[pj] - ys[pi]
            dl = np.hypot(ddx, ddy)
            dl = np.where(dl < 1e-9, 1.0, dl)
            fx, fy = (ddx / dl) * extra, (ddy / dl) * extra
            np.add.at(attract_push[:, 0], pi, fx)
            np.add.at(attract_push[:, 0], pj, -fx)
            np.add.at(attract_push[:, 1], pi, fy)
            np.add.at(attract_push[:, 1], pj, -fy)

        mag = np.hypot(attract_push[:, 0], attract_push[:, 1])
        with np.errstate(invalid='ignore', divide='ignore'):
            dirn = np.where(mag[:, None] > 1e-9, attract_push / np.maximum(mag, 1e-9)[:, None], 0.0)
        step_mag = np.minimum(mag * attract_rate, halves * 0.3)
        xs += dirn[:, 0] * step_mag
        ys += dirn[:, 1] * step_mag
        xs = np.clip(xs, halves, W - halves)
        ys = np.clip(ys, halves, H - halves)

        # 3. Hard repulsion: exact SAT/MTV, Gauss-Seidel (immediate).
        squares = [make_square(xs[i], ys[i], angles[i], halves[i]) for i in range(n)]
        pts = np.column_stack([xs, ys])
        tree_c2 = KDTree(pts)
        diag = halves * math.sqrt(2)
        max_diag = float(diag.max()) if n else 0.0
        neighbor_lists2 = tree_c2.query_ball_point(pts, diag + max_diag) if n > 1 else []
        seen2 = set()
        n_conflicts = 0
        for i in range(n):
            for j in neighbor_lists2[i] if n > 1 else ():
                if j <= i:
                    continue
                key = (i, j)
                if key in seen2:
                    continue
                seen2.add(key)
                mtv = _sat_mtv(squares[i], squares[j])
                if mtv is None:
                    continue
                n_conflicts += 1
                dx = mtv[0] * 0.5 * repulse_correction
                dy = mtv[1] * 0.5 * repulse_correction
                xs[i] += dx; ys[i] += dy
                xs[j] -= dx; ys[j] -= dy
                squares[i] = make_square(xs[i], ys[i], angles[i], halves[i])
                squares[j] = make_square(xs[j], ys[j], angles[j], halves[j])

        xs = np.clip(xs, halves, W - halves)
        ys = np.clip(ys, halves, H - halves)

        conflict_history.append(n_conflicts)
        if len(conflict_history) >= early_stop_window + early_stop_patience:
            recent = conflict_history[-early_stop_window:]
            prior = conflict_history[-early_stop_window - early_stop_patience:-early_stop_window]
            if min(prior) - min(recent) <= 0:
                break

    return list(zip(xs.tolist(), ys.tolist())), angles.tolist(), halves


# ── Hard zero-overlap guarantee + final corner growth ──────────────────────

def finalize_nonoverlap(tiles, angles, halves, min_overlap_frac=1e-7):
    """Relaxation gets close but doesn't reach EXACT zero (Gauss-Seidel
    dynamics oscillate on the last few grazing contacts). Greedily drop
    whichever tile has the most remaining conflicts each round -- removes
    close to the minimum number of tiles needed to guarantee zero overlap."""
    polys = [Polygon(make_square(cx, cy, a, h)) for (cx, cy), a, h in zip(tiles, angles, halves)]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    if not polys:
        return [], [], 0
    tree = STRtree(polys)

    conflicts = {i: set() for i in range(len(polys))}
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j <= i:
                continue
            ov = p.intersection(polys[j]).area
            smaller_area = min((halves[i] * 2) ** 2, (halves[j] * 2) ** 2)
            if ov > smaller_area * min_overlap_frac:
                conflicts[i].add(j)
                conflicts[j].add(i)

    alive = set(range(len(polys)))
    removed = 0
    while True:
        worst = max(alive, key=lambda i: len(conflicts[i] & alive))
        if not (conflicts[worst] & alive):
            break
        alive.discard(worst)
        removed += 1

    kept_idx = sorted(alive)
    kept_polys = [polys[i] for i in kept_idx]
    return kept_polys, kept_idx, removed


def finalize_nonoverlap_polys(polys, min_overlap_frac=1e-6):
    """Same greedy zero-overlap guarantee as `finalize_nonoverlap`, but for
    an arbitrary list of already-built Polygons (from grow_corners/fill_gaps
    combined) rather than reconstructing squares from centre+angle+half.
    This is the authoritative LAST word on overlap: corner-growth's own
    per-step checks can miss a conflict introduced by a LATER step in the
    same pass (its spatial index is only rebuilt once per pass, not after
    every single corner move), and fill_gaps' tiles are checked against the
    set as it stood at insertion time, not the final grown shapes. Removing
    a whole offending tile here (rather than leaving a partial overlap that
    renders as one tile visibly biting into its neighbour) is the only way
    to guarantee the shipped mosaic is genuinely overlap-free."""
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    if not polys:
        return [], 0
    tree = STRtree(polys)

    conflicts = {i: set() for i in range(len(polys))}
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j <= i:
                continue
            ov = p.intersection(polys[j]).area
            smaller_area = min(p.area, polys[j].area)
            if smaller_area > 0 and ov > smaller_area * min_overlap_frac:
                conflicts[i].add(j)
                conflicts[j].add(i)

    alive = set(range(len(polys)))
    removed = 0
    while True:
        worst = max(alive, key=lambda i: len(conflicts[i] & alive))
        if not (conflicts[worst] & alive):
            break
        alive.discard(worst)
        removed += 1

    kept_idx = sorted(alive)
    return [polys[i] for i in kept_idx], removed


def _reasonable_quad(poly, max_edge_ratio=2.5):
    """True if `poly`'s longest edge isn't disproportionately longer than
    its shortest -- the same square-ish/rectangular-not-wedge constraint
    grow_corners enforces, applied to trim results too."""
    coords = list(poly.exterior.coords[:-1])
    n = len(coords)
    if n < 3:
        return False
    edges = [math.hypot(coords[m][0] - coords[(m + 1) % n][0],
                         coords[m][1] - coords[(m + 1) % n][1])
             for m in range(n)]
    shortest = min(edges)
    if shortest < 1e-9:
        return False
    return max(edges) <= shortest * max_edge_ratio


def trim_overlaps(polys, n_passes=4, min_remaining_frac=0.25):
    """Zero-overlap guarantee that CUTS a tile's offending corner back
    instead of deleting the whole tile -- exactly what a real mosaicist
    does with a custom-cut boundary piece, and far cheaper on coverage than
    `finalize_nonoverlap_polys`' whole-tile removal.

    The cut is a SINGLE STRAIGHT LINE (half-plane clip, positioned midway
    between the two tiles' own centres) rather than subtracting the other
    tile's exact (possibly rotated) shape. An arbitrary-shape subtraction
    can leave a triangular sliver or wedge behind -- exactly the
    trapezoid/wedge-shaped tiles that stood out against the surrounding
    square tiles. A one-line clip can trim at most one corner/edge off a
    square, which keeps the result reading as "square with one corner cut",
    much closer to how a real tessera gets trimmed to fit."""
    from .optimizer import _clip_halfplane

    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if p is not None and not p.is_empty]

    for _pass in range(n_passes):
        tree = STRtree(polys)
        changed = False
        for i in range(len(polys)):
            p = polys[i]
            if p is None or p.is_empty:
                continue
            for j in tree.query(p):
                if j <= i:
                    continue
                q = polys[j]
                if q is None or q.is_empty:
                    continue
                if not p.intersects(q):
                    continue
                ov = p.intersection(q).area
                if ov <= 1e-6:
                    continue

                pc, qc = p.centroid, q.centroid
                mx, my = (pc.x + qc.x) / 2.0, (pc.y + qc.y) / 2.0
                nx_, ny_ = qc.x - pc.x, qc.y - pc.y
                nlen = math.hypot(nx_, ny_) or 1.0
                nx_, ny_ = nx_ / nlen, ny_ / nlen

                # Clip p on the side AWAY from q (normal points p->q, so p
                # keeps dot<=0), and q on the side away from p (opposite
                # normal) — both cuts share the same straight line, so the
                # two RESULTS are on complementary half-planes and provably
                # can't overlap each other. Clipping only ONE side (leaving
                # the other at its full original extent) does NOT give that
                # guarantee — the untouched tile was never actually
                # constrained to its own half of the line, so it can still
                # reach across into the trimmed tile's space. Both sides
                # must always be cut for the zero-overlap property to hold;
                # "who's the intruder" only decides which side gets deleted
                # outright if its remaining sliver is too small to keep.
                trimmed_p = _clip_halfplane(p, (mx, my), (nx_, ny_))
                trimmed_q = _clip_halfplane(q, (mx, my), (-nx_, -ny_))
                # A one-line clip can still leave a skewed sliver depending
                # on where the line happens to fall relative to the tile --
                # reject (treat as "nothing worth keeping") rather than ship
                # a wedge just because it's technically non-overlapping.
                if trimmed_p is not None and not _reasonable_quad(trimmed_p):
                    trimmed_p = None
                if trimmed_q is not None and not _reasonable_quad(trimmed_q):
                    trimmed_q = None
                keep_p = trimmed_p.area if trimmed_p is not None else 0.0
                keep_q = trimmed_q.area if trimmed_q is not None else 0.0

                polys[i] = trimmed_p if keep_p >= p.area * min_remaining_frac else None
                polys[j] = trimmed_q if keep_q >= q.area * min_remaining_frac else None
                changed = True
                p = polys[i]
                if p is None:
                    break

        polys = [pp for pp in polys if pp is not None and not pp.is_empty]
        if not changed:
            break

    return polys


def grow_corners(polys, W, H, max_grow_mult=0.5, n_passes=3, bs_iters=6):
    """Final touch-up: stretch each tile's own corners outward into any
    still-leftover void, turning rigid squares into slightly irregular (but
    still convex, still non-overlapping) quads that cover more of the gap.
    Every corner is tested independently via binary search for the largest
    offset along its own outward (centroid-to-corner) direction that doesn't
    cross into a neighbour or off the canvas. Monotonic (corners only grow),
    so a couple of passes let growth opportunities propagate."""
    polys = list(polys)
    if not polys:
        return polys
    canvas = ShBox(0, 0, W, H)

    # The broad-phase STRtree only reflects positions as of its last build.
    # Rebuilding once per PASS (rather than continuously) meant a tile grown
    # early in a pass was invisible to a later tile's overlap query within
    # that same pass -- its NEW (grown) footprint simply wasn't in the tree
    # yet, so a query against it could miss the conflict entirely (not just
    # use stale data for a candidate it DID find -- the candidate wouldn't
    # even be returned). Since growth is monotonic (corners only ever grow),
    # a later pass's fresh tree can't retroactively undo an overlap a stale
    # tree let happen. Rebuilding every `rebuild_every` tiles within a pass
    # keeps this bounded (not a full rebuild per tile) while keeping the
    # broad-phase honest enough that real conflicts actually get caught.
    rebuild_every = max(1, min(50, len(polys) // 10 or 1))

    for _pass in range(n_passes):
        tree = STRtree(polys)
        for i in range(len(polys)):
            if i > 0 and i % rebuild_every == 0:
                tree = STRtree(polys)
            coords = list(polys[i].exterior.coords[:-1])
            cx = sum(p[0] for p in coords) / len(coords)
            cy = sum(p[1] for p in coords) / len(coords)
            diag = max(math.hypot(px - cx, py - cy) for px, py in coords)
            max_step = diag * max_grow_mult

            new_coords = list(coords)
            changed = False
            for k in range(len(coords)):
                px, py = coords[k]
                dx, dy = px - cx, py - cy
                dl = math.hypot(dx, dy) or 1.0
                ux, uy = dx / dl, dy / dl

                def try_offset(off):
                    trial = list(new_coords)
                    trial[k] = (px + ux * off, py + uy * off)
                    if not canvas.contains(ShPoint(*trial[k])):
                        return None
                    # Moving one corner independently, with no convexity
                    # check, can stretch it past where the ADJACENT corners
                    # already sit -- shapely's is_valid/is_simple only rule
                    # out self-intersection, not a concave "dart"/arrowhead
                    # shape, which is exactly the notched tiles that were
                    # visibly biting into neighbours' space.
                    if not _is_convex(trial):
                        return None
                    # Convexity alone still allows a genuinely skewed
                    # trapezoid/kite (one corner stretched much further than
                    # its neighbours) -- real tesserae read as square-ish or
                    # a modest rectangle, not a wedge. Cap how far this
                    # corner's edges can stretch relative to the OTHER two
                    # edges meeting the opposite corner, so growth stays a
                    # gentle stretch rather than a lopsided point.
                    n_pts = len(trial)
                    edge_lens = [math.hypot(trial[m][0] - trial[(m + 1) % n_pts][0],
                                             trial[m][1] - trial[(m + 1) % n_pts][1])
                                 for m in range(n_pts)]
                    if max(edge_lens) > min(edge_lens) * 2.2:
                        return None
                    cand = Polygon(trial)
                    if not cand.is_valid or not cand.is_simple:
                        return None
                    for j in tree.query(cand):
                        if j == i:
                            continue
                        if cand.intersects(polys[j]) and cand.intersection(polys[j]).area > 1e-6:
                            return None
                    return cand

                lo, hi = 0.0, max_step
                best_cand = None
                full = try_offset(hi)
                if full is not None:
                    best, best_cand = hi, full
                else:
                    best = 0.0
                    for _bs in range(bs_iters):
                        mid = (lo + hi) / 2
                        cand = try_offset(mid)
                        if cand is not None:
                            best, best_cand = mid, cand
                            lo = mid
                        else:
                            hi = mid
                if best > 1e-6 and best_cand is not None:
                    new_coords[k] = (px + ux * best, py + uy * best)
                    changed = True

            if changed:
                cand = Polygon(new_coords)
                if cand.is_valid and cand.is_simple:
                    polys[i] = cand

    return polys


def fill_gaps(polys, topo, W, H, r_min, r_max, min_gap_area_factor=0.3, max_new=None,
              target_cells=3000, min_side_frac=0.3):
    """After row-tracing + corner-growth, real leftover void can still
    remain -- most often long, THIN, WIDENING-AND-NARROWING slivers between
    two differently-oriented row channels (neither side alone can grow a
    corner far enough to close it). A fixed-spacing grid walk mostly MISSES
    a sliver like that -- most sample points just land outside it.

    Instead, rasterize each leftover piece and compute its distance-to-
    boundary at every interior cell (a tiny local distance transform) --
    that's the LOCAL AVAILABLE WIDTH at every point in the gap. Greedily
    place a tile at the currently-widest remaining point, sized to that
    local width, clear the area around it, and repeat. A sliver that widens
    then narrows naturally gets a row of tiles that widens then narrows to
    match, instead of one fixed size that mostly doesn't fit.

    `min_side_frac`: smallest tile this may place, as a fraction of the
    minimum tile side (2 * r_min). The default 0.3 lets a narrowing sliver
    taper well below the user's stated minimum, which is the right
    behaviour for row-tracing's own boundary pieces. Pass 1.0 to forbid
    that entirely and only ever place tiles at or above the configured
    minimum size -- what a caller wants when the user's min/max range is
    meant as a hard guarantee about every tile in the output."""
    from shapely.ops import unary_union
    from scipy.ndimage import distance_transform_edt

    contour_angle = topo['contour_angle']
    local_scale = topo['local_scale']
    if not polys:
        return []

    covered = unary_union(polys)
    canvas = ShBox(0, 0, W, H)
    leftover = canvas.difference(covered)
    pieces = list(leftover.geoms) if hasattr(leftover, 'geoms') else [leftover]

    tree = STRtree(polys)
    new_polys = []
    min_side = 2 * r_min

    def _fits(wx, wy, ang, side):
        half = side / 2.0
        square = make_square(wx, wy, ang, half)
        if any(cx < -1e-6 or cx > W + 1e-6 or cy < -1e-6 or cy > H + 1e-6
               for (cx, cy) in square):
            return None   # never let a tile extend past the canvas edge
        cand = Polygon(square)
        if not cand.is_valid:
            cand = cand.buffer(0)
        conflict = any(
            cand.intersects(polys[j]) and cand.intersection(polys[j]).area > 1e-6
            for j in tree.query(cand)
        )
        if not conflict:
            conflict = any(
                cand.intersects(other) and cand.intersection(other).area > 1e-6
                for other in new_polys
            )
        return None if conflict else cand

    def _try_place(wx, wy, ang, base_side):
        for shrink in (1.0, 0.8, 0.62, 0.48, 0.36, 0.27, 0.2):
            side = base_side * shrink
            if side < min_side * min_side_frac:
                break
            # A hole left by a single dropped tile is only about one tile
            # wide, so a square at exactly the field angle often misses
            # fitting by a hair while a barely-rotated one drops straight
            # in -- that's the isolated one-tile gap in an otherwise regular
            # run. Try a few small rotations (and, for a square, the
            # equivalent quarter-turn) before giving up on this size. Kept
            # small so a filler never reads as misaligned against the run
            # it sits in; a visible gap and a visibly crooked tile are both
            # worse than a tile that's a few degrees off.
            for d_ang in _FILL_ANGLE_TRIES:
                cand = _fits(wx, wy, ang + d_ang, side)
                if cand is not None:
                    return cand, side
        return None, None

    for piece in pieces:
        if piece.area < (min_side ** 2) * min_gap_area_factor:
            continue
        minx, miny, maxx, maxy = piece.bounds
        pw, ph = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
        # Cell size must be able to RESOLVE A TILE, not just fit a fixed
        # cell budget. Sizing purely to ~target_cells (what this did) is
        # fine for a small sliver but catastrophic for the usual case: all
        # the grout in a mosaic is one connected region spanning the whole
        # canvas, so the budget spread over that gave ~36px cells for a
        # ~33px tile -- coarser than the very holes it was meant to find,
        # so genuine tile-sized gaps were invisible to it and never filled
        # (measured: 220 spots where a full min-size tile would fit but
        # none was placed). Cap the cell size against r_min so the distance
        # transform can always see a tile-sized hole, and only fall back to
        # a coarser grid if that would blow past a hard cell ceiling.
        res = max(0.5, min(r_min * 0.4, math.sqrt(pw * ph / target_cells)))
        rw = max(4, int(pw / res) + 2)
        rh = max(4, int(ph / res) + 2)
        if rw * rh > _FILL_MAX_CELLS:
            res = math.sqrt(pw * ph / float(_FILL_MAX_CELLS))
            rw = max(4, int(pw / res) + 2)
            rh = max(4, int(ph / res) + 2)

        # Rasterize by scan-converting the ring geometry rather than asking
        # shapely `contains` per cell -- at this (much finer) resolution the
        # per-point version would be hundreds of thousands of shapely calls
        # per piece.
        mask_u8 = np.zeros((rh, rw), np.uint8)
        _ring = np.asarray(piece.exterior.coords, dtype=np.float64)
        _ring = np.column_stack([(_ring[:, 0] - minx) / res, (_ring[:, 1] - miny) / res])
        cv2.fillPoly(mask_u8, [np.round(_ring).astype(np.int32)], 1)
        for _hole in piece.interiors:
            _h = np.asarray(_hole.coords, dtype=np.float64)
            _h = np.column_stack([(_h[:, 0] - minx) / res, (_h[:, 1] - miny) / res])
            cv2.fillPoly(mask_u8, [np.round(_h).astype(np.int32)], 0)
        mask = mask_u8.astype(bool)
        if not mask.any():
            continue
        dist = distance_transform_edt(mask) * res

        # Greedy "place at the widest remaining spot" needs the CURRENT
        # global argmax after every single placement, but a plain
        # np.argmax(dist) rescans the whole raster every time -- with cell
        # size now capped at r_min scale (needed to resolve tile-sized
        # holes at all, see above) rather than a fixed small cell budget,
        # the raster for the single connected grout region spanning a real
        # mosaic can be millions of cells, and this loop can run for
        # thousands of placements: an O((rw*rh)^2) blowup that reads as the
        # app hanging (reported directly: tile mode still showing the raw
        # circle preview well after "running in the background"). A block-
        # max structure keeps every argmax to the (small) grid of block
        # maxima, and a placement only invalidates the handful of blocks
        # its clearing touches -- not the whole raster -- so cost tracks
        # the number of tiles actually placed, not the raster's own size.
        B = 64
        nby, nbx = (rh + B - 1) // B, (rw + B - 1) // B
        block_max = np.zeros((nby, nbx), dtype=dist.dtype)
        for by in range(nby):
            for bx in range(nbx):
                block_max[by, bx] = dist[by * B:(by + 1) * B, bx * B:(bx + 1) * B].max()

        def _refresh_blocks(y0, y1, x0, x1):
            by0, by1 = max(0, y0 // B), min(nby - 1, (y1 - 1) // B)
            bx0, bx1 = max(0, x0 // B), min(nbx - 1, (x1 - 1) // B)
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    block_max[by, bx] = dist[by * B:(by + 1) * B, bx * B:(bx + 1) * B].max()

        for _ in range(rw * rh):
            by, bx = np.unravel_index(int(np.argmax(block_max)), block_max.shape)
            if block_max[by, bx] <= 0:
                break
            by0, by1 = by * B, min((by + 1) * B, rh)
            bx0, bx1 = bx * B, min((bx + 1) * B, rw)
            sub = dist[by0:by1, bx0:bx1]
            local_ry, local_rx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            ry, rx = by0 + local_ry, bx0 + local_rx
            best_d = float(dist[ry, rx])
            # Stop once the widest remaining spot can't host the smallest
            # tile this caller permits -- tied to the same floor as the
            # shrink loop so a strict caller isn't left scanning gaps it
            # could never legally fill.
            if best_d < r_min * max(0.4, min_side_frac):
                break
            wx = minx + (rx + 0.5) * res
            wy = miny + (ry + 0.5) * res
            # Local available width caps the tile size -- this is what makes
            # the row of fill tiles taper with the gap's own shape, on top
            # of whatever the image's own detail level would ask for.
            base_side = 2.0 * float(np.clip(min(best_d, local_scale(wx, wy)), r_min, r_max))
            ang = contour_angle(wx, wy)
            cand, placed_side = _try_place(wx, wy, ang, base_side)
            if cand is not None:
                new_polys.append(cand)
                if max_new is not None and len(new_polys) >= max_new:
                    return new_polys
            clear_r = max(1, int((placed_side or best_d) / res))
            y0, y1 = max(0, ry - clear_r), ry + clear_r + 1
            x0, x1 = max(0, rx - clear_r), rx + clear_r + 1
            dist[y0:y1, x0:x1] = 0
            _refresh_blocks(y0, y1, x0, x1)

    return new_polys


# ── Top-level orchestrator: matches `_flow_tiles`' external contract ───────

def flow_square_tiles(xs, ys, rs, W, H, topo, r_min, r_max,
                       n_iters=200, rng=None, colors_bgr=None):
    """Full pipeline: trace flowing rows (andamento) along the contour field
    -> guarantee zero overlap -> grow corners into leftover void.
    Returns (polys, source_idx): polys is a list of shapely Polygons,
    source_idx[i] is the index into xs/ys/rs of the nearest original point
    to polys[i]'s centroid (for colour lookup by the caller).

    `colors_bgr`: per-point BGR colour, parallel to xs/ys/rs. When given,
    row DIRECTION is driven by actual colour-boundary gradients (real image
    structure — an eye's curve, hair strands) instead of `topo`'s circle-
    size-based contour_angle, which only encodes local detail DENSITY and
    produces rows that swirl arbitrarily rather than following real features.
    Tile SIZE still comes from `topo`'s local_scale either way.

    `n_iters`/`rng` kept in the signature for backward compatibility with
    the `_flow_tiles` adapter's call site — row tracing is deterministic
    (no RNG, no iteration count) so both are unused here."""
    from scipy.spatial import KDTree

    if colors_bgr is not None and len(colors_bgr) == len(xs):
        color_angle = _build_color_direction_field(xs, ys, colors_bgr, W, H, rs=rs)
        topo = dict(topo)
        topo['contour_angle'] = color_angle

    # Start the first row near the centre of the canvas -- an arbitrary but
    # reasonable place for row growth to radiate outward from; the fallback
    # grid scan in RowTracer.run() picks up whatever that doesn't reach. A
    # fine fallback grid (rather than one sized off the average tile) means
    # independent row FAMILIES start close together in the first place,
    # instead of leaving a big gap between them for corner-growth to try to
    # close after the fact.
    tracer = RowTracer(topo, W, H, r_min, r_max)
    rows = tracer.run((W / 2.0, H / 2.0), extra_seed_grid=max(2.0, r_min * 0.5))
    if not rows:
        return [], []

    flat_pts = [(x, y) for row in rows for (x, y, ang, side) in row]
    flat_ang = [ang for row in rows for (x, y, ang, side) in row]
    flat_half = np.array([side / 2.0 for row in rows for (x, y, ang, side) in row])

    polys, kept_idx, _removed = finalize_nonoverlap(
        flat_pts, flat_ang, flat_half, min_overlap_frac=0.12)

    # Alternate fill -> grow -> trim for a few rounds: each technique opens
    # up opportunity for the others. fill_gaps' own greedy placement doesn't
    # perfectly tile its own leftover remainder; grow_corners then stretches
    # THOSE new tiles' corners into whatever's left; growth can introduce a
    # fresh conflict, which trim_overlaps cuts back (real mosaic boundary
    # pieces get custom-cut, not discarded — far cheaper on coverage than
    # deleting whole tiles, and still a hard overlap guarantee every round).
    # Converges fast in practice (each round's fill_gaps finds less to add
    # than the last); stops early once a round adds nothing.
    all_polys = polys
    for _round in range(4):
        filled = fill_gaps(all_polys, topo, W, H, r_min, r_max)
        if filled:
            all_polys = all_polys + filled
        all_polys = grow_corners(all_polys, W, H, max_grow_mult=0.3, n_passes=2)
        all_polys = trim_overlaps(all_polys)
        if not filled:
            break

    if not all_polys:
        return [], []
    src_tree = KDTree(np.column_stack([xs, ys]))
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in all_polys])
    _, src_idx = src_tree.query(centroids)
    return all_polys, src_idx.tolist()
