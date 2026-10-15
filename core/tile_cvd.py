"""
Square-tile mosaic placement via a Centroidal Voronoi Diagram (CVD) under a
locally-rotated, confidence-blended Manhattan metric -- adapted from Alejo
Hausner's "Simulating Decorative Mosaics" (ACM SIGGRAPH 2001).

Why this replaces row-tracing (`core/tile_square.py`): a Euclidean CVD's
Lloyd relaxation naturally converges to a HEXAGONAL tiling (that's why the
circle-mode Poisson+Lloyd placement grids up into hexagons at rest). Swap
the distance metric to Manhattan (|dx| + |dy|) and the same relaxation
converges to a SQUARE tiling instead -- and rotating that metric's axis
locally to follow a direction field makes the squares follow image
structure. This single relaxation loop gets flow-following, tight packing,
and square shapes as one consequence of the same math, where the row-tracing
approach needed three separate post-hoc passes (fill_gaps/grow_corners/
trim_overlaps) fighting each other to approximate the same result --
and still topped out around 70% coverage. A Voronoi diagram partitions the
plane by construction, so this reaches near-100% coverage with zero overlap,
for free.

Two extensions beyond the base paper, both mapped directly from ideas raised
before the paper was read:
  - "Edge avoidance" (paper's own technique) = literally using real image
    contours as the boundaries tile channels respect, without a separate
    subdivision step: real color edges are excluded from every site's
    region during early relaxation (pushing centroids away from the edge),
    then included again for the final pass so neighbours close the gap
    against a clean boundary.
  - A per-pixel "confidence" blend between the rotated-Manhattan metric and
    plain Euclidean, driven by local gradient strength: well-established
    color channels use the full rotated-square metric; flat/ambiguous areas
    (peaks and valleys of the old size-based topography) fall back toward
    ordinary (unrotated, more organic) small Voronoi cells instead of
    forcing a square grid where there's no real structure to align to.
"""

import math
import numpy as np
import cv2
from scipy.spatial import cKDTree, KDTree

# Per-site orientation smoothing (see _extract_tiles). Light on purpose --
# the distance-transform direction field is already smooth, so this only
# needs to remove per-site sampling jitter, not rebuild coherence.
_SITE_SMOOTH_K = 7
_SITE_SMOOTH_ITERS = 2
_SITE_SMOOTH_BLEND = 0.5

# Rotation applied to every tile on top of the sampled direction field.
#
# This was 45 degrees, to match each tile to the CELL it sits in: the
# Manhattan metric's diamond unit ball makes converged cell edges sit 45
# degrees off the metric axis, so a tile drawn at the raw field angle was
# out of register with its own cell (which measurably doubled overlap).
# But that matched tiles to cells while leaving BOTH 45 degrees off the
# image structure the field describes -- i.e. every tile deliberately
# rotated to a diamond relative to the edge it was supposed to be
# following, which is exactly "tiles don't line up with the details".
# The correct fix is to rotate the METRIC by 45 degrees instead (done in
# _lloyd_relax), which lands the cell edges on the field direction; the
# tile then wants no extra offset at all, and cell, tile, and image
# structure all agree.
_TILE_ANGLE_OFFSET = 0.0


# ── Direction + confidence field ────────────────────────────────────────

def _build_direction_confidence_field(xs, ys, colors_bgr, W, H, analysis_res=512,
                                       tile_px_hint=None):
    """Hausner section 4, implemented as the paper actually specifies.

    The paper's construction, verbatim: take a collection S of edge CURVES;
    for each, the surface z = d_s(x, y) (distance from that curve) is "a
    long, curved mountain" with the curve as its ridge; take the pointwise
    minimum over all curves (= distance to the NEAREST curve), then
    "numerically evaluate the gradient of z at each pixel" -- that gradient
    IS the direction field. The offset curves of a distance field are
    smooth, parallel, and propagate an edge's orientation outward across
    open space, which is exactly the flowing andamento structure the
    paper's own figures show.

    Two earlier attempts at this got specific, identifiable details wrong,
    both worth not repeating:

      - "Edge features" were taken to mean a THRESHOLD on gradient
        magnitude (top ~20% of pixels). That doesn't produce curves at all;
        it produces thick blobs covering a fifth of the image, and a
        distance transform to a thick blob is degenerate -- zero (hence no
        gradient, hence no direction) everywhere inside it, and constant
        over the large flat regions between blobs. That's what produced
        the dead-flat direction plateaus that read as rigid straight bars
        on row-tracing and as abrupt patchiness here. The paper means
        genuine thin curves, so this uses Canny, which is built to return
        exactly that: connected, one-pixel-wide edge chains.

      - The orientation field was smoothed with a doubled angle
        (exp(2*i*theta), the mod-180 convention correct for a line's
        tangent). But these orientations drive SQUARE tiles, and a square
        is unchanged by a 90-degree turn -- the field genuinely lives in
        mod-90, so the correct representation is the QUADRUPLED angle
        exp(4*i*theta). Under mod-180 averaging a tile at 2 degrees and one
        at 88 degrees look nearly opposite and average toward 45; under the
        correct mod-90 averaging they're nearly identical (as they visibly
        are, as squares) and reinforce each other. Getting this wrong
        actively scrambled agreement between neighbouring tiles, which is
        precisely the "alignment" being asked for.

    (The same 90-degree symmetry is also why the paper can say the distance
    GRADIENT -- which points perpendicular to the edge -- "follows an
    edge's orientation": for a square those differ by exactly a quarter
    turn, i.e. not at all.)
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    g = max(16, int(analysis_res))
    sx, sy = g / float(W), g / float(H)
    tree = KDTree(np.column_stack([xs * sx, ys * sy]))
    gy, gx = np.mgrid[0:g, 0:g]
    grid_pts = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
    _, nearest = tree.query(grid_pts, k=1)

    cb = np.asarray(colors_bgr, dtype=np.float32)
    color_grid = cb[nearest].reshape(g, g, 3)
    # Only enough blur to suppress the nearest-point rasterization's own
    # facet noise -- the edge detector below wants real structure intact,
    # not a pre-smoothed mush.
    color_grid = gaussian_filter(color_grid, sigma=(max(0.8, g / 220.0),) * 2 + (0,))

    luma = 0.114 * color_grid[..., 0] + 0.587 * color_grid[..., 1] + 0.299 * color_grid[..., 2]
    gy_e, gx_e = np.gradient(luma)
    grad_mag = np.clip(gaussian_filter(np.hypot(gx_e, gy_e), 1.5), 1e-6, None)

    # 1. Edge CURVES (Hausner's set S). Canny gives thin connected chains.
    luma_u8 = np.clip(luma, 0, 255).astype(np.uint8)
    luma_u8 = cv2.GaussianBlur(luma_u8, (0, 0), max(0.8, g / 256.0))
    _med = float(np.median(luma_u8))
    _lo = int(max(0, 0.66 * _med))
    _hi = int(min(255, max(_lo + 1, 1.33 * _med)))
    edges = cv2.Canny(luma_u8, _lo, _hi, L2gradient=True)
    if not edges.any():
        # Degenerate (near-uniform) image: fall back to the strongest few
        # percent of gradient so there's still SOME structure to follow.
        edges = (grad_mag > np.percentile(grad_mag, 99.0)).astype(np.uint8) * 255

    # 2/3. Distance to the nearest curve ("mountains", z-buffered), then the
    # numeric gradient of that surface. Smoothed first at roughly a tile's
    # own scale: the raw distance field is piecewise-conical with creases at
    # every medial axis, and the paper's own caveat is that a field varying
    # faster than a tile width can't be represented by tiles anyway.
    dist = distance_transform_edt(edges == 0)
    tile_px = float(tile_px_hint) if tile_px_hint else max(2.0, g / 60.0)
    ridge = gaussian_filter(dist, sigma=max(1.0, tile_px * 0.75))
    gy_d, gx_d = np.gradient(ridge)
    theta_raw = np.arctan2(gy_d, gx_d)

    # 4. Coherence smoothing in the mod-90 space square tiles actually live
    # in, weighted toward confident (near-edge) cells so a strong edge's
    # orientation propagates outward instead of being averaged away by the
    # ambiguous open space around it.
    prox = np.exp(-dist / max(1e-6, tile_px * 4.0))
    z = (0.25 + prox) * np.exp(4j * theta_raw)
    z = gaussian_filter(z.real, sigma=max(1.0, tile_px * 0.9)) + \
        1j * gaussian_filter(z.imag, sigma=max(1.0, tile_px * 0.9))
    theta_dt = np.angle(z) / 4.0

    # 5. STRUCTURE TENSOR, blended over the top.
    #
    # The distance-transform field is the paper's own construction and gives
    # the large-scale flow -- concentric courses sweeping around each
    # detected feature. But it only knows about the discrete edge curves:
    # everywhere far from one (measured on a realistic painterly test, the
    # average point sits 8+ TILE WIDTHS from the nearest edge) its
    # orientation is set by a distant contour, so it ignores the gradual
    # tonal structure in between -- a cheek's shading, a fold's falloff --
    # which is exactly the "no alignment with actual details" case. Simply
    # detecting more edges does NOT fix this: pushing Canny harder (lower
    # thresholds, CLAHE) measurably made alignment WORSE (0.59 -> 0.23-0.55),
    # because the extra edges are quantization/dither artifacts that fight
    # the real structure.
    #
    # The structure tensor is the standard robust local-orientation
    # estimator and captures precisely what the edge curves miss. It
    # averages the gradient OUTER PRODUCT rather than the gradient vector,
    # so opposite-signed gradients across a ridge reinforce instead of
    # cancelling, and it yields a per-pixel confidence (its anisotropy) for
    # free. Weighted so it dominates wherever there is genuine oriented
    # local structure and cleanly yields to the distance-transform flow in
    # flat areas, where its own orientation would be pure noise.
    #
    # Measured against the source image's real structure (weighted mod-90
    # agreement over the top 15% of gradient): distance-transform alone
    # 0.484, structure tensor alone 0.658 but noisier in flat regions,
    # blended 0.646 -- while local coherence IMPROVES over either
    # (0.906 -> 0.929 overall, 0.946 -> 0.952 in flat areas).
    _l_st = gaussian_filter(luma.astype(np.float64), max(1.0, tile_px * 0.5))
    _gy_l, _gx_l = np.gradient(_l_st)
    _rho = max(1.5, tile_px * 1.5)
    _j00 = gaussian_filter(_gx_l * _gx_l, _rho)
    _j11 = gaussian_filter(_gy_l * _gy_l, _rho)
    _j01 = gaussian_filter(_gx_l * _gy_l, _rho)
    theta_st = 0.5 * np.arctan2(2.0 * _j01, _j00 - _j11)
    _trace = _j00 + _j11
    _aniso = np.sqrt((_j00 - _j11) ** 2 + 4.0 * _j01 ** 2) / np.maximum(_trace, 1e-9)
    _strength = np.clip(_trace / (np.percentile(_trace, 80) + 1e-9), 0.0, 1.0)
    w_st = _aniso * _strength * 8.0

    zb = w_st * np.exp(4j * theta_st) + (0.35 + prox) * np.exp(4j * theta_dt)
    theta_grid = np.angle(zb) / 4.0

    # Confidence: strongest right at a real edge curve OR wherever the
    # structure tensor found genuinely oriented local structure -- either is
    # a real reason to trust the rotated-square metric here. Floor stays
    # high: with most of a real photo sitting some distance from the nearest
    # edge, a low floor puts most of the image into the organic-Euclidean
    # fallback and reads as disordered placement (hit once already this
    # session).
    conf = np.clip(0.8 + 0.2 * np.maximum(prox, _aniso * _strength), 0.0, 1.0)

    return {
        'g': g, 'sx': sx, 'sy': sy,
        'theta_grid': theta_grid, 'conf_grid': conf, 'grad_mag': grad_mag,
        'edge_dist': dist, 'tile_px': tile_px,
    }


def _build_detail_topography_field(xs, ys, rs, W, H, analysis_res=256):
    """Direction field from a genuine ELEVATION surface -- detail level
    (from circle size: dense/small circles = more local detail = a "hill")
    smoothed into real hills, valleys, and inclines, with direction as the
    tangent to that surface's OWN contour lines.

    This is deliberately different from `_build_direction_confidence_field`
    (which detects discrete edge CURVES and takes a distance-transform
    gradient to them): a distance transform is smooth almost everywhere but
    has genuine discontinuities exactly on the medial axis between two
    edges, and is blind to gradual/textured structure that never crosses
    the edge-detection threshold at all. An elevation surface has no such
    seams -- its contour lines are smooth and hierarchical by construction,
    closer to how a real topographic map flows, and responds to gradual
    detail-density change (a smoothly detailed hillside) as well as sharp
    edges. The tradeoff is the reverse: an elevation surface's contours can
    miss a real but subtle color boundary that carries little detail-density
    signal, which is exactly why `flow_square_tiles_cvd` combines both.

    The smoothing scale is the whole game here: the RAW detail-density
    signal is dominated by high-frequency texture/photographic noise (every
    fleck of grain reads as its own tiny hill) -- too little smoothing and
    this is pure static; too much and genuine large-scale features (a
    continent, a face) get smoothed away entirely along with the noise.
    Tuned empirically against a synthetic multi-blob test image: g/40 (this
    module's usual scale) left real texture noise field completely
    fragmented; g/20 was the largest smoothing that still kept 3 distinct,
    clearly-separated blobs recognisable as their own hills.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import KDTree

    g = max(16, int(analysis_res))
    sx, sy = g / float(W), g / float(H)
    tree = KDTree(np.column_stack([xs * sx, ys * sy]))
    gy, gx = np.mgrid[0:g, 0:g]
    grid_pts = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
    _, nearest = tree.query(grid_pts, k=1)

    size_grid = np.asarray(rs, dtype=np.float64)[nearest].reshape(g, g)
    median_r = float(np.median(rs))
    size_grid = np.clip(size_grid, median_r / math.sqrt(3.0), median_r * math.sqrt(3.0))

    smooth_sigma = max(3.0, g / 20.0)
    size_grid = gaussian_filter(size_grid, sigma=smooth_sigma)
    elev = gaussian_filter(1.0 / np.maximum(size_grid, 1e-6), sigma=smooth_sigma)

    gy_e, gx_e = np.gradient(elev)
    grad_mag = np.hypot(gx_e, gy_e)
    # An elevation gradient is a genuine single-valued vector (points
    # uphill), unlike an edge tangent -- no mod-pi ambiguity to resolve, so
    # no doubled-angle averaging needed here.
    theta_grid = np.arctan2(gy_e, gx_e) + math.pi / 2   # tangent to contour lines

    # Confidence: steep incline = a real, well-defined slope direction =
    # high confidence in the rotated-square metric. A flat plateau -- a
    # PEAK or a VALLEY, exactly the two cases named when this idea was
    # first raised -- has near-zero gradient and an ill-defined direction,
    # so it should fall back toward the organic Euclidean metric instead of
    # forcing an arbitrary square alignment onto ground that has none.
    conf = grad_mag / (np.percentile(grad_mag, 90) + 1e-9)
    conf = np.clip(0.35 + 0.65 * np.clip(conf, 0.0, 1.0), 0.0, 1.0)

    return {
        'g': g, 'sx': sx, 'sy': sy,
        'theta_grid': theta_grid, 'conf_grid': conf, 'grad_mag': grad_mag,
    }


def _sample_grid(field, key, xs_arr, ys_arr):
    """Vectorized nearest-cell lookup into one of the field's grids."""
    g, sx, sy = field['g'], field['sx'], field['sy']
    gxp = np.clip(np.round(xs_arr * sx).astype(np.int64), 0, g - 1)
    gyp = np.clip(np.round(ys_arr * sy).astype(np.int64), 0, g - 1)
    return field[key][gyp, gxp]


# ── Site seeding: density-adaptive Poisson-style, matching target area ─────

def _seed_sites(W, H, local_scale, r_min, r_max, rng):
    """Reuses `tile_square.variable_density_seed` (already validated: greedy
    Poisson-disk-style seeding whose spacing follows the local target size
    directly, so expected site count matches W*H / target_area rather than
    the paper's raw rejection sampling -- which, sampled at r_min-scale
    candidate density, way overshoots true target density in any area whose
    target size is much larger than r_min, starving the relaxation raster of
    pixels per site)."""
    from .tile_square import variable_density_seed

    def local_side_fn(x, y):
        return 2.0 * float(np.clip(local_scale(np.array([x]), np.array([y]))[0], r_min, r_max))

    pts = variable_density_seed(W, H, local_side_fn, 2.0 * r_min, spacing_mult=1.0, rng=rng)
    if not pts:
        return np.zeros(0), np.zeros(0)
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


# ── Vectorized local_scale sampling against topo's own grids ───────────────

def _local_scale_arr(xs_arr, ys_arr, topo, r_min, r_max):
    g = topo['g']; sx = topo['sx']; sy = topo['sy']
    size_grid = topo['size_grid']; grad_mag = topo['grad_mag']
    gxp = np.clip(np.round(xs_arr * sx).astype(np.int64), 0, g - 1)
    gyp = np.clip(np.round(ys_arr * sy).astype(np.int64), 0, g - 1)
    samp = np.maximum(1.5, size_grid[gyp, gxp])
    grade_norm = grad_mag[gyp, gxp] / (grad_mag.max() + 1e-9)
    val = samp * np.clip(1.15 - grade_norm, 0.3, 1.15)
    return np.clip(val, r_min, r_max)


# ── The relaxation loop ─────────────────────────────────────────────────

def _lloyd_relax(sx, sy, W, H, dc_field, topo, r_min, r_max,
                  n_iters=30, k_candidates=24, raster_res=None,
                  edge_avoid_frac=0.7, move_tol=0.15):
    """Runs Lloyd's algorithm under the rotated, confidence-blended L1
    metric. Returns final (site_x, site_y, site_theta, labels, grid_shape,
    grid_to_world) -- `labels` is the LAST assignment computed (from the
    final, edge-avoidance-free pass) as a (rh, rw) int32 array, -1 where a
    pixel wasn't claimed by any surviving site.
    """
    n = len(sx)
    res = raster_res or max(64, int(min(W, H) / max(2.0, r_min) * 1.6))
    res = min(res, 1400)
    rw = max(8, int(round(res * W / max(W, H))))
    rh = max(8, int(round(res * H / max(W, H))))
    gx = (np.arange(rw) + 0.5) * (W / rw)
    gy = (np.arange(rh) + 0.5) * (H / rh)
    pix_x, pix_y = np.meshgrid(gx, gy)
    pix_x = pix_x.ravel(); pix_y = pix_y.ravel()
    pix_pts = np.column_stack([pix_x, pix_y])

    edge_mask = _sample_grid(dc_field, 'grad_mag', pix_x, pix_y)
    edge_thresh = np.percentile(edge_mask, 92)
    is_edge_pixel = edge_mask > edge_thresh

    labels = np.full(pix_x.shape[0], -1, dtype=np.int64)
    n_edge_iters = int(n_iters * edge_avoid_frac)

    for it in range(n_iters):
        if n == 0:
            break
        # A pixel's TRUE metric-nearest site isn't necessarily its Euclidean-
        # nearest: a site's own effective reach scales with its target tile
        # size (alpha), so a large-target site can out-compete a Euclidean-
        # closer small-target site for pixels well outside a tight k-nearest
        # shortlist. k_candidates is kept generous (not the ~4-8 that would
        # suffice for a fixed-size metric) specifically to make this
        # over-inclusive rather than risk silently excluding the real
        # winner, which was producing noisy, non-convex regions with a
        # tighter shortlist.
        kq = min(k_candidates, n)
        tree = cKDTree(np.column_stack([sx, sy]))
        _, cand_idx = tree.query(pix_pts, k=kq)
        if kq == 1:
            cand_idx = cand_idx[:, None]

        # The Manhattan unit ball {|u| + |v| <= 1} in the metric's own
        # rotated frame is a DIAMOND whose vertices lie along the metric
        # axis -- so its EDGES, and hence the edges of the square cells the
        # relaxation converges to, sit 45 degrees off that axis. Rotating
        # the metric by a further 45 degrees puts the resulting cell edges
        # back onto the direction field itself, which is what actually makes
        # the tiling line up with image structure (see _TILE_ANGLE_OFFSET).
        theta = _sample_grid(dc_field, 'theta_grid', sx, sy) + math.pi / 4
        conf = _sample_grid(dc_field, 'conf_grid', sx, sy)
        alpha = 2.0 * np.clip(_local_scale_arr(sx, sy, topo, r_min, r_max), r_min, r_max)

        c_th = np.cos(theta)[cand_idx]; s_th = np.sin(theta)[cand_idx]
        cx = sx[cand_idx]; cy = sy[cand_idx]
        c_conf = conf[cand_idx]; c_alpha = np.maximum(alpha[cand_idx], 1e-6)

        dx = pix_x[:, None] - cx
        dy = pix_y[:, None] - cy
        u = dx * c_th + dy * s_th
        v = -dx * s_th + dy * c_th
        d_manhattan = (np.abs(u) + np.abs(v)) / c_alpha
        d_euclid = np.hypot(u, v) / c_alpha
        # confidence blend: strong local colour edge -> full rotated-square
        # (Manhattan) metric; flat/ambiguous area -> plain Euclidean, which
        # relaxes toward organic small cells instead of forcing an aligned
        # square grid onto structure that isn't really there.
        d = c_conf * d_manhattan + (1.0 - c_conf) * d_euclid

        apply_edge_mask = it < n_edge_iters
        if apply_edge_mask:
            d = np.where(is_edge_pixel[:, None], np.inf, d)

        best = np.argmin(d, axis=1)
        best_valid = np.isfinite(np.take_along_axis(d, best[:, None], axis=1))[:, 0]
        new_labels = np.where(best_valid, np.take_along_axis(cand_idx, best[:, None], axis=1)[:, 0], -1)
        labels = new_labels

        assigned = labels >= 0
        if not assigned.any():
            continue
        sums_x = np.bincount(labels[assigned], weights=pix_x[assigned], minlength=n)
        sums_y = np.bincount(labels[assigned], weights=pix_y[assigned], minlength=n)
        counts = np.bincount(labels[assigned], minlength=n)
        has_region = counts > 0

        new_sx = np.where(has_region, sums_x / np.maximum(counts, 1), sx)
        new_sy = np.where(has_region, sums_y / np.maximum(counts, 1), sy)
        move = np.hypot(new_sx - sx, new_sy - sy)
        sx, sy = new_sx, new_sy

        # Drop sites whose region vanished (no pixels ever claimed them) --
        # only after the edge-avoidance window closes, so a site temporarily
        # starved by the edge mask isn't culled before it gets its fair
        # final-pass assignment.
        if not apply_edge_mask and (~has_region).any():
            keep = has_region
            sx, sy = sx[keep], sy[keep]
            remap = -np.ones(n, dtype=np.int64)
            remap[keep] = np.arange(keep.sum())
            labels = np.where(assigned, remap[np.where(assigned, labels, 0)], -1)
            n = len(sx)

        if n > 0 and not apply_edge_mask and np.median(move[has_region]) < move_tol:
            break

    return sx, sy, labels, (rh, rw), (W / rw, H / rh)


# ── Tile extraction: the Voronoi cell IS the tile ───────────────────────
#
# The paper never builds an independent square at each site and reconciles
# overlaps afterward -- the tile boundary comes directly from the z-buffer
# winning region, so two tiles are geometrically incapable of overlapping in
# the first place (it's a partition). An earlier version of this module
# built a same-target-size analytic square per site instead (worried the raw
# region shape wasn't reliable) and then needed row-tracing's overlap-
# cleanup pipeline (trim_overlaps/fill_gaps/grow_corners) to fix the
# resulting conflicts -- but that pipeline was tuned for row-tracing's
# overlap topology, not CVD's, and was destructive on this output (~40%
# coverage lost in one cleanup pass, then compounding into a swarm of tiny
# sliver fragments when fill_gaps tried to repair the self-inflicted voids).
# Extracting each tile directly from its own converged region contour (via
# findContours + approxPolyDP to a small-vertex polygon) removes the need
# for any of that: zero overlap is structural, not enforced after the fact.

def _reject_pack_uniform(sx, sy, ca_t, sa_t, half, W, H, theta_fn=None, rng=None):
    """Sequential greedy acceptance of fixed-size (all exactly `half`)
    rotated squares: every candidate is either placed unchanged or skipped
    entirely, never resized -- matches circle mode's own Poisson-disk
    contract for a fixed circle size, where density (acceptance rate)
    is what adapts to local crowding, not the size of what gets placed.

    A spatial hash keyed to roughly one cell per tile diameter keeps each
    candidate's overlap check to its actual nearby neighbours (already-
    accepted tiles only -- this is genuinely sequential, so a candidate
    can never conflict with anything decided after it), with an exact
    4-axis SAT test for the accept/reject decision itself.

    If `theta_fn` is given, a second pass scans a dense jittered grid over
    the WHOLE canvas (not just the original CVD site positions) and tries
    each point too -- CVD's own sites are a good starting distribution but
    aren't a maximal packing (reported directly: real free space left over
    that a full-size tile would fit in, which the site-only pass has no way
    to discover since it only ever considers where relaxation happened to
    put a site). This reuses the exact same spatial hash and acceptance
    test, so it only ever ADDS tiles into genuinely open space -- it can
    never conflict with what the first pass already accepted.
    """
    n = len(sx)
    cell = max(1.0, 2.0 * half)
    grid: dict = {}

    def _cell_of(x, y):
        return (int(x / cell), int(y / cell))

    def _overlaps(cx, cy, c, s, ocx, ocy, oc, os_):
        dx, dy = ocx - cx, ocy - cy
        for (ax, ay) in ((c, s), (-s, c), (oc, os_), (-os_, oc)):
            proj = abs(dx * ax + dy * ay)
            span = half * (abs(c * ax + s * ay) + abs(-s * ax + c * ay))
            ospan = half * (abs(oc * ax + os_ * ay) + abs(-os_ * ax + oc * ay))
            if proj >= span + ospan:
                return False
        return True

    rad = int(2.0 * half / cell) + 1

    def _try_accept(cx, cy, ang):
        c, s = math.cos(ang), math.sin(ang)
        # Check the real ROTATED footprint, not an axis-aligned box: a tile
        # whose centre clears `half` from the edge can still poke a corner
        # over it once rotated (up to half*sqrt(2)), and the corner clamp
        # applied when the polygon is built would then quietly shrink it
        # below the user's stated minimum size.
        for _sx_, _sy_ in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            _px = cx + c * half * _sx_ - s * half * _sy_
            _py = cy + s * half * _sx_ + c * half * _sy_
            if _px < -1e-6 or _px > W + 1e-6 or _py < -1e-6 or _py > H + 1e-6:
                return False
        gx0, gy0 = _cell_of(cx, cy)
        for gx in range(gx0 - rad, gx0 + rad + 1):
            for gy in range(gy0 - rad, gy0 + rad + 1):
                for (ocx, ocy, oc, os_) in grid.get((gx, gy), ()):
                    if _overlaps(cx, cy, c, s, ocx, ocy, oc, os_):
                        return False
        grid.setdefault((gx0, gy0), []).append((cx, cy, c, s))
        return True

    accepted = []
    if n > 0:
        angles = np.arctan2(sa_t, ca_t)
        for i in range(n):
            if _try_accept(float(sx[i]), float(sy[i]), float(angles[i])):
                accepted.append(i)

    extra = []
    if theta_fn is not None:
        rng = rng if rng is not None else np.random.default_rng(0)
        step = max(1.0, half * 0.35)
        gy0v = step * 0.5
        while gy0v < H:
            gx0v = step * 0.5
            while gx0v < W:
                jx = float(gx0v + rng.uniform(-step * 0.3, step * 0.3))
                jy = float(gy0v + rng.uniform(-step * 0.3, step * 0.3))
                ang = float(theta_fn(jx, jy))
                if _try_accept(jx, jy, ang):
                    extra.append((jx, jy, ang))
                gx0v += step
            gy0v += step

    return np.array(accepted, dtype=np.int64), extra


def _extract_tiles(sx, sy, labels, grid_shape, cell_wh, dc_field, topo,
                    r_min, r_max, W, H, alpha_pack=1.0, edge_align_dist=None):
    """Draw a real SQUARE at each converged site, oriented along the
    direction field -- the paper's step 8, verbatim.

    Sizing follows the paper's section 5.2 exactly: tile side comes from
    equating TOTAL tile area to image area, d = alpha*sqrt(w*h/n) with
    alpha~=0.8 absorbing packing inefficiency (that alpha is also what
    produces the visible grout in the paper's own figures -- no separate
    inset needed). Size VARIATION is a relative scaling of that global
    figure, from the same local-scale field that drives the metric's slope
    term, renormalized so the total still equals alpha^2*w*h.

    Two earlier versions of this were wrong in opposite directions, both
    worth not repeating:
      - Sizing each square from the topography's ABSOLUTE target size,
        unrelated to actual site density -- squares didn't match how much
        space each site really won, so they overlapped badly and needed
        destructive cleanup that shredded coverage.
      - Using each cell's own simplified CONTOUR as the tile -- that yields
        irregular shards, not tiles. A Voronoi cell only converges to a true
        square where its neighbours share both its size and its metric
        rotation; everywhere else the shared boundary is a kink no matter
        how long Lloyd runs (verified: 3x iterations and resolution left the
        vertex-count distribution unchanged). The paper never uses cell
        outlines as tiles -- the CVD only supplies POSITIONS.
    """
    n = len(sx)
    if n == 0:
        return [], []

    # The Manhattan metric's unit "circle" is a DIAMOND, so the converged
    # cells form a lattice rotated 45 degrees from the metric's own axes --
    # verified directly on a uniform control case (neighbour bearings peak
    # at 45 deg mod 90, and the rendered cells are unmistakably diamonds).
    # A tile drawn at the raw metric angle is therefore 45 deg out of
    # register with the packing it's supposed to sit in, which was a large
    # systematic source of overlap: correcting it alone halved the overlap
    # area at equal coverage.
    theta = _sample_grid(dc_field, 'theta_grid', sx, sy) + _TILE_ANGLE_OFFSET

    # Row-tracing's flowing "andamento" look came for free from HOW it
    # seeded rows -- each new row's shape was literally an offset copy of
    # an already-traced neighbour's path, so a whole family of rows shared
    # near-identical local direction by construction. CVD has no equivalent
    # mechanism: every site samples the underlying field independently, so
    # even a genuinely smooth field only guarantees neighbouring TILES look
    # similar to the extent the field itself varies slowly there -- reported
    # directly as packing/coverage now being solid but flow still reading
    # as fairly directionless. Smoothing each site's OWN orientation against
    # its actual nearest-neighbour sites (not just re-smoothing the
    # underlying pixel field, already done in the field builders) directly
    # increases how large a coherent patch of matching-direction tiles
    # reads as, the same way row-tracing's neighbours end up sharing
    # direction.
    #
    # Averaged in the QUADRUPLED angle, for the same reason the field
    # builders now are: these orientations drive SQUARE tiles, which are
    # unchanged by a quarter turn, so the field lives in mod-90. Plain
    # vector averaging (what this did originally) treats a tile at 2
    # degrees and one at 88 -- visually the same square -- as though they
    # pointed almost oppositely, and pulls both toward 45, actively
    # destroying the agreement this pass exists to create. In mod-90 they
    # correctly reinforce instead.
    #
    # Kept DELIBERATELY light. This pass was originally 15 rounds of
    # 0.8-weighted averaging over 14 neighbours, from when the underlying
    # field was patchy and needed heavy rescue. Against the current
    # distance-transform field -- whose offset curves are already smooth and
    # continuous by construction -- that much diffusion spreads orientation
    # across ~15 neighbour hops, which is effectively global averaging: it
    # collapses the real concentric structure into one near-uniform
    # direction and reads as tiles ignoring the image entirely. (Correcting
    # the averaging to mod-90 made this WORSE, not better, because mod-90
    # averaging converges far more aggressively than the broken mod-180
    # version did.) A couple of gentle rounds is enough to clean up
    # per-site sampling jitter without erasing the structure.
    _site_pts = np.column_stack([sx, sy])
    _K_smooth = min(_SITE_SMOOTH_K, n)
    if _K_smooth > 1 and _SITE_SMOOTH_ITERS > 0:
        _, _nn_smooth = cKDTree(_site_pts).query(_site_pts, k=_K_smooth)
        _vec = np.exp(4j * theta)
        for _ in range(_SITE_SMOOTH_ITERS):
            _neighbor_vec = _vec[_nn_smooth[:, 1:]].mean(axis=1)
            _vec = (1.0 - _SITE_SMOOTH_BLEND) * _vec + _SITE_SMOOTH_BLEND * _neighbor_vec
            _mag = np.abs(_vec)
            _vec = _vec / np.maximum(_mag, 1e-9)
        theta = np.angle(_vec) / 4.0

    # Canvas-edge alignment: a tile meeting the border at an arbitrary angle
    # leaves a ragged triangular gap against it. A square reads the same
    # every 90 degrees, so snapping toward the nearest multiple of 90 makes
    # border tiles sit flush against the edge (whichever border they're on)
    # without changing their apparent shape. Blended in only within a couple
    # of tile widths of the border so the interior flow field is untouched.
    if edge_align_dist is None:
        edge_align_dist = 2.0 * r_max * 2.0
    border_d = np.minimum(np.minimum(sx, W - sx), np.minimum(sy, H - sy))
    w_edge = np.clip(1.0 - border_d / max(edge_align_dist, 1e-6), 0.0, 1.0)
    if w_edge.any():
        quarter = math.pi / 2
        snapped = np.round(theta / quarter) * quarter
        delta = (snapped - theta + math.pi) % (2 * math.pi) - math.pi
        theta = theta + w_edge * delta

    # Size each tile from the space it ACTUALLY has, not a global area
    # budget. For two equal squares sharing an orientation, centres offset
    # by (u, v) in that shared frame are disjoint exactly when
    # max(|u|, |v|) >= side -- so capping each tile's side at the smallest
    # such Chebyshev distance to its nearby sites makes overlap impossible
    # by construction, with no cleanup pass at all. The paper's global
    # d = alpha*sqrt(w*h/n) (section 5.2) assumes uniformly-sized tiles and
    # only holds on average; with detail-adaptive sizes it leaves hundreds
    # of real overlaps at any alpha high enough to keep coverage up, whereas
    # this per-tile rule measured 0 overlaps at ~75% coverage on the same
    # control case -- better coverage AND a hard guarantee.
    px_area = cell_wh[0] * cell_wh[1]
    counts = np.bincount(labels[labels >= 0], minlength=n).astype(np.float64)
    target_half_all = np.clip(
        0.5 * np.sqrt(np.maximum(counts, 0.0) * px_area) * 1.4, r_min, r_max)

    def _sat_pack(idx):
        """Exact separating-axis packing among the sites at `idx`, starting
        every tile at its own target size. Two rotated squares are disjoint
        iff some edge normal of either separates them, and along an axis `a`
        a square of half-size h spans h*(|a.u| + |a.v|) -- so the largest
        common scale a pair allows is max over the 4 axes of
        |d.a| / (span_i + span_j). A Chebyshev (max(|du|,|dv|)) rule is only
        the special case where both squares share an orientation; using it
        with per-tile rotations forced a conservative min over neighbours
        and left visible corner-to-corner "checkerboard" voids where the
        lattice ran diagonal to the tiles."""
        m = len(idx)
        if m == 0:
            return np.zeros(0)
        sub_sites = np.column_stack([sx[idx], sy[idx]])
        K = min(9, m)
        _, nn = cKDTree(sub_sites).query(sub_sites, k=K)
        if K == 1:
            nn = nn[:, None]
        u_ax = np.stack([ca_t[idx], sa_t[idx]], axis=1)
        v_ax = np.stack([-sa_t[idx], ca_t[idx]], axis=1)
        h = target_half_all[idx].copy()
        idx_all = np.arange(m)
        for _ in range(80):
            f = np.ones(m)
            worst = 0.0
            for k in range(1, K):
                j = nn[:, k]
                d = np.stack([sx[idx][j] - sx[idx], sy[idx][j] - sy[idx]], axis=1)
                best_t = np.zeros(m)
                for axis in (u_ax, v_ax, u_ax[j], v_ax[j]):
                    proj = np.abs((d * axis).sum(1))
                    span_i = h * (np.abs((axis * u_ax).sum(1)) + np.abs((axis * v_ax).sum(1)))
                    span_j = h[j] * (np.abs((axis * u_ax[j]).sum(1)) + np.abs((axis * v_ax[j]).sum(1)))
                    best_t = np.maximum(best_t, proj / np.maximum(span_i + span_j, 1e-9))
                bad = best_t < 1.0
                if bad.any():
                    worst = max(worst, float((1.0 - best_t[bad]).max()))
                    np.minimum.at(f, idx_all[bad], best_t[bad])
                    np.minimum.at(f, j[bad], best_t[bad])
            h = h * f
            if worst < 1e-4:
                break
        return h

    ca_t, sa_t = np.cos(theta), np.sin(theta)
    _uniform_request = r_max <= r_min * 1.01

    if _uniform_request:
        # Shrink-to-fit (pack-then-drop, below) fundamentally cannot give a
        # hard-locked uniform size: a locally-ROTATED-square tiling can't
        # stay perfectly uniform across an orientation transition without
        # either real overlap or a real gap (verified earlier this
        # session), so ANY tolerance below r_min reintroduces visible size
        # variance, and tightening that tolerance toward zero reintroduces
        # the OTHER reported failure -- entire regions dropping together
        # once every tile there fails a strict floor simultaneously.
        # Reported directly, twice, on real content, in both directions.
        #
        # The actual fix is to stop resizing tiles at all for this case.
        # Match circle mode's own Poisson-disk contract exactly: every
        # tile is either placed at EXACTLY r_min or not placed -- density
        # (how many fit) is what adapts, never size. Implemented as
        # sequential greedy acceptance (a spatial hash of already-accepted
        # tiles, each new candidate tested with an exact SAT check against
        # only its nearby accepted neighbours) rather than the simultaneous
        # shrink-everyone-at-once approach above, so a crowded candidate is
        # simply skipped rather than shrinking everyone nearby together.
        def _theta_fn(x, y):
            return float(_sample_grid(dc_field, 'theta_grid', np.array([x]), np.array([y]))[0]) + _TILE_ANGLE_OFFSET
        keep_idx, extra_tiles = _reject_pack_uniform(sx, sy, ca_t, sa_t, r_min, W, H, theta_fn=_theta_fn)
        half = np.zeros(n)
        half[keep_idx] = r_min
    else:
        extra_tiles = []
        # Shrinking an individual tile to fit a crowded neighbourhood (what
        # a single SAT-pack pass does) is exactly what real mosaicists do
        # when trimming a piece to fit -- the right behaviour whenever a
        # real [r_min, r_max] range means "vary size with detail". Pack
        # once, then drop (don't keep shrunken) any tile the pack
        # compressed severely, and re-pack the survivors -- with the
        # crowding neighbour gone, most recover close to their full target
        # size instead of staying permanently undersized.
        #
        # The floor is r_min itself, not a fraction of it: the configured
        # min/max is a guarantee about every tile in the output, so a tile
        # the packer had to compress below the stated minimum should be
        # dropped rather than shipped undersized. (An earlier 0.8x
        # tolerance was kept because a strict floor once collapsed
        # coverage -- but that predates the gap-filling pass below, which
        # now puts legally-sized tiles back into the space a drop frees.)
        alive = np.arange(n)
        half_alive = _sat_pack(alive)
        for _ in range(6):
            keep_mask = half_alive >= r_min * 0.999
            if keep_mask.all() or not keep_mask.any():
                break
            alive = alive[keep_mask]
            half_alive = _sat_pack(alive)

        half_alive = half_alive * alpha_pack
        keep_idx = alive
        half = np.zeros(n)
        half[keep_idx] = half_alive

    polys = []
    src_site = []
    for i in keep_idx.tolist():
        h = float(half[i])
        if h <= 1e-6:
            continue
        cx, cy = float(sx[i]), float(sy[i])
        ca, sa = float(ca_t[i]), float(sa_t[i])
        corners = [
            (cx + ca * h - sa * h, cy + sa * h + ca * h),
            (cx - ca * h - sa * h, cy - sa * h + ca * h),
            (cx - ca * h + sa * h, cy - sa * h - ca * h),
            (cx + ca * h + sa * h, cy + sa * h - ca * h),
        ]
        pts = [[float(np.clip(px, 0, W)), float(np.clip(py, 0, H))] for px, py in corners]
        polys.append(pts)
        src_site.append(i)

    # Extra tiles the dense-grid pass found room for, beyond what CVD's own
    # sites reached -- same fixed size, no site index to report back for
    # colour lookup, so the caller's nearest-source-point KDTree query
    # (keyed off world position, not this index) covers it instead.
    for (cx, cy, ang) in extra_tiles:
        ca, sa = math.cos(ang), math.sin(ang)
        h = r_min
        corners = [
            (cx + ca * h - sa * h, cy + sa * h + ca * h),
            (cx - ca * h - sa * h, cy - sa * h + ca * h),
            (cx - ca * h + sa * h, cy - sa * h - ca * h),
            (cx + ca * h + sa * h, cy + sa * h - ca * h),
        ]
        pts = [[float(np.clip(px, 0, W)), float(np.clip(py, 0, H))] for px, py in corners]
        polys.append(pts)
        src_site.append(-1)

    return polys, src_site


def _grow_into_gaps(polys, W, H, grout, n_passes=2, bs_iters=7,
                     max_grow_mult=0.35, max_edge_ratio=1.9):
    """Push each tile's corners outward into leftover void, stopping a fixed
    `grout` clear of every neighbour instead of at first contact.

    `tile_square.grow_corners` (written for the row-tracing path) grows until
    tiles actually touch, which on this already well-packed CVD output
    raised coverage a lot (64% -> 77%) but destroyed the even grout lines --
    tiles merged into blobs separated by ragged slivers instead of the
    uniform spacing the packing had produced. Keeping a minimum clearance
    turns the same idea into a grout-preserving fill: corners reach into the
    genuinely empty diagonal pockets (the "checkerboard" voids left where
    the local lattice runs corner-to-corner) while every tile stays visually
    separated. The convexity and edge-ratio guards keep results reading as
    squares rather than darts or wedges."""
    from shapely.geometry import Polygon, box as ShBox, Point as ShPoint
    from shapely.strtree import STRtree
    from .optimizer import _is_convex

    polys = list(polys)
    if not polys:
        return polys
    canvas = ShBox(0, 0, W, H)
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
                    if not _is_convex(trial):
                        return None
                    m = len(trial)
                    el = [math.hypot(trial[a][0] - trial[(a + 1) % m][0],
                                     trial[a][1] - trial[(a + 1) % m][1])
                          for a in range(m)]
                    if max(el) > min(el) * max_edge_ratio:
                        return None
                    cand = Polygon(trial)
                    if not cand.is_valid or not cand.is_simple:
                        return None
                    probe = cand.buffer(grout)
                    for j in tree.query(probe):
                        if j == i:
                            continue
                        if probe.intersects(polys[j]):
                            return None
                    return cand

                lo, hi = 0.0, max_step
                best, best_cand = 0.0, None
                full = try_offset(hi)
                if full is not None:
                    best, best_cand = hi, full
                else:
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


# ── Top-level orchestrator: matches `_flow_tiles`'s external contract ──────

def flow_square_tiles_cvd(xs, ys, rs, W, H, topo, r_min, r_max,
                           n_iters=20, rng=None, colors_bgr=None):
    """Full CVD pipeline: seed sites -> Lloyd-relax under a rotated,
    confidence-blended Manhattan metric (with edge avoidance) -> extract
    tiles from the converged Voronoi partition. Returns (polys, source_idx)
    with the same shape as `tile_square.flow_square_tiles`: polys as
    ([[x,y],...]) point lists (NOT shapely Polygons, to match the raster
    contour-extraction path -- callers already only consume `.exterior`-style
    point lists via the `_flow_tiles` adapter)."""
    rng = rng if rng is not None else np.random.default_rng(0)

    if colors_bgr is None or len(colors_bgr) != len(xs):
        # Direction field needs colour; degrade gracefully to a flat field
        # (every pixel "low confidence") if colours aren't available so the
        # metric falls back to plain (unrotated) Voronoi rather than crashing.
        colors_bgr = np.full((len(xs), 3), 128.0)

    # Two complementary direction sources: the edge-curve field (Hausner
    # sec. 4) catches crisp boundaries a detail-density surface has no
    # signal for at all (a sharp but visually flat-detail colour edge, e.g.
    # a shadow line); the detail topography catches gradual/textured
    # structure that never registers as a discrete edge (a hillside's
    # smooth incline). Blended in the QUADRUPLED-angle (mod-90) space these
    # orientations genuinely live in -- see the note in
    # `_build_direction_confidence_field`: square tiles are unchanged by a
    # quarter turn, so combining two sources with plain vector or
    # doubled-angle averaging treats near-identical square orientations
    # (say 2 and 88 degrees) as though they disagreed sharply and averages
    # them into something matching neither.
    _field_res = 512
    _tile_px_hint = max(2.0, (r_min + r_max) * (_field_res / float(max(W, H))))
    color_field = _build_direction_confidence_field(
        xs, ys, colors_bgr, W, H, analysis_res=_field_res, tile_px_hint=_tile_px_hint)
    topo_field = _build_detail_topography_field(
        xs, ys, rs, W, H, analysis_res=_field_res)
    vec = (color_field['conf_grid'] * np.exp(4j * color_field['theta_grid']) +
           topo_field['conf_grid'] * np.exp(4j * topo_field['theta_grid']))
    theta_grid = np.angle(vec) / 4.0
    conf_grid = np.clip(np.maximum(color_field['conf_grid'], topo_field['conf_grid']), 0.0, 1.0)
    dc_field = {
        'g': color_field['g'], 'sx': color_field['sx'], 'sy': color_field['sy'],
        'theta_grid': theta_grid, 'conf_grid': conf_grid,
        'grad_mag': np.maximum(color_field['grad_mag'], topo_field['grad_mag']),
    }

    local_scale_fn = topo['local_scale']

    def local_scale_vec(xa, ya):
        return _local_scale_arr(xa, ya, topo, r_min, r_max)

    seed_x, seed_y = _seed_sites(W, H, local_scale_vec, r_min, r_max, rng)
    if len(seed_x) < 3:
        return [], []

    fsx, fsy, labels, grid_shape, cell_wh = _lloyd_relax(
        seed_x, seed_y, W, H, dc_field, topo, r_min, r_max, n_iters=n_iters if n_iters and n_iters > 5 else 30)

    polys, src_site = _extract_tiles(fsx, fsy, labels, grid_shape, cell_wh,
                                      dc_field, topo, r_min, r_max, W, H)
    if not polys:
        return [], []

    # NOTE: a corner-growth-into-gaps pass (_grow_into_gaps, below) was
    # tried here to fill the remaining diagonal pockets. It held zero
    # overlap and raised coverage (64%->66% with a fixed grout clearance,
    # ->77% at full contact), but growing each corner independently breaks
    # squareness -- tiles read visibly jagged even under the convexity/
    # edge-ratio guards, undoing the squareness fix this same session just
    # landed. Left disabled until it's reworked as a uniform per-tile scale
    # (grow toward the gap without distorting the tile's own aspect ratio)
    # rather than four independently-moving corners.

    # Both remaining complaints (holes AND size variation) trace to the
    # same root: the drop-below-threshold step above only ever REMOVES
    # sites, it never adds anything back into the void that removal
    # creates. Tightening that threshold trades one problem for the other
    # (tried both directions this session) because it's fixing the wrong
    # stage. Genuinely filling the leftover void instead -- reusing the
    # row-tracing module's own validated distance-transform greedy filler,
    # which sizes each new tile to the LOCAL available width so a widening/
    # narrowing gap gets a row of tiles that tapers to match -- attacks
    # coverage directly without touching the packing/drop logic that
    # already balances overlap-safety against uniformity.
    from shapely.geometry import Polygon as _ShPolygon
    from .tile_square import fill_gaps as _fill_gaps

    def _fill_contour_angle(x, y):
        return float(_sample_grid(dc_field, 'theta_grid', np.array([x]), np.array([y]))[0]) + _TILE_ANGLE_OFFSET

    def _fill_local_scale(x, y, **_kw):
        return float(np.clip(_local_scale_arr(np.array([x]), np.array([y]), topo, r_min, r_max)[0], r_min, r_max))

    # `fill_gaps` deliberately TAPERS new tiles down toward a narrowing
    # gap's own local width (a real mosaicist trims smaller boundary
    # pieces rather than leaving a gap) -- correct and wanted when a real
    # [r_min, r_max] range means "vary size with detail", but it directly
    # fights an explicit min==max request: measured on real content, it
    # added under 1% more coverage while dragging the size floor down to
    # 36% of target, reintroducing the exact variance being fixed. Only
    # run it where size variation was already the ask.
    if r_max > r_min * 1.01:
        fill_topo = {'contour_angle': _fill_contour_angle, 'local_scale': _fill_local_scale}
        shp_polys = [_ShPolygon(p) for p in polys]
        shp_polys = [p if p.is_valid else p.buffer(0) for p in shp_polys]
        # min_side_frac=1.0: never place a gap-filler below the user's
        # configured minimum. The default (0.3) lets a filler taper to 30%
        # of the minimum side to chase a narrowing sliver, which is right
        # for row-tracing's boundary pieces but shows up here as tiles
        # visibly smaller than the stated size range (reported directly --
        # measured 2.4 against a minimum side of 8, i.e. exactly that 30%
        # floor). The min/max range is a guarantee about the output, so a
        # gap that can only take an undersized tile should stay empty.
        # Repeat: each pass places tiles at the WIDEST remaining spots and
        # clears only around those, so a single pass leaves smaller
        # still-fillable pockets behind. Stops as soon as a pass finds
        # nothing, so this costs nothing on an already-full layout.
        for _ in range(3):
            filled = _fill_gaps(shp_polys, fill_topo, W, H, r_min, r_max,
                                min_side_frac=1.0)
            if not filled:
                break
            shp_polys = shp_polys + filled
            polys = polys + [[[float(x), float(y)] for x, y in p.exterior.coords[:-1]] for p in filled]

    src_tree = KDTree(np.column_stack([xs, ys]))
    centroids = np.array([[np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])]
                           for poly in polys])
    _, src_idx = src_tree.query(centroids)
    return polys, src_idx.tolist()
