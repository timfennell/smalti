"""
Spatial error-diffusion color assignment for irregular circle grids.

Principle (optical color mixing / pointillism):
  When a circle's target color can't be exactly matched by any palette color,
  the residual error is distributed to spatial neighbors. Those neighbors
  compensate by skewing their own assignment. The eye averages adjacent circles
  at viewing distance and perceives intermediate tones not in the palette.

Performance: fully vectorized using k-NN precomputation.
  - One KDTree.query(k=K) call for ALL neighbors at once — O(N log N)
  - No per-circle ball queries — was O(N²) in the old version
  - The forward-propagation loop is still O(N) in Python but has no inner
    KDTree calls, so 13K circles takes milliseconds not minutes.
"""

import time
import numpy as np
from scipy.spatial import KDTree

# How often the per-dot loops below explicitly yield the GIL.  These are pure-Python
# loops running in a background thread — CPython does release the GIL periodically on
# its own switch-interval, but on a very large mosaic this loop can otherwise dominate
# a core for many seconds at a stretch, and a scipy/numpy C call inside it may hold the
# GIL for its own duration too, making the UI (server + browser) feel "soft-stopped".
# A cheap explicit yield every _YIELD_EVERY dots keeps the server responsive.
_YIELD_EVERY = 1000


def error_diffusion_assign(
    xs: np.ndarray,
    ys: np.ndarray,
    rs: np.ndarray,
    source_lab: np.ndarray,         # (H, W, 3)
    palette_lab: np.ndarray,        # (N_colors, 3)
    dither_strength: float = 0.7,
    blend_radius_multiplier: float = 3.5,
    k_neighbors: int = 16,          # fixed k-NN (enough for a wider blend radius to matter)
    kernel: str = 'sierra',         # 'sierra' (Sierra-3) or 'ball' (legacy 1/d)
    stop_event=None,                # checked every _YIELD_EVERY dots so Cancel is near-instant
) -> np.ndarray:
    """Returns int32 (N,) palette index per circle via spatial error diffusion.

    'sierra' uses the classic Sierra-3 error-diffusion kernel — its specific forward-
    weighted distribution gives cleaner, blue-noise-like dithering (fewer clumps/worms)
    than an evenly-spread ball.  The kernel is mapped onto the irregular dot grid by
    finding the dot nearest each kernel cell."""
    N = len(xs)
    H, W = source_lab.shape[:2]
    if N == 0:
        return np.zeros(0, dtype=np.int32)

    # Strength >1 amplifies the diffused error unboundedly (runaway smearing) — clamp it.
    dither_strength = float(np.clip(dither_strength, 0.0, 1.0))
    blend_radius_multiplier = float(np.clip(blend_radius_multiplier, 0.0, 16.0))
    pts = np.column_stack([xs, ys])
    tree = KDTree(pts)
    pal = palette_lab.astype(np.float32)

    # Circles are processed in a fixed scan order (see `order` below), and residual
    # error only ever flows to NOT-YET-processed neighbors. Circles near the tail of
    # that order (in practice: the bottom edge, since the 'ball' order sorts by y
    # ascending) can receive error from many earlier-processed neighbors but have few
    # or no later neighbors to pass their own error on to — so with strong dither
    # settings (high strength + wide blend radius), error keeps accumulating into
    # those cells with nowhere to drain, sometimes reaching hundreds of ΔE — enough to
    # drag an otherwise-correct target (e.g. pure black) across a palette decision
    # boundary into a visibly wrong colour. A stray non-background colour hugging one
    # edge of an otherwise-uniform region is the signature of this. Capping the
    # magnitude of error actually APPLIED (not the accumulator itself, so future
    # propagation math is unaffected) fixes this without changing normal-case
    # dithering, where single-hop residuals are far below this bound anyway.
    #
    # 10.0 was tuned only against that failure mode, without accounting for
    # palettes with wide gaps between adjacent hues (30-80+ ΔE) — a transition
    # between two such colours needs far more than 10 ΔE of nudge to build a
    # real dithered blend zone, so it was snapping straight to a hard edge
    # instead. 30.0 buys a wider, smoother transition on those palettes while
    # keeping the dead-end runaway case (which was reaching into the
    # hundreds) capped well below where it visibly drags a stray colour into
    # an unrelated region.
    _MAX_APPLIED_ERR_MAG = 30.0

    def _target(i, err_i):
        ix = int(np.clip(xs[i], 0, W - 1)); iy = int(np.clip(ys[i], 0, H - 1))
        err_mag = float(np.sqrt(err_i @ err_i))
        if err_mag > _MAX_APPLIED_ERR_MAG:
            err_i = err_i * (_MAX_APPLIED_ERR_MAG / err_mag)
        t = source_lab[iy, ix].astype(np.float32) + err_i * dither_strength
        t[0] = np.clip(t[0], 0.0, 100.0)
        t[1] = np.clip(t[1], -128.0, 127.0)
        t[2] = np.clip(t[2], -128.0, 127.0)
        return t

    error     = np.zeros((N, 3), dtype=np.float32)
    indices   = np.zeros(N, dtype=np.int32)
    processed = np.zeros(N, dtype=bool)

    if kernel == 'sierra':
        # Sierra-3 kernel: (dx, dy, weight)/32, forward of the current pixel.
        SIERRA = [(1, 0, 5), (2, 0, 3),
                  (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                  (-1, 2, 2), (0, 2, 3), (1, 2, 2)]
        kw = np.array([w for _, _, w in SIERRA], np.float32) / 32.0
        Kn = len(SIERRA)
        sd, _ = tree.query(pts, k=2, workers=1)
        spacing = max(float(np.median(sd[:, 1])), 1e-3)
        # Blend radius SCALES the kernel footprint: bigger radius spreads the Sierra
        # pattern over more dots (more diffusion), smaller keeps it tight.  3.5 = 1× (the
        # classic one-pixel-step kernel).
        kstep = spacing * max(0.4, blend_radius_multiplier / 3.5)
        # Serpentine rows (alternate x-direction) to avoid directional bias.
        row = np.floor(ys / spacing).astype(np.int64)
        sgn = np.where(row % 2 == 0, 1.0, -1.0).astype(np.float32)
        xkey = np.where(row % 2 == 0, xs, -xs)
        order = np.lexsort((xkey, row))
        # Batched: for every dot, map each kernel cell to its nearest dot (one query).
        tgt = np.empty((N, Kn, 2), np.float32)
        for kk, (dx, dy, _w) in enumerate(SIERRA):
            tgt[:, kk, 0] = xs + sgn * (dx * kstep)
            tgt[:, kk, 1] = ys + (dy * kstep)
        kd, kj = tree.query(tgt.reshape(-1, 2), workers=1)
        kd = kd.reshape(N, Kn); kj = kj.reshape(N, Kn).astype(np.int32)
        kvalid = (kd < max(kstep, spacing) * 0.9) & (kj != np.arange(N)[:, None])

        for step_i, i in enumerate(order):
            if step_i % _YIELD_EVERY == 0:
                time.sleep(0)
                if stop_event and stop_event.is_set():
                    raise InterruptedError("cancelled")
            target = _target(i, error[i])
            diffs = pal - target
            best = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
            indices[i] = best; processed[i] = True
            if dither_strength < 0.01:
                continue
            residual = target - pal[best]
            if float(residual @ residual) < 0.5:      # < ~0.7 ΔE — nothing to spread
                continue
            nbrs = kj[i]
            m = kvalid[i] & (~processed[nbrs])
            if m.any():
                error[nbrs[m]] += residual[None, :] * kw[m][:, None]
        return indices

    # ── Legacy 'ball' method (evenly-weighted 1/d over a radius) ──────────────
    k = min(k_neighbors + 1, N)
    neighbor_dists, neighbor_ids = tree.query(pts, k=k, workers=1)
    neighbor_dists = neighbor_dists[:, 1:].astype(np.float32)
    neighbor_ids   = neighbor_ids[:, 1:].astype(np.int32)
    r_search = (rs * blend_radius_multiplier).astype(np.float32)
    order = np.lexsort([xs, ys])
    for step_i, i in enumerate(order):
        if step_i % _YIELD_EVERY == 0:
            time.sleep(0)
            if stop_event and stop_event.is_set():
                raise InterruptedError("cancelled")
        target = _target(i, error[i])
        diffs = pal - target
        best = int(np.argmin(np.einsum('ij,ij->i', diffs, diffs)))
        indices[i] = best; processed[i] = True
        residual = target - pal[best]
        if float(residual @ residual) < 4.0 or dither_strength < 0.01:
            continue
        nbr_i = neighbor_ids[i]; nbr_d = neighbor_dists[i]
        valid_mask = (nbr_d <= r_search[i]) & (~processed[nbr_i])
        if not valid_mask.any():
            continue
        valid_ids = nbr_i[valid_mask]
        w = 1.0 / (nbr_d[valid_mask] + 1e-6); w /= w.sum()
        error[valid_ids] += residual[None, :] * w[:, None]
    return indices
