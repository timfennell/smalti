"""
Numba-accelerated, multi-core Poisson-disk gap-fill.

The pure-Python version in poisson_sampler.py (SpatialGrid + poisson_gap_fill) is
correct but is fundamentally a Python-dict-heavy sequential loop: profiling showed
has_conflict() alone (millions of dict lookups walking neighbouring grid cells)
accounts for ~80% of total gap-fill time on a real job. Two independent problems,
two independent fixes:

  1. Per-call Python/dict overhead → replaced with a flat-array "linked cell list"
     spatial grid (classic technique from particle simulations), compiled with
     Numba to native code. No algorithm change, just no interpreter/dict overhead.

  2. Placement is inherently sequential (each new circle's accept/reject decision
     depends on every circle already placed) → can't parallelize the algorithm
     itself. Instead we parallelize the *canvas*: split it into tiles, and let
     each tile independently fill only its "core" region — inset from every
     tile-to-tile boundary by a margin of at least (2*r_max + gap), so two tiles
     running concurrently can never place circles close enough to conflict with
     each other. That's real, safe multi-core work via numba.prange (which
     releases the GIL, unlike Python threads).

     The margin strips left unfilled between tiles (and any small tile-local
     under-fill from Poisson randomness) are then finished in a single sequential
     pass using the existing, proven poisson_sampler.poisson_gap_fill — now aware
     of every circle every tile placed, so there's no visible seam: the boundary
     region is filled with exactly the same conflict-checking as a normal single-
     threaded run, just over a much smaller remaining area.

Note: all "state-mutating" helpers below are top-level njit functions taking
explicit array/scalar arguments (rather than nested closures) — numba's support
for nested-function closures capturing `nonlocal` state is fragile/incomplete,
this flat style is the well-supported pattern.
"""

import math
import numpy as np
from numba import njit, prange

from .poisson_sampler import SpatialGrid, poisson_gap_fill as _poisson_gap_fill_sequential


# ── Numba-compiled flat-array spatial grid ("linked cell list") ────────────────
# head[key]      = index of the most-recently-inserted circle in that cell, or -1
# nxt[i]         = index of the PREVIOUS circle inserted into the same cell as
#                  circle i (singly-linked list via array), or -1
# cx/cy/cr[i]    = that circle's position/radius (flat, parallel arrays)
# Coordinates are in GLOBAL canvas space throughout; grid cell indices are offset
# by (origin_gx, origin_gy) so the head array only spans this tile's local extent
# (plus margin) rather than the whole canvas.

@njit(inline='always')
def _cell_key(x, y, cell_size, origin_gx, origin_gy, grid_cols):
    gx = int(math.floor(x / cell_size)) - origin_gx
    gy = int(math.floor(y / cell_size)) - origin_gy
    return gy * grid_cols + gx


@njit(inline='always')
def _grid_insert(head, nxt, cx_arr, cy_arr, cr_arr, count, cap,
                  x, y, r, cell_size, origin_gx, origin_gy, grid_cols):
    idx = count[0]
    if idx >= cap:
        return -1
    cx_arr[idx] = x
    cy_arr[idx] = y
    cr_arr[idx] = r
    key = _cell_key(x, y, cell_size, origin_gx, origin_gy, grid_cols)
    nxt[idx] = head[key]
    head[key] = idx
    count[0] = idx + 1
    return idx


@njit(inline='always')
def _grid_has_conflict(head, nxt, cx_arr, cy_arr, cr_arr, max_r,
                        x, y, r, gap, cell_size, origin_gx, origin_gy,
                        grid_cols, grid_rows):
    ccx = int(math.floor(x / cell_size)) - origin_gx
    ccy = int(math.floor(y / cell_size)) - origin_gy
    sr = int(math.ceil((r + max_r[0] + gap) / cell_size)) + 1
    for dy in range(-sr, sr + 1):
        gy = ccy + dy
        if gy < 0 or gy >= grid_rows:
            continue
        row_base = gy * grid_cols
        for dx in range(-sr, sr + 1):
            gx = ccx + dx
            if gx < 0 or gx >= grid_cols:
                continue
            idx = head[row_base + gx]
            while idx != -1:
                ex = cx_arr[idx]
                ey = cy_arr[idx]
                er = cr_arr[idx]
                dxp = x - ex
                dyp = y - ey
                rr = r + er + gap
                if dxp * dxp + dyp * dyp < rr * rr:
                    return True
                idx = nxt[idx]
    return False


@njit(inline='always')
def _mark_covered(covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols, x, y):
    cr_ = int(y) // cover_cell - seed_row0
    cc_ = int(x) // cover_cell - seed_col0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r2 = cr_ + dr
            c2 = cc_ + dc
            if 0 <= r2 < cover_rows and 0 <= c2 < cover_cols:
                covered[r2, c2] = True


@njit
def _run_from(
    sx, sy,
    head, nxt, cx_arr, cy_arr, cr_arr, count, cap, max_r,
    out_x, out_y, out_r, n_out,
    active_x, active_y, active_r,
    covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols,
    core_x0, core_y0, core_x1, core_y1,
    radius_map, gap, candidates_k,
    cell_size, origin_gx, origin_gy, grid_cols, grid_rows,
):
    """Grow a Bridson active-list placement outward from one seed point.
    Mutates the grid/output arrays in place; returns the updated n_out."""
    H_map, W_map = radius_map.shape
    iy = int(sy)
    ix = int(sx)
    if iy < 0:
        iy = 0
    if iy >= H_map:
        iy = H_map - 1
    if ix < 0:
        ix = 0
    if ix >= W_map:
        ix = W_map - 1
    sr = radius_map[iy, ix]

    if sx - sr < core_x0 or sx + sr > core_x1 or sy - sr < core_y0 or sy + sr > core_y1:
        _mark_covered(covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols, sx, sy)
        return n_out
    if _grid_has_conflict(head, nxt, cx_arr, cy_arr, cr_arr, max_r,
                           sx, sy, sr, gap, cell_size, origin_gx, origin_gy,
                           grid_cols, grid_rows):
        _mark_covered(covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols, sx, sy)
        return n_out

    _grid_insert(head, nxt, cx_arr, cy_arr, cr_arr, count, cap,
                 sx, sy, sr, cell_size, origin_gx, origin_gy, grid_cols)
    if sr > max_r[0]:
        max_r[0] = sr
    out_x[n_out] = sx
    out_y[n_out] = sy
    out_r[n_out] = sr
    n_out += 1
    _mark_covered(covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols, sx, sy)

    active_x[0] = sx
    active_y[0] = sy
    active_r[0] = sr
    n_active = 1

    while n_active > 0:
        i = np.random.randint(0, n_active)
        px = active_x[i]
        py = active_y[i]
        pr = active_r[i]
        found = False
        for _ in range(candidates_k):
            if n_out >= cap - 1:
                break
            angle = np.random.uniform(0.0, 2.0 * math.pi)
            min_d = pr + gap + 0.5
            min_d2 = pr * 0.1 + 0.5
            if min_d2 > min_d:
                min_d = min_d2
            dist = np.random.uniform(min_d, min_d * 2.2)
            cxp = px + dist * math.cos(angle)
            cyp = py + dist * math.sin(angle)
            iix = int(cxp)
            iiy = int(cyp)
            if iix < 0 or iix >= W_map or iiy < 0 or iiy >= H_map:
                continue
            crr = radius_map[iiy, iix]
            if cxp - crr < core_x0 or cxp + crr > core_x1:
                continue
            if cyp - crr < core_y0 or cyp + crr > core_y1:
                continue
            if _grid_has_conflict(head, nxt, cx_arr, cy_arr, cr_arr, max_r,
                                   cxp, cyp, crr, gap, cell_size, origin_gx, origin_gy,
                                   grid_cols, grid_rows):
                continue
            _grid_insert(head, nxt, cx_arr, cy_arr, cr_arr, count, cap,
                         cxp, cyp, crr, cell_size, origin_gx, origin_gy, grid_cols)
            if crr > max_r[0]:
                max_r[0] = crr
            out_x[n_out] = cxp
            out_y[n_out] = cyp
            out_r[n_out] = crr
            n_out += 1
            _mark_covered(covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols, cxp, cyp)
            active_x[n_active] = cxp
            active_y[n_active] = cyp
            active_r[n_active] = crr
            n_active += 1
            found = True
            break
        if not found:
            # swap-remove (order doesn't matter — active point selection is random)
            n_active -= 1
            active_x[i] = active_x[n_active]
            active_y[i] = active_y[n_active]
            active_r[i] = active_r[n_active]

    return n_out


@njit
def _fill_tile(
    radius_map, gap, candidates_k, seed,
    core_x0, core_y0, core_x1, core_y1,       # tile's own placement region (global px)
    existing_x, existing_y, existing_r,        # circles pre-seeded into this tile's grid
    cell_size, origin_gx, origin_gy, grid_cols, grid_rows, cap,
    cover_cell, cover_rows, cover_cols, seed_row0, seed_col0,
):
    """Fill one tile's core region. Runs independently per-tile (no cross-tile
    reads/writes), so it's safe to call from a numba.prange loop."""
    np.random.seed(seed)

    head = np.full(grid_cols * grid_rows, -1, dtype=np.int64)
    nxt = np.full(cap, -1, dtype=np.int64)
    cx_arr = np.empty(cap, dtype=np.float64)
    cy_arr = np.empty(cap, dtype=np.float64)
    cr_arr = np.empty(cap, dtype=np.float64)
    count = np.zeros(1, dtype=np.int64)
    max_r = np.zeros(1, dtype=np.float64)

    for i in range(len(existing_x)):
        ex, ey, er = existing_x[i], existing_y[i], existing_r[i]
        if er > max_r[0]:
            max_r[0] = er
        _grid_insert(head, nxt, cx_arr, cy_arr, cr_arr, count, cap,
                     ex, ey, er, cell_size, origin_gx, origin_gy, grid_cols)

    out_x = np.empty(cap, dtype=np.float64)
    out_y = np.empty(cap, dtype=np.float64)
    out_r = np.empty(cap, dtype=np.float64)
    n_out = 0

    covered = np.zeros((cover_rows, cover_cols), dtype=np.bool_)

    active_x = np.empty(cap, dtype=np.float64)
    active_y = np.empty(cap, dtype=np.float64)
    active_r = np.empty(cap, dtype=np.float64)

    n_out = _run_from(
        (core_x0 + core_x1) * 0.5, (core_y0 + core_y1) * 0.5,
        head, nxt, cx_arr, cy_arr, cr_arr, count, cap, max_r,
        out_x, out_y, out_r, n_out,
        active_x, active_y, active_r,
        covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols,
        core_x0, core_y0, core_x1, core_y1,
        radius_map, gap, candidates_k,
        cell_size, origin_gx, origin_gy, grid_cols, grid_rows,
    )

    all_cells = np.empty(cover_rows * cover_cols, dtype=np.int64)
    for i in range(cover_rows * cover_cols):
        all_cells[i] = i
    np.random.shuffle(all_cells)

    for ci in range(all_cells.shape[0]):
        if n_out >= cap - 1:
            break
        flat = all_cells[ci]
        row = flat // cover_cols
        col = flat % cover_cols
        if covered[row, col]:
            continue
        gx = (seed_col0 + col + np.random.uniform(0.1, 0.9)) * cover_cell
        gy = (seed_row0 + row + np.random.uniform(0.1, 0.9)) * cover_cell
        if gx < core_x0:
            gx = core_x0
        if gx > core_x1:
            gx = core_x1
        if gy < core_y0:
            gy = core_y0
        if gy > core_y1:
            gy = core_y1
        n_out = _run_from(
            gx, gy,
            head, nxt, cx_arr, cy_arr, cr_arr, count, cap, max_r,
            out_x, out_y, out_r, n_out,
            active_x, active_y, active_r,
            covered, seed_row0, seed_col0, cover_cell, cover_rows, cover_cols,
            core_x0, core_y0, core_x1, core_y1,
            radius_map, gap, candidates_k,
            cell_size, origin_gx, origin_gy, grid_cols, grid_rows,
        )

    return out_x[:n_out].copy(), out_y[:n_out].copy(), out_r[:n_out].copy()


@njit(parallel=True)
def _fill_all_tiles(
    radius_map, gap, candidates_k, base_seed,
    tile_core_x0, tile_core_y0, tile_core_x1, tile_core_y1,
    tile_existing_start, tile_existing_count,
    existing_x_all, existing_y_all, existing_r_all,
    cell_size, tile_origin_gx, tile_origin_gy, tile_grid_cols, tile_grid_rows, tile_cap,
    tile_cover_cell, tile_cover_rows, tile_cover_cols, tile_seed_row, tile_seed_col,
    out_x_all, out_y_all, out_r_all, out_count,
):
    n_tiles = tile_core_x0.shape[0]
    for t in prange(n_tiles):
        start = tile_existing_start[t]
        cnt = tile_existing_count[t]
        ex = existing_x_all[start:start + cnt]
        ey = existing_y_all[start:start + cnt]
        er = existing_r_all[start:start + cnt]
        ox, oy, orad = _fill_tile(
            radius_map, gap, candidates_k, base_seed + t,
            tile_core_x0[t], tile_core_y0[t], tile_core_x1[t], tile_core_y1[t],
            ex, ey, er,
            cell_size, tile_origin_gx[t], tile_origin_gy[t],
            tile_grid_cols[t], tile_grid_rows[t], tile_cap[t],
            tile_cover_cell[t], tile_cover_rows[t], tile_cover_cols[t],
            tile_seed_row[t], tile_seed_col[t],
        )
        n = ox.shape[0]
        cap_t = out_x_all.shape[1]
        if n > cap_t:
            n = cap_t
        out_x_all[t, :n] = ox[:n]
        out_y_all[t, :n] = oy[:n]
        out_r_all[t, :n] = orad[:n]
        out_count[t] = n


def _choose_tile_grid(n_tiles_target: int, W: int, H: int):
    """Pick a rows x cols split roughly matching the canvas aspect ratio."""
    cols = max(1, int(round(math.sqrt(n_tiles_target * W / max(H, 1)))))
    rows = max(1, int(round(n_tiles_target / cols)))
    return rows, cols


def poisson_gap_fill_parallel(
    existing: list,
    radius_map: np.ndarray,
    gap: float,
    candidates_k: int = 30,
    progress_callback=None,
    preview_callback=None,
    preview_every: int = 1000,
    n_threads: int | None = None,
) -> list:
    """Drop-in replacement for poisson_sampler.poisson_gap_fill: same signature,
    same return contract (new circles only — caller appends to `existing`), but
    does the bulk of the placement work in parallel across CPU cores.

    Falls back to the plain sequential version outright for small jobs, where
    tiling overhead isn't worth it and the whole thing finishes in well under a
    second anyway."""
    H, W = radius_map.shape
    r_min = float(radius_map.min())
    r_max = float(radius_map.max())
    placement_gap = max(gap, 0.0)

    # Tiny job — not worth tiling/JIT-warmup overhead. Threshold picked from
    # profiling: the pure-Python version does ~1,300-1,400 circles/sec, so a
    # canvas expected to need only a few thousand circles finishes in ~1-2s
    # either way.
    if W * H < 400_000:
        return _poisson_gap_fill_sequential(
            existing, radius_map, gap, candidates_k,
            progress_callback, preview_callback, preview_every,
        )

    import os
    n_tiles_target = n_threads or os.cpu_count() or 4
    n_tiles_target = max(1, min(n_tiles_target, 32))
    rows, cols = _choose_tile_grid(n_tiles_target, W, H)

    # Margin: two tiles running concurrently must never be able to place circles
    # close enough to conflict — worst case is two max-radius circles right at
    # the shared boundary, so each side needs at least (r_max + gap) of buffer,
    # doubled for safety against the coarse cover-grid's own rounding.
    margin = (r_max + placement_gap) * 2.2 + 2.0
    cell_size = max(r_min / 1.42, 0.5)

    tile_w = W / cols
    tile_h = H / rows

    existing_arr = np.asarray(existing, dtype=np.float64).reshape(-1, 3) if existing else \
        np.zeros((0, 3), dtype=np.float64)

    tiles = []
    for ty in range(rows):
        for tx in range(cols):
            core_x0 = tx * tile_w
            core_x1 = (tx + 1) * tile_w if tx < cols - 1 else float(W)
            core_y0 = ty * tile_h
            core_y1 = (ty + 1) * tile_h if ty < rows - 1 else float(H)
            # Shrink the core away from every INTERNAL tile-to-tile boundary by
            # `margin`, and away from the true CANVAS edge by `placement_gap` —
            # _run_from's own boundary check is a bare `core_x0`/`core_x1`
            # comparison (matching poisson_sampler's `placement_gap` edge rule
            # exactly), so an edge tile must start with that inset applied
            # already rather than at the literal 0/W canvas boundary.
            if tx > 0:
                core_x0 += margin
            else:
                core_x0 += placement_gap
            if tx < cols - 1:
                core_x1 -= margin
            else:
                core_x1 -= (1 + placement_gap)
            if ty > 0:
                core_y0 += margin
            else:
                core_y0 += placement_gap
            if ty < rows - 1:
                core_y1 -= margin
            else:
                core_y1 -= (1 + placement_gap)
            if core_x1 <= core_x0 or core_y1 <= core_y0:
                continue   # tile too small after margin shrink — skip, cleanup pass covers it
            tiles.append((core_x0, core_y0, core_x1, core_y1))

    if not tiles:
        # Degenerate (tiny canvas relative to margin) — just run sequential.
        return _poisson_gap_fill_sequential(
            existing, radius_map, gap, candidates_k,
            progress_callback, preview_callback, preview_every,
        )

    n_tiles = len(tiles)
    core_x0_arr = np.array([t[0] for t in tiles], dtype=np.float64)
    core_y0_arr = np.array([t[1] for t in tiles], dtype=np.float64)
    core_x1_arr = np.array([t[2] for t in tiles], dtype=np.float64)
    core_y1_arr = np.array([t[3] for t in tiles], dtype=np.float64)

    origin_gx = np.empty(n_tiles, dtype=np.int64)
    origin_gy = np.empty(n_tiles, dtype=np.int64)
    grid_cols = np.empty(n_tiles, dtype=np.int64)
    grid_rows = np.empty(n_tiles, dtype=np.int64)
    cap_arr = np.empty(n_tiles, dtype=np.int64)
    cover_cell_arr = np.empty(n_tiles, dtype=np.int64)
    cover_rows_arr = np.empty(n_tiles, dtype=np.int64)
    cover_cols_arr = np.empty(n_tiles, dtype=np.int64)
    seed_row_arr = np.empty(n_tiles, dtype=np.int64)
    seed_col_arr = np.empty(n_tiles, dtype=np.int64)

    existing_start = np.empty(n_tiles, dtype=np.int64)
    existing_count = np.empty(n_tiles, dtype=np.int64)
    ex_lists, ey_lists, er_lists = [], [], []
    cursor = 0

    cover_cell = max(1, int(r_min * 1.8))

    for t, (cx0, cy0, cx1, cy1) in enumerate(tiles):
        gx0 = int(math.floor((cx0 - margin) / cell_size)) - 1
        gy0 = int(math.floor((cy0 - margin) / cell_size)) - 1
        gx1 = int(math.ceil((cx1 + margin) / cell_size)) + 1
        gy1 = int(math.ceil((cy1 + margin) / cell_size)) + 1
        origin_gx[t] = gx0
        origin_gy[t] = gy0
        grid_cols[t] = max(1, gx1 - gx0)
        grid_rows[t] = max(1, gy1 - gy0)

        area = (cx1 - cx0) * (cy1 - cy0)
        # Generous safe upper bound on circles this tile could ever place —
        # real Poisson-disk packing density is far below this; it only needs
        # to never be reached, not be tight.
        cap_arr[t] = max(256, int(area / max(r_min * r_min, 0.25)) + 64)

        seed_col0 = int((cx0 - margin) // cover_cell) - 1
        seed_row0 = int((cy0 - margin) // cover_cell) - 1
        seed_col1 = int((cx1 + margin) // cover_cell) + 1
        seed_row1 = int((cy1 + margin) // cover_cell) + 1
        seed_row_arr[t] = seed_row0
        seed_col_arr[t] = seed_col0
        cover_rows_arr[t] = max(1, seed_row1 - seed_row0)
        cover_cols_arr[t] = max(1, seed_col1 - seed_col0)
        cover_cell_arr[t] = cover_cell

        if len(existing_arr):
            ex_mask = (
                (existing_arr[:, 0] >= cx0 - margin) & (existing_arr[:, 0] <= cx1 + margin) &
                (existing_arr[:, 1] >= cy0 - margin) & (existing_arr[:, 1] <= cy1 + margin)
            )
            sub = existing_arr[ex_mask]
        else:
            sub = existing_arr
        ex_lists.append(sub[:, 0]); ey_lists.append(sub[:, 1]); er_lists.append(sub[:, 2])
        existing_start[t] = cursor
        existing_count[t] = len(sub)
        cursor += len(sub)

    existing_x_all = np.concatenate(ex_lists) if ex_lists else np.zeros(0)
    existing_y_all = np.concatenate(ey_lists) if ey_lists else np.zeros(0)
    existing_r_all = np.concatenate(er_lists) if er_lists else np.zeros(0)

    out_cap = int(cap_arr.max())
    out_x_all = np.zeros((n_tiles, out_cap), dtype=np.float64)
    out_y_all = np.zeros((n_tiles, out_cap), dtype=np.float64)
    out_r_all = np.zeros((n_tiles, out_cap), dtype=np.float64)
    out_count = np.zeros(n_tiles, dtype=np.int64)

    rng = np.random.default_rng(1)
    base_seed = int(rng.integers(0, 2**31 - 1))

    radius_map_f64 = radius_map.astype(np.float64)

    if progress_callback:
        progress_callback(0)

    _fill_all_tiles(
        radius_map_f64, float(placement_gap), int(candidates_k), base_seed,
        core_x0_arr, core_y0_arr, core_x1_arr, core_y1_arr,
        existing_start, existing_count,
        existing_x_all, existing_y_all, existing_r_all,
        cell_size, origin_gx, origin_gy, grid_cols, grid_rows, cap_arr,
        cover_cell_arr, cover_rows_arr, cover_cols_arr, seed_row_arr, seed_col_arr,
        out_x_all, out_y_all, out_r_all, out_count,
    )

    parallel_placed = []
    for t in range(n_tiles):
        n = int(out_count[t])
        for i in range(n):
            parallel_placed.append((float(out_x_all[t, i]), float(out_y_all[t, i]), float(out_r_all[t, i])))

    if progress_callback:
        progress_callback(len(parallel_placed))
    if preview_callback:
        preview_callback(list(existing) + parallel_placed)

    # ── Sequential cleanup pass ──────────────────────────────────────────────
    # Everything the parallel phase placed is now just more "existing" circles
    # as far as this pass is concerned — it fills whatever's left (the margin
    # strips between tiles, plus any small tile-local under-fill) with the
    # exact same conflict-checking as a normal single-threaded run. No seams:
    # the boundary regions are placed with full knowledge of every circle every
    # tile produced, not independently guessed at.
    combined_existing = list(existing) + parallel_placed

    def _cleanup_progress(n):
        if progress_callback:
            progress_callback(len(parallel_placed) + n)

    cleanup_placed = _poisson_gap_fill_sequential(
        combined_existing, radius_map, gap, candidates_k,
        _cleanup_progress if progress_callback else None,
        preview_callback, preview_every,
    )

    return parallel_placed + cleanup_placed
