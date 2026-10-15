"""
G-code pen-plotter export.

Turns a mosaic into pen-plotter G-code, one file per colour (so you swap pens between
files).  Two fill strategies, both a single continuous pen-down stroke per shape so the
plotter never lifts mid-shape:

  • circle  → an Archimedean spiral from the outer edge inward to the centre.
  • polygon → contour-parallel "onion" spiral: the cell outline is offset inward by the
    pen width repeatedly and the rings are joined nearest-point into one stroke, from the
    outer edge inward to the middle.

The pen tip width sets the spacing between spiral passes so the fill has full coverage.
Coordinates are emitted in millimetres (G21), Y flipped so the plot isn't mirrored.

Shape DRAW ORDER uses the same multi-pass dispersion as the CNC paint-robot JSON export
(see cnc_export.py): shapes are split into a few passes so no two shapes in the same pass
are near each other, then each pass is travelled in a short serpentine order.  Without
this, drawing shapes in raw generation order produces a path that zig-zags randomly across
the whole canvas — slow to plot and, for wet media, revisits still-wet neighbours.
"""

import math
from pathlib import Path

import numpy as np

from .units import UNITS_TO_INCHES
from .optimizer import _voronoi_polygons
from .raster_export import _rounded_cell_points, _tile_inset_points, _hex_rgb   # reuse cell shaping
from .cnc_export import _serpentine                          # shared pass-ordering logic


def _to_mm(value: float, unit: str) -> float:
    return value * UNITS_TO_INCHES.get(unit, UNITS_TO_INCHES['mm']) * 25.4


_MAX_CHORD_MM = 0.25   # straight-segment length small enough to read as a smooth curve


def _circle_spiral(cx, cy, R, step):
    """Closed outer perimeter, THEN an Archimedean spiral from R inward to the centre
    (radial pitch = step).  Tracing the perimeter first means the true outline of the
    dot is always fully inked as one continuous closed pass, not just touched at a
    single point before the spiral immediately starts shrinking inward."""
    if R <= 1e-4:
        return [(cx, cy)]
    b = step / (2.0 * math.pi)
    pts = []

    def dtheta_at(r):
        # Chord length is bounded by an absolute mm value at every radius (not just a
        # minimum angular step) — the old floor let chords grow unboundedly on large
        # circles, faceting them.  Capped above so tiny circles don't take huge leaps.
        return min(0.5, _MAX_CHORD_MM / max(r, _MAX_CHORD_MM))

    # 1) Closed outer perimeter at the full radius.
    theta = 0.0
    while theta < 2.0 * math.pi:
        pts.append((cx + R * math.cos(theta), cy + R * math.sin(theta)))
        theta += dtheta_at(R)
    pts.append((cx + R, cy))   # close exactly back to the start point

    # 2) Continue into the inward Archimedean spiral from the same angle.
    theta = 0.0
    r = R
    while r > 1e-3:
        theta += dtheta_at(r)
        r = R - b * theta
        if r <= 1e-3:
            break
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    pts.append((cx, cy))
    return pts


def _resample_closed(coords, max_chord):
    """Densify a closed boundary so no chord is longer than max_chord."""
    pts = list(coords)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    out = []
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        out.append((ax, ay))
        d = math.hypot(bx - ax, by - ay)
        k = int(d / max_chord)
        for j in range(1, k):
            t = j / k
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


def _polygon_spiral(coords, site, step):
    """Closed outer perimeter, THEN a single-stroke inward spiral filling the (convex)
    polygon, converging to `site` — the software-generated centre point (the Voronoi
    cell's generating dot), which is NOT generally the geometric centroid.  Tracing the
    outline first means the cell's true edge is always fully inked as one continuous
    closed pass before the fill starts shrinking inward, instead of the outline only
    ever being touched at a single point.  The spiral itself walks the boundary while
    scaling it toward the site: each loop shrinks by one pen-width, so the pitch along
    the longest ray is ≤ pen width → full coverage, and it always ends at the site."""
    px, py = site
    ring = _resample_closed(coords, step)
    if len(ring) < 3:
        return []
    maxD = max(math.hypot(x - px, y - py) for x, y in ring)
    if maxD < 1e-4:
        return [(px, py)]
    pitch = step * 0.9                       # slight overlap for solid coverage
    N = max(1, int(math.ceil(maxD / pitch)))
    total = len(ring)
    # 1) Closed outer perimeter (c=1.0 all the way around, back to the start point).
    path = list(ring) + [ring[0]]
    # 2) Continue into the inward spiral from the same point.
    for k in range(N * total + 1):
        c = 1.0 - (k / total) / N            # scale: 1 (outer edge) → 0 (site)
        if c <= 0:
            break
        bx, by = ring[k % total]
        path.append((px + c * (bx - px), py + c * (by - py)))
    path.append((px, py))
    return path


def _order_multipass(pts_xyzs, passes):
    """Order a layer's shapes (x_mm, y_mm, size_mm, payload) for efficient, dry-safe
    plotting.  `size_mm` is the shape's own on-paper footprint radius (circle radius,
    or a polygon cell's centre-to-farthest-vertex distance) — adjacency MUST be judged
    per-pair from each shape's own size, not from a single global distance.  A mosaic
    routinely mixes huge and tiny dots/cells in the same layer (that's the whole point
    of densify/gradient dithering); a fixed global spacing threshold sized for the
    average shape lets a big shape sit "far enough" center-to-center from a small
    neighbour while their edges are still touching, or in the other direction wastes
    passes forcing tiny shapes far apart from each other unnecessarily.

    Two properties, and they must NOT be traded against each other:
      • non-adjacent passes — no two shapes whose footprints would touch (or nearly
        touch) ever land in the SAME pass.  Decided GLOBALLY across the whole layer —
        deciding it cluster-by-cluster silently breaks it, since a shape near a cluster
        boundary has no idea what pass its neighbour in the *next* cluster landed in.
      • locality — once pass membership is fixed, the pen should still travel through
        each pass's shapes in a spatially coherent order (finish one region, then the
        next) rather than a raw generation-order zig-zag across the whole canvas.

    So: assign passes globally first (correctness), then reorder the shapes *within*
    each pass using local clustering (efficiency) — pass membership itself is never
    touched by the clustering step."""
    n = len(pts_xyzs)
    if n < 3:
        return list(pts_xyzs)
    xy_all = np.array([(x, y) for x, y, _s, _p in pts_xyzs], dtype=float)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy_all).query(xy_all, k=2)
    nn = float(np.median(d[:, 1])) if n >= 2 else 1.0
    nn = max(nn, 1e-6)
    band = max(2.0 * nn, 1e-6)
    # Two shapes conflict once the gap between their edges drops below this margin —
    # sep_factor > 1 leaves a visible gap (not just barely-not-touching) for a wet
    # medium; a few more passes than "requested" fall out naturally in the greedy
    # assignment below wherever the local packing genuinely needs it.
    sep_factor = 1.15 if passes <= 2 else 1.3
    margin = 0.15 * nn

    if passes <= 1:
        return _local_serpentine(list(pts_xyzs), nn, band)

    # Pass assignment: global greedy coloring keyed on each pair's OWN sizes, so a
    # same-pass pair is never closer than the sum of their footprints (scaled) — a
    # hard geometric guarantee, not a best-effort.  `passes` only informs sep_factor
    # above; dense/uneven packing can force more passes than requested, and we never
    # silently violate the spacing just to hit a fixed count.
    buckets = _assign_passes_greedy(list(pts_xyzs), sep_factor, margin)

    out = []
    for p in buckets:
        out.extend(_local_serpentine(p, nn, band))
    return out


def _assign_passes_greedy(pts_xyzs, sep_factor, margin):
    """Greedily colour shapes into passes so that within any single pass, every pair
    of shapes has centre-distance >= (size_i + size_j) * sep_factor + margin —
    guaranteed by construction (each shape is placed in the first pass bucket where it
    doesn't conflict with anything already there, checked via a per-bucket spatial
    grid sized to the largest shape present), unlike a fixed-bucket-count scheme that
    has to dump unplaceable overflow somewhere and risk violating the spacing."""
    from collections import defaultdict
    sizes = [s for _x, _y, s, _p in pts_xyzs]
    max_size = max(sizes) if sizes else 0.0
    # Cell sized so any conflicting pair (however large) is always within the 3x3
    # neighbourhood scanned by `fits` — correctness first, even if coarse for a layer
    # with one huge outlier shape.
    cell = max(2.0 * max_size * sep_factor + margin, 1e-6)
    grids = [defaultdict(list)]   # one grid per pass bucket, grown on demand
    buckets = [[]]

    def fits(grid, cx, cy, x, y, size):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py, psize) in grid.get((cx + dx, cy + dy), ()):
                    required = (size + psize) * sep_factor + margin
                    if (px - x) ** 2 + (py - y) ** 2 < required * required:
                        return False
        return True

    # Serpentine visiting order for placement itself keeps spatially-close shapes from
    # being considered in a wildly different order, which in turn keeps the number of
    # buckets actually used close to the geometric minimum.
    for (x, y, size, payload) in _serpentine(list(pts_xyzs), cell):
        cx, cy = int(x // cell), int(y // cell)
        placed = False
        for b in range(len(buckets)):
            if fits(grids[b], cx, cy, x, y, size):
                grids[b][(cx, cy)].append((x, y, size))
                buckets[b].append((x, y, size, payload))
                placed = True
                break
        if not placed:
            grids.append(defaultdict(list))
            buckets.append([])
            grids[-1][(cx, cy)].append((x, y, size))
            buckets[-1].append((x, y, size, payload))
    return buckets


def _local_serpentine(pts_xyzs, nn, band):
    """Traversal order for ONE pass's shapes: tile into local clusters of a modest
    size and visit the clusters in a serpentine, serpentining within each cluster too
    — so the pen works through one region before moving to the next, instead of a
    single canvas-wide serpentine whose rows can span the whole width."""
    n = len(pts_xyzs)
    if n < 3:
        return _serpentine(list(pts_xyzs), band)
    xy = np.array([(x, y) for x, y, _s, _p in pts_xyzs], dtype=float)
    TARGET_PER_CLUSTER = 40
    cluster_span = nn * math.sqrt(max(TARGET_PER_CLUSTER, 1) / math.pi) * 2.0
    cluster_span = max(cluster_span, band)
    gx = np.floor(xy[:, 0] / cluster_span).astype(int)
    gy = np.floor(xy[:, 1] / cluster_span).astype(int)
    from collections import defaultdict
    cell_members = defaultdict(list)
    for i, key in enumerate(zip(gx, gy)):
        cell_members[key].append(i)

    keys = sorted(cell_members.keys(), key=lambda k: k[1])
    from itertools import groupby
    cluster_order = []
    for i, (_gy, grp) in enumerate(groupby(keys, key=lambda k: k[1])):
        row = sorted(grp, key=lambda k: k[0])
        if i % 2 == 1:
            row.reverse()
        cluster_order.extend(row)

    out = []
    for key in cluster_order:
        members = [pts_xyzs[i] for i in cell_members[key]]
        out.extend(_serpentine(members, band))
    return out


def _norm_hex(h):
    return (h or '').strip().lstrip('#').lower()


def compute_layer_spirals(
    layer_results, mode, working_w, working_h, scale, pen_width_mm,
    poly_inset=0.08, poly_smooth=0.0, r_min=1.0, r_max=10.0, passes=2,
    canvas_hex=None,
):
    """Shared geometry: for every layer, the list of spiral fill-paths (each a list
    of (x_mm, y_mm), NOT Y-flipped) that draw its shapes with full pen-width fill —
    circle mode = Archimedean spiral, polygon mode = onion spiral converging on each
    cell's generating dot.  Used by both the G-code writer (below) and the plotter-
    ready SVG writer (plot_svg_export.py) so the two outputs are geometrically
    identical, just serialized differently.  Draw order comes from _order_multipass
    (non-adjacent, cluster-local — see its own docstring) rather than raw generation
    order, same reasoning for an SVG a plotter will trace as for direct G-code.

    A layer whose colour exactly matches `canvas_hex` means "let the paper/surface
    show through here" — there is nothing to physically draw for a pen plotter (paint
    the same colour as the paper is a no-op), so it's skipped in the returned paths.
    In polygon mode its dots still take part in the Voronoi tessellation (their cell
    boundaries still shape their neighbours' cells correctly) — only the actual
    spiral-drawing step is skipped for that layer.

    Returns: list[list[list[(x_mm, y_mm)]]] — one list of shape-paths per layer,
    in the same order as `layer_results`.
    """
    step = max(float(pen_width_mm), 0.05)

    cells = None
    ci = None
    site_radius_mm = None
    xs = ys = None
    if mode == 'polygon':
        xsL, ysL, ciL = [], [], []
        for li, (circles, _h, _n) in enumerate(layer_results):
            for (x, y, r) in circles:
                xsL.append(x); ysL.append(y); ciL.append(li)
        xs = np.asarray(xsL, float); ys = np.asarray(ysL, float)
        ci = np.asarray(ciL, np.int32)
        if len(xs) >= 3:
            cells = _voronoi_polygons(xs, ys, working_w, working_h)
    inset_px = poly_inset * (r_min + r_max) * 0.5

    if mode == 'polygon' and cells is not None:
        site_radius_mm = np.zeros(len(xs), dtype=float)
        for gi in range(len(xs)):
            poly = cells[gi]
            if poly is None or len(poly) < 3:
                continue
            poly_arr = np.asarray(poly, dtype=float)
            dv = np.hypot(poly_arr[:, 0] - xs[gi], poly_arr[:, 1] - ys[gi])
            site_radius_mm[gi] = float(dv.max()) * scale

    tiles = None
    tile_layer_idx = None
    if mode == 'tile':
        from .optimizer import _flow_tiles
        xsT, ysT, rsT, ciT = [], [], [], []
        for li, (circles, _h, _n) in enumerate(layer_results):
            for (x, y, r) in circles:
                xsT.append(x); ysT.append(y); rsT.append(r); ciT.append(li)
        xsT = np.asarray(xsT, float); ysT = np.asarray(ysT, float)
        rsT = np.asarray(rsT, float); ciT = np.asarray(ciT, np.int32)
        if len(xsT) >= 3:
            layer_bgr = [tuple(reversed(_hex_rgb(hexc))) for _c, hexc, _n in layer_results]
            colors_bgr = np.asarray([layer_bgr[c] for c in ciT], dtype=np.float32)
            tiles, tile_pt_idx = _flow_tiles(xsT, ysT, rsT, colors_bgr, working_w, working_h)
            tile_layer_idx = (ciT[np.asarray(tile_pt_idx, dtype=np.int64)]
                               if tiles else np.array([], dtype=np.int32))

    def shape_paths_for_layer(li, circles):
        if mode == 'polygon' and cells is not None:
            idxs = np.where(ci == li)[0]
            site_pts = [
                (float(xs[gi]) * scale, float(ys[gi]) * scale, float(site_radius_mm[gi]), int(gi))
                for gi in idxs
            ]
            for x_mm, y_mm, _size, gi in _order_multipass(site_pts, passes):
                poly = cells[gi]
                if poly is None:
                    continue
                pts = _rounded_cell_points(poly, poly_smooth, inset_px)
                if pts is None or len(pts) < 3:
                    continue
                coords_mm = [(px * scale, py * scale) for px, py in pts]
                sp = _polygon_spiral(coords_mm, (x_mm, y_mm), step)
                if len(sp) >= 2:
                    yield sp
        elif mode == 'tile' and tiles is not None:
            idxs = np.where(tile_layer_idx == li)[0]
            tile_pts = []
            for ti in idxs:
                poly_arr = np.asarray(tiles[ti], dtype=float)
                cx, cy = poly_arr.mean(axis=0)
                size = float(np.hypot(poly_arr[:, 0] - cx, poly_arr[:, 1] - cy).max())
                tile_pts.append((cx * scale, cy * scale, size * scale, int(ti)))
            for x_mm, y_mm, _size, ti in _order_multipass(tile_pts, passes):
                pts = _tile_inset_points(tiles[ti], inset_px, poly_smooth)
                if pts is None or len(pts) < 3:
                    continue
                coords_mm = [(px * scale, py * scale) for px, py in pts]
                sp = _polygon_spiral(coords_mm, (x_mm, y_mm), step)
                if len(sp) >= 2:
                    yield sp
        else:  # circle
            circ_pts = [(x * scale, y * scale, r * scale, r * scale) for (x, y, r) in circles]
            for x_mm, y_mm, r_mm, _size in _order_multipass(circ_pts, passes):
                if r_mm <= step * 0.5:
                    # Target circle is smaller than (or equal to) the pen's own width —
                    # a single touch-down IS the mark (its diameter = pen width). The old
                    # code instead clamped the spiral's centerline radius up to step*0.5,
                    # which draws ink from centerline-halfwidth to centerline+halfwidth =
                    # a full 2×pen-width blob, nearly double the intended circle — this
                    # single point avoids that inflation entirely.
                    yield [(x_mm, y_mm)]
                    continue
                sp = _circle_spiral(x_mm, y_mm, r_mm - step * 0.5, step)
                if len(sp) >= 2:
                    yield sp

    canvas_norm = _norm_hex(canvas_hex)
    return [
        [] if canvas_norm and _norm_hex(hexc) == canvas_norm
        else list(shape_paths_for_layer(li, circles))
        for li, (circles, hexc, _name) in enumerate(layer_results)
    ]


def write_gcode(
    layer_results,          # list of (circles, hex, name) — circles in working px
    canvas_hex: str,
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    dpi: float,
    mode: str,              # 'circle' | 'polygon'
    pen_width_mm: float,
    poly_inset: float = 0.08,
    poly_smooth: float = 0.0,
    r_min: float = 1.0,
    r_max: float = 10.0,
    passes: int = 2,        # draw-order passes per colour (avoids adjacent-shape jumps)
    per_layer: bool = True,
    feed_draw: float = 1500.0,
    feed_travel: float = 3000.0,
    z_up: float = 5.0,
    z_down: float = 0.0,
    decimals: int = 3,
    origin: str = 'bottom-left',   # where G-code (0,0) sits: 'top-left' | 'top-right' |
                                    # 'bottom-left' | 'bottom-right'
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale = _to_mm(physical_long_edge, unit) / float(max(working_w, working_h))  # px → mm
    canvas_w_mm = working_w * scale
    canvas_h_mm = working_h * scale
    fx = f"%.{decimals}f"
    reg_mark_paths = []
    if reg_marks:
        from .registration_marks import mark_paths
        reg_mark_paths = mark_paths(
            canvas_w_mm, canvas_h_mm,
            _to_mm(reg_mark_dist, unit), _to_mm(reg_mark_size, unit), reg_mark_shape,
        )

    # Working-space coords are always top-left origin, Y-down (standard image
    # convention). `origin` picks which corner of the artwork becomes G-code (0,0);
    # each axis is independently mirrored or left alone to land there.
    flip_x = origin in ('top-right', 'bottom-right')
    flip_y = origin in ('top-left', 'top-right')

    def flip(x_mm, y_mm):
        return (canvas_w_mm - x_mm if flip_x else x_mm,
                y_mm if flip_y else canvas_h_mm - y_mm)

    layer_spirals = compute_layer_spirals(
        layer_results, mode, working_w, working_h, scale, pen_width_mm,
        poly_inset, poly_smooth, r_min, r_max, passes, canvas_hex=canvas_hex,
    )

    def shape_paths_for_layer(li, _circles):
        return layer_spirals[li]

    def emit_shape(lines, path):
        x0, y0 = flip(*path[0])
        lines.append(f"G0 X{fx % x0} Y{fx % y0}")
        lines.append(f"G1 Z{fx % z_down} F{feed_draw:.0f}")
        for (mx, my) in path[1:]:
            fxm, fym = flip(mx, my)
            lines.append(f"G1 X{fx % fxm} Y{fx % fym} F{feed_draw:.0f}")
        lines.append(f"G0 Z{fx % z_up}")

    def header(color, n_shapes):
        return [
            "; Smalti - pen-plotter G-code",
            f"; colour {color}   shapes {n_shapes}   pen {pen_width_mm} mm   mode {mode}",
            f"; canvas {working_w * scale:.1f} x {working_h * scale:.1f} mm",
            "G21 ; millimetres", "G90 ; absolute", "G17",
            f"G0 Z{fx % z_up} F{feed_travel:.0f}",
        ]

    paths_written = []

    if per_layer:
        for li, (circles, hexc, _name) in enumerate(layer_results):
            shapes = list(shape_paths_for_layer(li, circles))
            if not shapes:
                continue
            lines = header(hexc, len(shapes))
            for sp in shapes:
                emit_shape(lines, sp)
            # Same registration marks on every per-colour file, at identical
            # physical positions, so separately plotted layers can be aligned
            # back against each other after the fact.
            for mp in reg_mark_paths:
                emit_shape(lines, mp)
            lines += ["G0 X0 Y0", "M2 ; end"]
            p = out / f"plot_layer_{li + 1:02d}_{hexc.lstrip('#').lower()}.gcode"
            p.write_text("\n".join(lines) + "\n")
            paths_written.append(str(p))
    else:
        lines = header(canvas_hex, sum(len(c) for c, _h, _n in layer_results))
        for li, (circles, hexc, _name) in enumerate(layer_results):
            shapes = list(shape_paths_for_layer(li, circles))
            if not shapes:
                continue
            lines.append(f"; --- colour {hexc} - change pen, then resume ---")
            lines.append("M0 ; pause for pen change")
            for sp in shapes:
                emit_shape(lines, sp)
        for mp in reg_mark_paths:
            emit_shape(lines, mp)
        lines += ["G0 X0 Y0", "M2 ; end"]
        p = out / "plot_all.gcode"
        p.write_text("\n".join(lines) + "\n")
        paths_written.append(str(p))

    return paths_written
