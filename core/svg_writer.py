"""
SVG output: per-layer + combined file.
Uses physical SVG units (cm or in) for plotter-ready output.

Airbrush mode: same circle data, rendered as Gaussian blobs via radial gradients.
An EQ curve (list of scale multipliers, one per radius band) controls how much
each circle size is spread — like an audio EQ but for blob softness vs size.
"""

import math
import numpy as np
import svgwrite
from pathlib import Path
from .units import svg_canvas_size, to_svg_px, UNITS_TO_INCHES, PX_PER_INCH


def _physical_unit_str(unit: str) -> str:
    return {'in': 'in', 'cm': 'cm', 'mm': 'mm'}.get(unit, 'px')


def _svg_header_attrs(svg_w, svg_h, margin=0.0):
    """(width_attr, height_attr, viewbox_attr, bg_xywh_attrs) for an <svg>
    padded by `margin` on every side. SVG viewBoxes support a non-zero,
    negative origin natively, so padding for outward (positive-dist)
    registration marks only needs the viewBox/width/height/background
    changed -- every artwork coordinate elsewhere in these functions is
    computed in the ORIGINAL (un-padded) [0,svg_w]x[0,svg_h] frame and
    needs no translation at all; the shifted viewBox does that for free."""
    vb_x, vb_y = -margin, -margin
    vb_w, vb_h = svg_w + 2 * margin, svg_h + 2 * margin
    width_attr = f'width="{vb_w:.3f}px" height="{vb_h:.3f}px"'
    viewbox_attr = f'viewBox="{vb_x:.3f} {vb_y:.3f} {vb_w:.3f} {vb_h:.3f}"'
    bg_xywh = f'x="{vb_x:.3f}" y="{vb_y:.3f}" width="{vb_w:.3f}" height="{vb_h:.3f}"'
    return width_attr, viewbox_attr, bg_xywh


def _add_reg_marks_svgwrite(dwg, svg_w, svg_h, dist_px, size_px, shape,
                             stroke='#000000', stroke_width=1.0, offset=(0.0, 0.0)):
    from .registration_marks import all_marks
    for p in all_marks(svg_w, svg_h, dist_px, size_px, shape, offset):
        if p[0] == 'circle':
            _, cx, cy, r = p
            dwg.add(dwg.circle(center=(round(cx, 3), round(cy, 3)), r=round(r, 3),
                                fill='none', stroke=stroke, stroke_width=stroke_width))
        else:
            _, x0, y0, x1, y1 = p
            dwg.add(dwg.line(start=(round(x0, 3), round(y0, 3)), end=(round(x1, 3), round(y1, 3)),
                              stroke=stroke, stroke_width=stroke_width))


def _weighted_voronoi_cells(xs, ys, rs, W, H, inset_ratio):
    """
    Compute weighted Voronoi cells matching the JS canvas preview exactly.

    Uses Delaunay triangulation for exact true-neighbor adjacency (no K-NN
    false-neighbor gaps). Boundary between i and j sits at ri/(ri+rj) along i→j.

    Per-edge inset: each boundary shifts toward i by inset_ratio/2 × dist(i,j),
    so the visible gap between adjacent cells = inset_ratio × dist(i,j).
    If the shifted boundary passes i (t ≤ 0), the cell is dominated and omitted.

    Returns a list of (polygon_exterior_coords | None) per input point.
    """
    import numpy as np
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon, box
    from collections import defaultdict

    n = len(xs)
    if n < 3:
        return [None] * n

    points = np.column_stack([xs, ys])
    tri = Delaunay(points)

    # Delaunay edges = exact Voronoi adjacencies
    neighbors: dict = defaultdict(set)
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        for u, v in ((a, b), (b, c), (a, c)):
            neighbors[u].add(v)
            neighbors[v].add(u)

    LARGE = max(W, H) * 10
    bbox = box(0, 0, W, H)
    cells = []

    for i in range(n):
        ri = float(rs[i])
        xi, yi = float(xs[i]), float(ys[i])
        cell = bbox
        dominated = False

        for j in neighbors[i]:
            rj = float(rs[j])
            xj, yj = float(xs[j]), float(ys[j])
            dx, dy = xj - xi, yj - yi
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue

            t = ri / (ri + rj)
            if inset_ratio > 0:
                t = t - inset_ratio * 0.5
                if t <= 0:
                    dominated = True
                    break

            bx, by = xi + t * dx, yi + t * dy
            ux, uy = dx / dist, dy / dist
            px, py = -uy, ux

            # Half-plane: keep the side containing xi
            hp = Polygon([
                (bx + px * LARGE - ux * LARGE, by + py * LARGE - uy * LARGE),
                (bx - px * LARGE - ux * LARGE, by - py * LARGE - uy * LARGE),
                (bx - px * LARGE,               by - py * LARGE),
                (bx + px * LARGE,               by + py * LARGE),
            ])

            try:
                cell = cell.intersection(hp)
            except Exception:
                pass

            if cell.is_empty:
                break

        if dominated or cell.is_empty:
            cells.append(None)
            continue

        if cell.geom_type == 'MultiPolygon':
            cell = max(cell.geoms, key=lambda g: g.area)

        if cell.area < 1:
            cells.append(None)
            continue

        cells.append(list(cell.exterior.coords))

    return cells, neighbors


def _gaussian_stops(n_stops: int = 8) -> list[tuple[str, float]]:
    """
    Produce gradient stops that approximate a Gaussian falloff.
    sigma = 0.42 × radius gives ~1% opacity at the edge.
    """
    sigma = 0.42
    stops = []
    for i in range(n_stops):
        t = i / (n_stops - 1)          # 0 → 1 (centre → edge)
        opacity = math.exp(-(t ** 2) / (2 * sigma ** 2))
        stops.append((f'{t*100:.0f}%', round(opacity, 4)))
    return stops


_GAUSSIAN_STOPS = _gaussian_stops()


def _eq_scale(r: float, r_min: float, r_max: float, eq_curve: list[float]) -> float:
    """
    Map circle radius → blob scale multiplier using the EQ curve.
    eq_curve has N values corresponding to N evenly-spaced radius bands.
    Interpolates linearly between bands.
    """
    n = len(eq_curve)
    if n == 1 or r_max <= r_min:
        return eq_curve[0]
    t = (r - r_min) / (r_max - r_min)   # 0 = smallest, 1 = largest
    t = max(0.0, min(1.0, t))
    idx = t * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return eq_curve[lo] * (1 - frac) + eq_curve[hi] * frac


def _airbrush_layer_svg(circles, color_hex, output_path, svg_w, svg_h, scale,
                         r_min, r_max, eq_curve, bg_hex='white', extra_svg='', margin=0.0):
    stops_xml = ''.join(
        f'<stop offset="{off}" stop-color="{color_hex}" stop-opacity="{op}"/>'
        for off, op in _GAUSSIAN_STOPS
    )
    grad = (
        f'<radialGradient id="ag" cx="50%" cy="50%" r="50%"'
        f' gradientUnits="objectBoundingBox">{stops_xml}</radialGradient>'
    )
    # Draw largest splats first (bottom), smallest last (top) — SVG's painter's-algorithm
    # stacking otherwise lets a big splat drawn after a small one fully bury it, even
    # though the small dot is visually meant to read as sitting "on top".
    ordered = sorted(circles, key=lambda c: c[2] * _eq_scale(c[2], r_min, r_max, eq_curve), reverse=True)
    elems = []
    for x, y, r in ordered:
        br = r * _eq_scale(r, r_min, r_max, eq_curve) * scale
        cx, cy = x * scale, y * scale
        elems.append(
            f'<rect x="{cx-br:.3f}" y="{cy-br:.3f}"'
            f' width="{br*2:.3f}" height="{br*2:.3f}"'
            f' fill="url(#ag)"/>'
        )
    width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' {width_attr}'
        f' {viewbox_attr}>'
        f'<defs>{grad}</defs>'
        f'<rect {bg_xywh} fill="{bg_hex}"/>'
        f'<g stroke="none">{"".join(elems)}</g>'
        f'{extra_svg}'
        f'</svg>'
    )
    with open(output_path, 'w') as f:
        f.write(svg)


def _airbrush_combined_svg(layers, canvas_hex, output_path, svg_w, svg_h, scale,
                             r_min, r_max, eq_curve, unified_order=True, extra_svg='', margin=0.0):
    """`unified_order=True`: a single size-ordered pass across ALL colours — largest
    splats first (bottom), smallest last (top) — so a later colour's big splat never
    fully buries an earlier colour's small dot regardless of colour/layer order.  Same
    convention already used by the raster/video export and the live JS preview.

    `unified_order=False`: the old behaviour — one <g> per colour, colours stacked in
    layer order, each colour's own splats drawn in their original order.  Useful when
    a workflow needs every colour's splats grouped together (e.g. per-colour selection
    in an SVG editor) rather than interleaved by size."""
    defs_parts = []
    for idx, (_, color_hex) in enumerate(layers):
        stops_xml = ''.join(
            f'<stop offset="{off}" stop-color="{color_hex}" stop-opacity="{op}"/>'
            for off, op in _GAUSSIAN_STOPS
        )
        defs_parts.append(
            f'<radialGradient id="ag{idx}" cx="50%" cy="50%" r="50%"'
            f' gradientUnits="objectBoundingBox">{stops_xml}</radialGradient>'
        )

    if unified_order:
        all_splats = []   # (br, cx, cy, layer_idx)
        for idx, (circles, _) in enumerate(layers):
            for x, y, r in circles:
                br = r * _eq_scale(r, r_min, r_max, eq_curve) * scale
                all_splats.append((br, x * scale, y * scale, idx))
        all_splats.sort(key=lambda s: s[0], reverse=True)

        elems = [
            f'<rect x="{cx-br:.3f}" y="{cy-br:.3f}"'
            f' width="{br*2:.3f}" height="{br*2:.3f}"'
            f' fill="url(#ag{idx})"/>'
            for br, cx, cy, idx in all_splats
        ]
        body = f'<g stroke="none">{"".join(elems)}</g>'
    else:
        groups = []
        for idx, (circles, _) in enumerate(layers):
            elems = []
            for x, y, r in circles:
                br = r * _eq_scale(r, r_min, r_max, eq_curve) * scale
                cx, cy = x * scale, y * scale
                elems.append(
                    f'<rect x="{cx-br:.3f}" y="{cy-br:.3f}"'
                    f' width="{br*2:.3f}" height="{br*2:.3f}"'
                    f' fill="url(#ag{idx})"/>'
                )
            groups.append(f'<g stroke="none">{"".join(elems)}</g>')
        body = ''.join(groups)

    width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' {width_attr}'
        f' {viewbox_attr}>'
        f'<defs>{"".join(defs_parts)}</defs>'
        f'<rect {bg_xywh} fill="{canvas_hex}"/>'
        f'{body}'
        f'{extra_svg}'
        '</svg>'
    )
    with open(output_path, 'w') as f:
        f.write(svg)


def _coords_to_pts(coords, scale):
    return ' '.join(f'{x*scale:.3f},{y*scale:.3f}' for x, y in coords)


def _smooth_polygon_elem(coords, scale, smooth, inset_scaled=0.0):
    """Return an SVG <polygon> or <path> element for one cell — a single fill-only
    shape (NO stroke).  Grout between cells is geometric: each cell is inset by
    ``inset_scaled`` output-px, so the canvas colour shows through the real gaps.

    smooth=0, inset=0 → straight <polygon> (cells touch).
    otherwise         → morphological OPENING: erode by (corner_radius + inset),
                        dilate by corner_radius.  This rounds every corner with
                        true SVG arc commands AND pulls the whole shape inward by
                        ``inset`` — one clean closed curve per cell, no outline."""
    if smooth <= 0 and inset_scaled <= 0.05:
        return f'<polygon points="{_coords_to_pts(coords, scale)}"/>'
    return _opening_rounded_path(list(coords)[:-1], scale, smooth, inset_scaled)


def _opening_rounded_path(coords, scale, smooth, inset_scaled=0.0):
    """Morphological-opening rounded+inset path (SVG). coords: (x,y) list, no dup.

    r      = corner radius (from smooth)
    E      = erosion depth = r + inset_scaled  → shape ends up inset by inset_scaled
    """
    pts = [(x * scale, y * scale) for x, y in coords]
    n = len(pts)
    if n < 3:
        return f'<polygon points="{" ".join(f"{p[0]:.3f},{p[1]:.3f}" for p in pts)}"/>'

    # Signed area (→ scale + orientation) and centroid (for inward-normal test).
    area2 = 0.0
    cxg = 0.0
    cyg = 0.0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        area2 += a[0] * b[1] - b[0] * a[1]
        cxg += a[0]
        cyg += a[1]
    cxg /= n
    cyg /= n
    cell_scale = math.sqrt(abs(area2) * 0.5)
    r = smooth * cell_scale * 0.5     # matches JS: target corner radius
    E = r + inset_scaled              # erosion depth (rounding + grout inset)
    if E < 0.01:
        return f'<polygon points="{" ".join(f"{p[0]:.3f},{p[1]:.3f}" for p in pts)}"/>'

    def build_edges(idxs):
        out = []
        for i in idxs:
            a = pts[i]
            b = pts[(i + 1) % n]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                continue
            dx /= ln
            dy /= ln
            nx = -dy
            ny = dx
            if nx * (cxg - a[0]) + ny * (cyg - a[1]) < 0:   # point inward
                nx = -nx
                ny = -ny
            out.append({'dx': dx, 'dy': dy, 'nx': nx, 'ny': ny, 'len': ln,
                        'px': a[0] + nx * E, 'py': a[1] + ny * E})   # offset by E
        return out

    def isect(e1, e2):
        denom = e1['dx'] * e2['dy'] - e1['dy'] * e2['dx']
        if abs(denom) < 1e-9:
            return None
        t = ((e2['px'] - e1['px']) * e2['dy'] - (e2['py'] - e1['py']) * e2['dx']) / denom
        return (e1['px'] + t * e1['dx'], e1['py'] + t * e1['dy'])

    # Erode by E: drop edges whose along-edge span inverts (collapsed).
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
        worst = -1
        worst_val = 0.0
        if ok:
            for k in range(m):
                cS = C[k]
                cE = C[(k + 1) % m]
                span = (cE[0] - cS[0]) * Es[k]['dx'] + (cE[1] - cS[1]) * Es[k]['dy']
                if span < -1e-6 and span < worst_val:
                    worst_val = span
                    worst = k
        if ok and worst < 0:
            centres = C
            break
        if worst < 0:   # intersection failed → drop shortest edge
            worst = min(range(len(idxs)), key=lambda k: Es[k]['len'] if k < len(Es) else 1e18)
        idxs.pop(worst)

    # Fully collapsed → inscribe a circle (radius = inradius − inset).  Cells too
    # small to survive the grout inset simply vanish (become part of the gap).
    if not centres or len(centres) < 3 or len(idxs) < 3:
        rin = math.inf
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                continue
            dist = abs(dx * (cyg - a[1]) - dy * (cxg - a[0])) / ln
            rin = min(rin, dist)
        rin = (rin - inset_scaled) if math.isfinite(rin) else 0.0
        if rin < 0.5:
            return ''   # too small after grout inset → omit (grout swallows it)
        return (f'<path d="M {cxg + rin:.3f} {cyg:.3f}'
                f' A {rin:.3f} {rin:.3f} 0 0 1 {cxg - rin:.3f} {cyg:.3f}'
                f' A {rin:.3f} {rin:.3f} 0 0 1 {cxg + rin:.3f} {cyg:.3f} Z"/>')

    Efin = build_edges(idxs)
    m = len(Efin)

    # r≈0 → pure inset with sharp corners: straight polygon through eroded vertices.
    if r < 0.01:
        pts_str = ' '.join(f'{c[0]:.3f},{c[1]:.3f}' for c in centres)
        return f'<polygon points="{pts_str}"/>'

    # Dilate by r: tangent points sit r inside each eroded vertex → land on the
    # edge line offset inward by (E − r) = inset.  Arcs of radius r at the vertices.
    def tan_start(k):
        c = centres[k]
        e = Efin[k]
        return (c[0] - e['nx'] * r, c[1] - e['ny'] * r)

    def tan_end(k):
        c = centres[(k + 1) % m]
        e = Efin[k]
        return (c[0] - e['nx'] * r, c[1] - e['ny'] * r)

    t0 = tan_start(0)
    d = [f'M {t0[0]:.3f} {t0[1]:.3f}']
    for k in range(m):
        te = tan_end(k)
        d.append(f'L {te[0]:.3f} {te[1]:.3f}')
        c = centres[(k + 1) % m]
        ts = tan_start((k + 1) % m)     # start of next edge = other end of this arc
        # Choose sweep/large-arc so the arc bulges OUTWARD (away from centre c),
        # robust to merged corners spanning more than 180°.
        a0 = math.atan2(te[1] - c[1], te[0] - c[0])
        a1 = math.atan2(ts[1] - c[1], ts[0] - c[0])
        d_ccw = (a1 - a0) % (2 * math.pi)
        e1 = Efin[k]
        e2 = Efin[(k + 1) % m]
        outx = -(e1['nx'] + e2['nx'])   # outward bisector
        outy = -(e1['ny'] + e2['ny'])
        mid_ccw = a0 + d_ccw / 2.0      # midpoint angle if sweeping increasing
        px = math.cos(mid_ccw)
        py = math.sin(mid_ccw)
        if px * outx + py * outy >= 0:  # increasing-angle arc bulges outward
            sweep = 1
            span = d_ccw
        else:
            sweep = 0
            span = 2 * math.pi - d_ccw
        large = 1 if span > math.pi else 0
        d.append(f'A {r:.3f} {r:.3f} 0 {large} {sweep} {ts[0]:.3f} {ts[1]:.3f}')
    d.append('Z')
    return f'<path d="{" ".join(d)}"/>'


def _mix_hex(a: str, b: str) -> str:
    """Average two hex colors: used so edge(i→j) == edge(j→i)."""
    return '#{:02x}{:02x}{:02x}'.format(
        (int(a[1:3], 16) + int(b[1:3], 16)) // 2,
        (int(a[3:5], 16) + int(b[3:5], 16)) // 2,
        (int(a[5:7], 16) + int(b[5:7], 16)) // 2,
    )


def _gradient_triangles_svg(xs, ys, rs, colors_hex, cells, neighbors, scale, W, H, smooth=0.0, inset_scaled=0.0):
    """
    Build SVG defs + elements for pie-slice gradient triangles.

    Each cell is a fan of triangles (centre → vertex[e] → vertex[e+1]).  Each
    triangle carries a linearGradient from the centre (cell base colour) out to
    that edge's midpoint, fading to the halfway colour between this cell and the
    neighbour nearest that edge:  edge_color = avg(colors_hex[i], colors_hex[j]).
    The average is identical from both sides, so shared edges match perfectly.

    When smooth>0 each cell's triangle fan is clipped to the SAME rounded outline
    used by the flat/export polygons (morphological opening), so gradient cells
    round exactly like everything else and never poke into the grout.
    Returns (defs_xml, elems_xml).
    """
    from scipy.spatial import KDTree
    import numpy as _np

    n = len(xs)
    if n == 0:
        return '', ''

    # KD-tree to find which neighbor j is closest to each edge midpoint
    pts = _np.column_stack([xs, ys])
    tree = KDTree(pts)

    defs = []
    elems = []
    grad_id = 0
    clip_id = 0

    for i in range(n):
        coords = cells[i]
        if coords is None:
            continue
        cx, cy = float(xs[i]), float(ys[i])
        center_color = colors_hex[i]
        true_neighbors = neighbors.get(i, set())

        poly = list(coords)
        np_poly = len(poly) - 1  # exterior.coords has a closing duplicate vertex

        cell_tris = []
        for e in range(np_poly):
            v0 = poly[e]
            v1 = poly[(e + 1) % np_poly]
            mx = (v0[0] + v1[0]) * 0.5
            my = (v0[1] + v1[1]) * 0.5

            # Find the true Delaunay neighbor closest to this edge midpoint.
            # Query k=3 and pick the closest one that is actually a true neighbor.
            k_query = min(4, n)
            dists, idxs = tree.query([mx, my], k=k_query)
            neighbor_j = None
            for idx in (idxs if hasattr(idxs, '__iter__') else [idxs]):
                if int(idx) != i and int(idx) in true_neighbors:
                    neighbor_j = int(idx)
                    break
            edge_color = _mix_hex(center_color, colors_hex[neighbor_j]) if neighbor_j is not None else center_color

            gid = f'pg{grad_id}'
            grad_id += 1
            defs.append(
                f'<linearGradient id="{gid}"'
                f' x1="{cx*scale:.3f}" y1="{cy*scale:.3f}"'
                f' x2="{mx*scale:.3f}" y2="{my*scale:.3f}"'
                f' gradientUnits="userSpaceOnUse">'
                f'<stop offset="0%" stop-color="{center_color}"/>'
                f'<stop offset="100%" stop-color="{edge_color}"/>'
                f'</linearGradient>'
            )
            pts_str = (
                f'{cx*scale:.3f},{cy*scale:.3f} '
                f'{v0[0]*scale:.3f},{v0[1]*scale:.3f} '
                f'{v1[0]*scale:.3f},{v1[1]*scale:.3f}'
            )
            cell_tris.append(f'<polygon points="{pts_str}" fill="url(#{gid})" stroke="none"/>')

        if smooth > 0 or inset_scaled > 0.05:
            # Clip the fan to the rounded+inset cell outline (same shape as flat cells)
            # so gradient cells match and the grout gap shows the canvas through.
            cid = f'gc{clip_id}'
            clip_id += 1
            clip_shape = _opening_rounded_path(poly[:-1], scale, smooth, inset_scaled)
            if clip_shape:
                defs.append(f'<clipPath id="{cid}">{clip_shape}</clipPath>')
                elems.append(f'<g clip-path="url(#{cid})">{"".join(cell_tris)}</g>')
        else:
            elems.extend(cell_tris)

    return ''.join(defs), ''.join(elems)


# ── Shared-circumcenter cells (identical geometry to the on-screen preview) ────
#
# The preview builds cells from _voronoi_polygons: every Voronoi vertex is a
# Delaunay-triangle circumcenter, SHARED exactly between the cells around it, so
# adjacent cells interlock with no floating-point gaps.  The old SVG engine used
# _weighted_voronoi_cells (independent per-cell shapely clips) whose vertices did
# NOT coincide → the little triangular gaps at three-cell junctions.  Using the
# same shared-circumcenter polygons here makes the export match the preview and
# removes those gaps.  Grout is GEOMETRIC: each cell is inset by half the grout
# width so the canvas colour shows through real gaps — every cell is one clean
# fill-only closed shape (no stroke outline layer to confuse a vector editor).

def _closed_cells(polys):
    """Convert _voronoi_polygons output ([[x,y],…] | None) to closed-with-duplicate
    coord lists, the format _smooth_polygon_elem / _gradient_triangles_svg expect."""
    out = []
    for p in polys:
        if p is None or len(p) < 3:
            out.append(None)
        else:
            pts = [(float(x), float(y)) for x, y in p]
            out.append(pts + [pts[0]])   # closing duplicate
    return out


def _voronoi_layer_svg(circles, color_hex, output_path, svg_w, svg_h, scale,
                        inset_ratio, r_min, r_max, bg_hex='white', gradient=False, smooth=0.0,
                        extra_svg='', margin=0.0):
    from .optimizer import _voronoi_polygons, _delaunay_neighbors
    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]
    rs = [c[2] for c in circles]
    W = svg_w / scale
    H = svg_h / scale
    cells = _closed_cells(_voronoi_polygons(xs, ys, W, H))
    # Gap between two adjacent cells = inset_ratio·(r_min+r_max); each cell insets half.
    inset_scaled = inset_ratio * (r_min + r_max) * 0.5 * scale

    if gradient:
        neighbors = {i: set(n) for i, n in enumerate(_delaunay_neighbors(xs, ys, W, H))}
        colors_hex = [color_hex] * len(xs)
        defs_xml, elems_xml = _gradient_triangles_svg(xs, ys, rs, colors_hex, cells, neighbors, scale, W, H, smooth, inset_scaled)
        width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' {width_attr}'
            f' {viewbox_attr}>'
            f'<defs>{defs_xml}</defs>'
            f'<rect {bg_xywh} fill="{bg_hex}"/>'
            f'{elems_xml}{extra_svg}</svg>'
        )
    else:
        elems = [_smooth_polygon_elem(c, scale, smooth, inset_scaled) for c in cells if c is not None]
        width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' {width_attr}'
            f' {viewbox_attr}>'
            f'<rect {bg_xywh} fill="{bg_hex}"/>'
            f'<g fill="{color_hex}" stroke="none">{"".join(elems)}</g>'
            f'{extra_svg}'
            f'</svg>'
        )
    with open(output_path, 'w') as f:
        f.write(svg)


def _voronoi_combined_svg(layers, canvas_hex, output_path, svg_w, svg_h, scale,
                           inset_ratio, r_min, r_max, gradient=False, smooth=0.0, extra_svg='', margin=0.0):
    from .optimizer import _voronoi_polygons, _delaunay_neighbors
    all_xs, all_ys, all_rs, all_colors = [], [], [], []
    for circles, color_hex in layers:
        for x, y, r in circles:
            all_xs.append(x)
            all_ys.append(y)
            all_rs.append(r)
            all_colors.append(color_hex)

    W = svg_w / scale
    H = svg_h / scale
    cells = _closed_cells(_voronoi_polygons(all_xs, all_ys, W, H))
    neighbors = {i: set(n) for i, n in enumerate(_delaunay_neighbors(all_xs, all_ys, W, H))}
    # Geometric grout: each cell insets by half the gap width → clean fill-only shapes.
    inset_scaled = inset_ratio * (r_min + r_max) * 0.5 * scale

    if gradient:
        defs_xml, elems_xml = _gradient_triangles_svg(
            all_xs, all_ys, all_rs, all_colors, cells, neighbors, scale, W, H, smooth, inset_scaled
        )
        width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' {width_attr}'
            f' {viewbox_attr}>'
            f'<defs>{defs_xml}</defs>'
            f'<rect {bg_xywh} fill="{canvas_hex}"/>'
            f'{elems_xml}{extra_svg}</svg>'
        )
    else:
        # One clean fill-only shape per cell, grouped by colour.  The grout is the
        # geometric inset (canvas rect shows through), so there is no stroke layer.
        color_elems: dict[str, list[str]] = {}
        for coords, color_hex in zip(cells, all_colors):
            if coords is None:
                continue
            elem = _smooth_polygon_elem(coords, scale, smooth, inset_scaled)
            if elem:
                color_elems.setdefault(color_hex, []).append(elem)
        groups = [
            f'<g fill="{col}" stroke="none">{"".join(elems)}</g>'
            for col, elems in color_elems.items()
        ]
        width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' {width_attr}'
            f' {viewbox_attr}>'
            f'<rect {bg_xywh} fill="{canvas_hex}"/>'
            + ''.join(groups) + extra_svg +
            '</svg>'
        )
    with open(output_path, 'w') as f:
        f.write(svg)


def _compute_tile_data(layer_results, svg_w, svg_h, scale):
    """One shared tile tessellation across ALL layers, in SVG-px units, each
    tile resolved to the hex colour of its nearest source point. Computed ONCE
    by write_all_outputs and handed to every per-layer file AND the combined
    file, so physically cut-out per-colour tiles still fit back together
    against the combined layout -- unlike polygon mode's per-file Voronoi
    (independent per layer is fine there, cells re-touch at the same shared
    boundary regardless of which points fed them), a tile's exact footprint
    has to match exactly across files or the cut pieces won't align."""
    from .optimizer import _flow_tiles
    from .raster_export import _hex_rgb
    all_xs, all_ys, all_rs, all_hex, all_bgr = [], [], [], [], []
    for circles, hex_color, *_rest in layer_results:
        bgr = tuple(reversed(_hex_rgb(hex_color)))
        for x, y, r in circles:
            all_xs.append(x * scale); all_ys.append(y * scale); all_rs.append(r * scale)
            all_hex.append(hex_color); all_bgr.append(bgr)
    if len(all_xs) < 3:
        return [], []
    colors_bgr = np.asarray(all_bgr, dtype=np.float32)
    tiles, tile_pt_idx = _flow_tiles(all_xs, all_ys, all_rs, colors_bgr, svg_w, svg_h)
    tile_hex = [all_hex[i] for i in tile_pt_idx]
    return tiles, tile_hex


def _tile_layer_svg(circles, color_hex, output_path, svg_w, svg_h, scale,
                     inset_ratio, r_min, r_max, tile_data=None,
                     bg_hex='white', extra_svg='', margin=0.0):
    from .raster_export import _tile_inset_points
    inset_scaled = inset_ratio * (r_min + r_max) * 0.5 * scale
    if tile_data is not None:
        tiles, tile_hex = tile_data
        my_tiles = [t for t, h in zip(tiles, tile_hex) if h == color_hex]
    else:
        # Fallback for direct calls outside write_all_outputs: a layer-isolated
        # tessellation from just this layer's own points (no cross-file match
        # guarantee -- only write_all_outputs' shared `tile_data` gives that).
        from .optimizer import _flow_tiles
        from .raster_export import _hex_rgb
        xs = [c[0] * scale for c in circles]
        ys = [c[1] * scale for c in circles]
        rs = [c[2] * scale for c in circles]
        my_tiles = []
        if len(xs) >= 3:
            bgr = tuple(reversed(_hex_rgb(color_hex)))
            colors_bgr = np.asarray([bgr] * len(xs), dtype=np.float32)
            my_tiles, _ = _flow_tiles(xs, ys, rs, colors_bgr, svg_w, svg_h)
    elems = []
    for poly in my_tiles:
        pts = _tile_inset_points(poly, inset_scaled, 0.0)
        if pts is None or len(pts) < 3:
            continue
        elems.append(f'<polygon points="{" ".join(f"{x:.3f},{y:.3f}" for x, y in pts)}"/>')
    width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' {width_attr}'
        f' {viewbox_attr}>'
        f'<rect {bg_xywh} fill="{bg_hex}"/>'
        f'<g fill="{color_hex}" stroke="none">{"".join(elems)}</g>'
        f'{extra_svg}'
        f'</svg>'
    )
    with open(output_path, 'w') as f:
        f.write(svg)


def _tile_combined_svg(layers, canvas_hex, output_path, svg_w, svg_h, scale,
                        inset_ratio, r_min, r_max, tile_data=None,
                        extra_svg='', margin=0.0):
    from .raster_export import _tile_inset_points
    inset_scaled = inset_ratio * (r_min + r_max) * 0.5 * scale
    if tile_data is not None:
        tiles, tile_hex = tile_data
    else:
        from .optimizer import _flow_tiles
        from .raster_export import _hex_rgb
        all_xs, all_ys, all_rs, tile_hex_src, all_bgr = [], [], [], [], []
        for circles, color_hex in layers:
            bgr = tuple(reversed(_hex_rgb(color_hex)))
            for x, y, r in circles:
                all_xs.append(x * scale); all_ys.append(y * scale); all_rs.append(r * scale)
                tile_hex_src.append(color_hex); all_bgr.append(bgr)
        tiles, tile_hex = [], []
        if len(all_xs) >= 3:
            colors_bgr = np.asarray(all_bgr, dtype=np.float32)
            tiles, tile_pt_idx = _flow_tiles(all_xs, all_ys, all_rs, colors_bgr, svg_w, svg_h)
            tile_hex = [tile_hex_src[i] for i in tile_pt_idx]
    color_elems: dict[str, list[str]] = {}
    for poly, color_hex in zip(tiles, tile_hex):
        pts = _tile_inset_points(poly, inset_scaled, 0.0)
        if pts is None or len(pts) < 3:
            continue
        elem = f'<polygon points="{" ".join(f"{x:.3f},{y:.3f}" for x, y in pts)}"/>'
        color_elems.setdefault(color_hex, []).append(elem)
    groups = [
        f'<g fill="{col}" stroke="none">{"".join(elems)}</g>'
        for col, elems in color_elems.items()
    ]
    width_attr, viewbox_attr, bg_xywh = _svg_header_attrs(svg_w, svg_h, margin)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' {width_attr}'
        f' {viewbox_attr}>'
        f'<rect {bg_xywh} fill="{canvas_hex}"/>'
        + ''.join(groups) + extra_svg +
        '</svg>'
    )
    with open(output_path, 'w') as f:
        f.write(svg)


def write_layer_svg(
    circles: list[tuple[float, float, float]],
    color_hex: str,
    output_path: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    dpi: float = 96.0,
    mode: str = 'circle',
    r_min: float = 1.0,
    r_max: float = 10.0,
    eq_curve: list[float] = None,
    poly_inset: float = 2.0,
    poly_gradient: bool = False,
    poly_smooth: float = 0.0,
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
    tile_data=None,   # (tiles, tile_hex) precomputed once by write_all_outputs
) -> None:
    svg_w, svg_h, scale = svg_canvas_size(working_w, working_h, physical_long_edge, unit, dpi)
    extra_svg = ''
    margin = 0.0
    dist_px = size_px = 0.0
    if reg_marks:
        from .registration_marks import marks_svg_snippet, required_margin
        dist_px = to_svg_px(reg_mark_dist, unit, dpi)
        size_px = to_svg_px(reg_mark_size, unit, dpi)
        margin = required_margin(dist_px, size_px, reg_mark_shape)
        # Per-layer files draw marks in THIS layer's own colour (not the
        # combined file's neutral black) -- e.g. for cutting/printing each
        # colour from its own material pass, the mark should come from that
        # same pass rather than needing a separate black one.
        extra_svg = marks_svg_snippet(svg_w, svg_h, dist_px, size_px, reg_mark_shape, stroke=color_hex)

    if mode == 'airbrush':
        _airbrush_layer_svg(
            circles, color_hex, output_path, svg_w, svg_h, scale,
            r_min, r_max, eq_curve or [1.0], extra_svg=extra_svg, margin=margin,
        )
        return

    if mode == 'polygon':
        _voronoi_layer_svg(circles, color_hex, output_path, svg_w, svg_h, scale,
                            poly_inset, r_min, r_max, gradient=poly_gradient, smooth=poly_smooth,
                            extra_svg=extra_svg, margin=margin)
        return

    if mode == 'tile':
        _tile_layer_svg(circles, color_hex, output_path, svg_w, svg_h, scale,
                         poly_inset, r_min, r_max, tile_data=tile_data,
                         extra_svg=extra_svg, margin=margin)
        return

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{svg_w + 2*margin:.3f}px", f"{svg_h + 2*margin:.3f}px"),
        viewBox=f"{-margin:.3f} {-margin:.3f} {svg_w + 2*margin:.3f} {svg_h + 2*margin:.3f}",
    )
    dwg.add(dwg.rect(insert=(-margin, -margin), size=(svg_w + 2*margin, svg_h + 2*margin), fill="white"))
    g = dwg.add(dwg.g(fill=color_hex, stroke="none"))
    for x, y, r in circles:
        g.add(dwg.circle(
            center=(round(x * scale, 3), round(y * scale, 3)),
            r=round(r * scale, 3),
        ))
    if reg_marks:
        _add_reg_marks_svgwrite(dwg, svg_w, svg_h, dist_px, size_px, reg_mark_shape, stroke=color_hex)
    dwg.save(pretty=False)


def write_combined_svg(
    layers: list[tuple[list[tuple[float, float, float]], str]],
    canvas_hex: str,
    output_path: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    dpi: float = 96.0,
    mode: str = 'circle',
    r_min: float = 1.0,
    r_max: float = 10.0,
    eq_curve: list[float] = None,
    poly_inset: float = 2.0,
    poly_gradient: bool = False,
    poly_smooth: float = 0.0,
    airbrush_unified_order: bool = True,
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
    tile_data=None,   # (tiles, tile_hex) precomputed once by write_all_outputs
) -> None:
    svg_w, svg_h, scale = svg_canvas_size(working_w, working_h, physical_long_edge, unit, dpi)
    extra_svg = ''
    margin = 0.0
    dist_px = size_px = 0.0
    if reg_marks:
        from .registration_marks import marks_svg_snippet, required_margin, contrast_color
        dist_px = to_svg_px(reg_mark_dist, unit, dpi)
        size_px = to_svg_px(reg_mark_size, unit, dpi)
        margin = required_margin(dist_px, size_px, reg_mark_shape)
        # Combined file has no single layer colour to take, so use black on a light canvas /
        # white on a dark one (plain black disappears on a dark canvas).
        extra_svg = marks_svg_snippet(svg_w, svg_h, dist_px, size_px, reg_mark_shape,
                                      stroke=contrast_color(canvas_hex))

    if mode == 'airbrush':
        _airbrush_combined_svg(
            layers, canvas_hex, output_path, svg_w, svg_h, scale,
            r_min, r_max, eq_curve or [1.0], unified_order=airbrush_unified_order,
            extra_svg=extra_svg, margin=margin,
        )
        return

    if mode == 'polygon':
        _voronoi_combined_svg(layers, canvas_hex, output_path, svg_w, svg_h, scale,
                               poly_inset, r_min, r_max, gradient=poly_gradient, smooth=poly_smooth,
                               extra_svg=extra_svg, margin=margin)
        return

    if mode == 'tile':
        _tile_combined_svg(layers, canvas_hex, output_path, svg_w, svg_h, scale,
                            poly_inset, r_min, r_max, tile_data=tile_data,
                            extra_svg=extra_svg, margin=margin)
        return

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{svg_w + 2*margin:.3f}px", f"{svg_h + 2*margin:.3f}px"),
        viewBox=f"{-margin:.3f} {-margin:.3f} {svg_w + 2*margin:.3f} {svg_h + 2*margin:.3f}",
    )
    dwg.add(dwg.rect(insert=(-margin, -margin), size=(svg_w + 2*margin, svg_h + 2*margin), fill=canvas_hex))
    for circles, color_hex in layers:
        g = dwg.add(dwg.g(fill=color_hex, stroke="none"))
        for x, y, r in circles:
            g.add(dwg.circle(
                center=(round(x * scale, 3), round(y * scale, 3)),
                r=round(r * scale, 3),
            ))
    if reg_marks:
        _add_reg_marks_svgwrite(dwg, svg_w, svg_h, dist_px, size_px, reg_mark_shape,
                                stroke=contrast_color(canvas_hex))
    dwg.save(pretty=False)


def write_all_outputs(
    layer_results: list[tuple[list, str, str]],  # (circles, hex, name)
    canvas_hex: str,
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    dpi: float = 96.0,
    mode: str = 'circle',
    r_min: float = 1.0,
    r_max: float = 10.0,
    eq_curve: list[float] = None,
    poly_inset: float = 2.0,
    poly_gradient: bool = False,
    poly_smooth: float = 0.0,
    per_layer: bool = True,
    airbrush_unified_order: bool = True,
    reg_marks: bool = False,
    reg_mark_dist: float = 0.0,   # in `unit`, same as physical_long_edge
    reg_mark_size: float = 0.0,
    reg_mark_shape: str = 'circle',
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    combined_layers = []

    # Tile mode: one shared tessellation across all layers, computed once and
    # reused by every per-layer file AND the combined file (see _compute_tile_data)
    # so cut-out per-colour tiles still fit back together against the combined
    # layout, instead of each file re-deriving its own independent placement.
    tile_data = None
    if mode == 'tile':
        svg_w, svg_h, tile_scale = svg_canvas_size(working_w, working_h, physical_long_edge, unit, dpi)
        tile_data = _compute_tile_data(layer_results, svg_w, svg_h, tile_scale)

    for i, (circles, hex_color, name) in enumerate(layer_results):
        # Per-colour layer file (one SVG per palette colour) — optional. Marks are
        # drawn identically on every layer file too (same reasoning as the combined
        # one below), since separately cut/printed layers need the same registration
        # points to be aligned back against each other.
        if per_layer:
            fname = out / f"layer_{i+1:02d}_{hex_color.lstrip('#').lower()}.svg"
            write_layer_svg(
                circles, hex_color, str(fname),
                working_w, working_h, physical_long_edge, unit, dpi, mode,
                r_min, r_max, eq_curve, poly_inset, poly_gradient, poly_smooth,
                reg_marks=reg_marks, reg_mark_dist=reg_mark_dist,
                reg_mark_size=reg_mark_size, reg_mark_shape=reg_mark_shape,
                tile_data=tile_data,
            )
            paths.append(str(fname))
        combined_layers.append((circles, hex_color))

    combined_path = str(out / "combined.svg")
    write_combined_svg(
        combined_layers, canvas_hex, combined_path,
        working_w, working_h, physical_long_edge, unit, dpi, mode,
        r_min, r_max, eq_curve, poly_inset, poly_gradient, poly_smooth,
        airbrush_unified_order=airbrush_unified_order,
        reg_marks=reg_marks, reg_mark_dist=reg_mark_dist,
        reg_mark_size=reg_mark_size, reg_mark_shape=reg_mark_shape,
        tile_data=tile_data,
    )
    paths.append(combined_path)
    return paths
