"""
Machine/CNC JSON export.

Writes every mosaic dot as physical-world coordinates + size + colour, grouped by
colour, so a paint robot (e.g. a CNC-router platform with a NEMA-17 driven paint
syringe) can read it directly: pick a colour, paint all its dots, swap, repeat.

Coordinate system: origin at the TOP-LEFT of the canvas, +x right, +y down, in the
project's physical units (matches the on-screen image).  Canvas height is included
so a bottom-left / y-up machine can flip trivially:  y_machine = canvas.height - y.
"""

import json
from pathlib import Path


def _serpentine(pts, band):
    """Order dots in a boustrophedon (snake) path — fast, but paints neighbours
    back-to-back (bad for wet paint).  Kept as an option."""
    if band <= 1e-9 or not pts:
        return pts
    from itertools import groupby
    keyed = sorted(pts, key=lambda p: (round(p[1] / band), p[0]))
    out = []
    for i, (_, grp) in enumerate(groupby(keyed, key=lambda p: round(p[1] / band))):
        row = list(grp)
        if i % 2 == 1:
            row.reverse()
        out.extend(row)
    return out


def _multipass(pts, min_sep, band, max_passes=3):
    """Split a layer's dots into a few PASSES so that within any single pass no two
    dots are closer than min_sep.  The passes are painted in order, so every dot's
    near neighbours land in a *different* pass: a whole pass (many dots ≈ real time)
    is laid down before the head comes back to fill the gaps between them — by which
    point the first pass has had time to dry.  The last pass is a mop-up for the odd
    dots that were too close to fit into an earlier pass (capped at max_passes so the
    plan stays the 2–3 passes you'd actually run).

    pts: list of (x, y, r).  Returns a list of passes, each a serpentine-ordered
    list of (x, y, r)."""
    n = len(pts)
    if n == 0:
        return []
    if min_sep <= 1e-9:
        return [_serpentine(list(pts), band)]
    from collections import defaultdict
    cell = min_sep
    ms2 = min_sep * min_sep
    last = max_passes - 1

    def fits(g, cx, cy, x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py) in g.get((cx + dx, cy + dy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < ms2:
                        return False
        return True

    def assign(items, npass):
        """Greedily place each dot in the lowest pass (0..npass-1) whose already-placed
        dots are all ≥ min_sep away.  Returns (passes, leftovers) — dots that didn't fit
        any of the npass passes."""
        grids = [defaultdict(list) for _ in range(npass)]
        buckets = [[] for _ in range(npass)]
        leftovers = []
        for (x, y, r) in _serpentine(list(items), band):
            cx, cy = int(x // cell), int(y // cell)
            for p in range(npass):
                if fits(grids[p], cx, cy, x, y):
                    grids[p][(cx, cy)].append((x, y))
                    buckets[p].append((x, y, r))
                    break
            else:
                leftovers.append((x, y, r))
        return buckets, leftovers

    # Passes 0 … last-1 are clean independent sets (no two dots within min_sep).
    buckets, leftovers = assign(pts, last) if last >= 1 else ([], list(pts))
    # The final pass is the mop-up.  Order it so it *also* disperses — recursively lay
    # its own independent sub-sets end to end — so even the leftovers rarely paint two
    # neighbours back to back, without adding another top-level pass.
    if leftovers:
        sub, extra = assign(leftovers, 8)
        final = []
        for s in sub:
            final.extend(_serpentine(s, band))
        final.extend(_serpentine(extra, band))
        buckets.append(final)
    return [_serpentine(pp, band) if p < last else pp
            for p, pp in enumerate(buckets) if pp]


def write_cnc_json(
    layer_results,          # list of (circles, hex, name)
    canvas_hex: str,
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str = 'cm',
    dpi: float = 96.0,
    passes: int = 2,         # 1 = single serpentine (fast) · 2–3 = wet-paint-safe passes
    decimals: int = 4,
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale = physical_long_edge / float(max(working_w, working_h))   # px → physical units
    canvas_w = round(working_w * scale, decimals)
    canvas_h = round(working_h * scale, decimals)

    all_r = [r for circles, _, _ in layer_results for (_, _, r) in circles]
    med_r = sorted(all_r)[len(all_r) // 2] if all_r else 1.0
    band = max(2.0 * med_r * scale, 1e-6)

    passes = int(max(1, min(3, passes)))
    multipass = passes >= 2

    # Each colour is painted in `passes` PASSES.  Within one pass every dot is at least
    # min_sep apart, so a whole pass is laid down before the head returns to fill the
    # gaps between them — giving the first dots time to dry.  min_sep is set from the
    # real dot SPACING (median nearest-neighbour distance); more passes → wider spacing
    # per pass → longer drying between neighbours.
    import numpy as np
    all_xy = np.array([(x * scale, y * scale)
                       for circles, _, _ in layer_results for (x, y, _r) in circles], dtype=float)
    if len(all_xy) >= 2:
        from scipy.spatial import cKDTree
        _d, _ = cKDTree(all_xy).query(all_xy, k=2)
        nn = float(np.median(_d[:, 1]))          # median nearest-neighbour spacing
    else:
        nn = 2.0 * med_r * scale
    nn = max(nn, 1e-6)
    # 2 passes → checkerboard spacing; 3 passes → wider spacing per pass.
    min_sep = (1.3 if passes <= 2 else 1.55) * nn
    target_passes = passes

    def _min_consecutive(dots):
        m = float('inf')
        for a, b in zip(dots, dots[1:]):
            d = ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5
            m = min(m, d)
        return m if m != float('inf') else 0.0

    layers = []
    total = 0
    per_color = {}
    worst_gap = float('inf')
    max_passes = 1
    for i, (circles, hexc, _name) in enumerate(layer_results):
        pts = [(x * scale, y * scale, r * scale) for (x, y, r) in circles]
        if multipass:
            layer_passes = _multipass(pts, min_sep, band, max_passes=target_passes)
        else:
            layer_passes = [_serpentine(pts, band)]
        max_passes = max(max_passes, len(layer_passes))
        dots = []
        pass_counts = []
        for pnum, pp in enumerate(layer_passes):
            pass_counts.append(len(pp))
            for (X, Y, R) in pp:
                dots.append({'x': round(X, decimals), 'y': round(Y, decimals),
                             'd': round(2 * R, decimals), 'r': round(R, decimals),
                             'pass': pnum})
        if len(dots) > 1:
            worst_gap = min(worst_gap, _min_consecutive(dots))
        layers.append({'index': i, 'color': hexc, 'count': len(dots),
                       'passes': pass_counts, 'dots': dots})
        total += len(dots)
        per_color[hexc] = len(dots)

    data = {
        'app': 'smalti',
        'format': 'paint-robot',
        'version': 1,
        'units': unit,
        'dpi': dpi,
        'coordinate_system': 'top-left origin, +x right, +y down, physical units; '
                             'y_machine = canvas.height - y for a bottom-left/y-up machine',
        'canvas': {
            'width': canvas_w,
            'height': canvas_h,
            'unit': unit,
            'background': canvas_hex,
        },
        'counts': {'total': total, 'colors': len(layers), 'per_color': per_color},
        'palette': [hexc for _c, hexc, _n in layer_results],
        'path_order': 'multipass' if multipass else 'serpentine',
        'wet_paint': {
            'passes_per_color': max_passes,                    # ~2–3 passes per colour
            'min_separation': round(min_sep, decimals),        # min spacing within a pass
            'achieved_min_consecutive_gap': round(worst_gap, decimals) if worst_gap != float('inf') else None,
            'note': 'Each colour is painted in passes (each dot has a "pass" field). '
                    'Within a pass every dot is ≥ min_separation apart; neighbouring dots '
                    'fall in later passes, so a whole pass is laid down before the head '
                    'returns nearby — letting the first dots dry. Paint passes in order.',
        },
        'layers': layers,
    }

    p = out / 'mosaic_paint.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=1)
    return [str(p)]
