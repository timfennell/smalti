"""
3D export: turn the mosaic into a colored point cloud for Blender / 3D tools.

Each mosaic dot becomes a 3D point carrying position, color and a `radius`
attribute (so you can instance spheres scaled per-point via geometry nodes).

Layouts
-------
flat   : every point on the image plane (Z-up, X-right), depth = 0.
cloud  : ANAMORPHIC cloud.  Each point is placed somewhere along the ray from a
         virtual camera through its 2D image position.  Because every point stays
         on its own camera-ray, the whole cloud still resolves into the original
         picture when viewed from that camera — from any other angle it's a
         scattered cloud.  Each point's radius is scaled by its distance from the
         camera so the resolved image keeps the correct apparent dot sizes.

Coordinates: image plane in X (right) / Z (up), depth along Y.  The camera sits
in front at (0, -D, 0) looking toward +Y, so the picture resolves from Blender's
Front view.  Units match the project's physical long-edge setting.

Outputs mosaic_pointcloud.ply (colored, with radius) and a matching .csv.
"""

import csv
import math
import numpy as np
from pathlib import Path


def _hex_rgb(h: str):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _iso_frame():
    """Orthonormal frame for the isometric layout.  D is the view axis (points
    scatter along it and it's the direction an ORTHOGRAPHIC iso camera looks
    along); R (right) and U (up) span the image plane perpendicular to D, so the
    picture resolves undistorted from a 3/4 isometric camera (az 45°, el 35.26°)."""
    D = np.array([1.0, -1.0, 1.0]); D /= np.linalg.norm(D)     # right / toward-viewer / up
    up = np.array([0.0, 0.0, 1.0])
    R = np.cross(up, D); R /= np.linalg.norm(R)
    U = np.cross(D, R);  U /= np.linalg.norm(U)
    return R, U, D


def camera_setup(layout: str, working_w: int, working_h: int,
                 physical_long_edge: float, camera_dist: float = 2.0) -> dict:
    """Camera that resolves the mosaic for a given layout.  Returns location,
    look-at target, projection type and framing (used by the Blender script and
    documented in file headers)."""
    scale = physical_long_edge / float(max(working_w, working_h))
    half_w = working_w * 0.5 * scale
    half_h = working_h * 0.5 * scale
    L = physical_long_edge
    Dcam = max(camera_dist, 0.1) * L
    if layout == 'cloud':
        return {'type': 'PERSP', 'location': (0.0, -Dcam, 0.0), 'target': (0.0, 0.0, 0.0),
                'dist': Dcam, 'fit_w': 2 * half_w, 'fit_h': 2 * half_h}
    if layout == 'iso':
        R, U, D = _iso_frame()
        loc = (D * Dcam).tolist()
        return {'type': 'ORTHO', 'location': tuple(loc), 'target': (0.0, 0.0, 0.0),
                'ortho_scale': 2 * max(half_w, half_h), 'fit_w': 2 * half_w, 'fit_h': 2 * half_h}
    # flat
    return {'type': 'ORTHO', 'location': (0.0, -Dcam, 0.0), 'target': (0.0, 0.0, 0.0),
            'ortho_scale': 2 * max(half_w, half_h), 'fit_w': 2 * half_w, 'fit_h': 2 * half_h}


def _norm_hex(h):
    return (h or '').strip().lstrip('#').lower()


def _placed_points(layer_results, working_w, working_h, physical_long_edge,
                   layout='cloud', spread=0.4, camera_dist=2.0, seed=1, canvas_hex=None):
    """Yield (X, Y, Z, r, g, b, radius) for every dot, positioned per layout.

    A layer whose colour exactly matches `canvas_hex` means "let the surface show
    through here" — there's nothing to physically place for a point cloud/print,
    same reasoning as the G-code/Plot SVG exporters — so it's skipped entirely."""
    scale = physical_long_edge / float(max(working_w, working_h))
    cx = working_w * 0.5
    cy = working_h * 0.5
    L = physical_long_edge
    Dcam = max(camera_dist, 0.1) * L
    spread = float(max(0.0, min(0.95, spread)))
    is_flat = (layout == 'flat') or (spread <= 1e-6)
    Riso, Uiso, Diso = _iso_frame()
    rng = np.random.default_rng(int(seed))
    canvas_norm = _norm_hex(canvas_hex)
    for circles, hexc, _name in layer_results:
        if canvas_norm and _norm_hex(hexc) == canvas_norm:
            continue
        r_, g_, b_ = _hex_rgb(hexc)
        for (x, y, rad) in circles:
            X0 = (x - cx) * scale
            Z0 = -(y - cy) * scale
            R0 = rad * scale
            if is_flat:
                X, Y, Z, radius = X0, 0.0, Z0, R0
            elif layout == 'iso':
                t = rng.uniform(-spread, spread) * L
                P = X0 * Riso + Z0 * Uiso + t * Diso
                X, Y, Z, radius = float(P[0]), float(P[1]), float(P[2]), R0
            else:  # cloud
                t = 1.0 + rng.uniform(-spread, spread)
                X = t * X0
                Y = -Dcam + t * Dcam
                Z = t * Z0
                radius = R0 * t
            yield (X, Y, Z, r_, g_, b_, radius)


def write_pointcloud(
    layer_results,          # list of (circles, hex, name)
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str = 'cm',
    layout: str = 'cloud',      # 'flat' | 'cloud'
    depth: float = 0.4,         # cloud depth spread (0 = flat, 1 = ±100% of camera dist)
    camera_dist: float = 2.0,   # camera distance as a multiple of the long edge
    seed: int = 1,
    write_csv: bool = True,
    canvas_hex: str = None,     # a layer matching this colour is omitted entirely
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(_placed_points(layer_results, working_w, working_h, physical_long_edge,
                               layout=layout, spread=depth, camera_dist=camera_dist, seed=seed,
                               canvas_hex=canvas_hex))

    ply = out / 'mosaic_pointcloud.ply'
    with open(ply, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'comment Smalti mosaic point cloud (layout={layout}, units={unit})\n')
        f.write(f'element vertex {len(rows)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('property float radius\n')
        f.write('end_header\n')
        for (X, Y, Z, r_, g_, b_, radius) in rows:
            f.write(f'{X:.4f} {Y:.4f} {Z:.4f} {r_} {g_} {b_} {radius:.4f}\n')
    paths = [str(ply)]

    if write_csv:
        csvp = out / 'mosaic_pointcloud.csv'
        with open(csvp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y', 'z', 'r', 'g', 'b', 'radius'])
            for (X, Y, Z, r_, g_, b_, radius) in rows:
                w.writerow([f'{X:.4f}', f'{Y:.4f}', f'{Z:.4f}', r_, g_, b_, f'{radius:.4f}'])
        paths.append(str(csvp))

    return paths


# ── Phase 2: 3D cell solids (faceted crystals ↔ rounded potatoes) ──────────────

def _profile(h: float, round_amt: float) -> float:
    """Ring scale at normalized height h∈[-1,1].
    round_amt 0 → linear taper (straight-sided crystal / bipyramid),
    round_amt 1 → circular profile sqrt(1-h²) (rounded ellipsoid)."""
    lin = 1.0 - abs(h)
    rnd = math.sqrt(max(0.0, 1.0 - h * h))
    return (1.0 - round_amt) * lin + round_amt * rnd


def write_mesh_solids(
    layer_results,
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str = 'cm',
    r_min: float = 1.0,
    r_max: float = 10.0,
    round_amt: float = 0.5,   # 0 = faceted crystal, 1 = rounded potato
    relief: float = 1.0,      # depth multiplier: front/back extent = dot_radius × relief
    poly_inset: float = 0.06,
    layout: str = 'flat',     # 'flat' | 'cloud' | 'iso'
    scatter: float = 0.4,     # cloud/iso depth-scatter amount
    camera_dist: float = 2.0,
    seed: int = 1,
) -> list[str]:
    """Turn each Voronoi cell into a closed 3D solid whose front/back depth comes
    from its dot's radius, and export as OBJ (+ MTL colours) for Blender.

    The cell outline sits in the X–Z plane (image), the solid bulges along ±Y.
    round_amt blends the vertical profile (crystal ↔ potato) AND how many rings /
    how round the outline is."""
    import numpy as np
    from .optimizer import _voronoi_polygons
    from .raster_export import _rounded_cell_points

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale = physical_long_edge / float(max(working_w, working_h))
    cx0 = working_w * 0.5
    cy0 = working_h * 0.5
    round_amt = float(max(0.0, min(1.0, round_amt)))
    inset_phys = poly_inset * (r_min + r_max) * 0.5 * scale
    n_mid = 1 + round(round_amt * 4)            # intermediate rings: 1 (crystal) … 5 (potato)
    heights = [(-1.0 + 2.0 * (k + 1) / (n_mid + 1)) for k in range(n_mid)]

    xs, ys, rs, cols = [], [], [], []
    for circles, hexc, _name in layer_results:
        for (x, y, r) in circles:
            xs.append(x); ys.append(y); rs.append(r); cols.append(hexc)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cells = _voronoi_polygons(xs, ys, working_w, working_h) if len(xs) >= 3 else [None] * len(xs)

    # Layout: place each whole solid on the plane (flat), along a front camera ray
    # scaled by depth (cloud, resolves from front), or scattered along the iso axis
    # (iso, resolves from an isometric camera).
    is_flat = (layout == 'flat') or (scatter <= 1e-6)
    Riso, Uiso, Diso = _iso_frame()
    L = physical_long_edge
    Dcam = max(camera_dist, 0.1) * L
    Ccam = np.array([0.0, -Dcam, 0.0])
    rng = np.random.default_rng(int(seed))

    verts = []                     # list of (X,Y,Z)
    faces_by_color: dict = {}      # hex → list of face tuples (1-based idx)

    def add_v(x, y, z):
        verts.append((float(x), float(y), float(z)))
        return len(verts)          # 1-based

    for i, poly in enumerate(cells):
        if poly is None or len(poly) < 3:
            continue
        # Cell outline → physical X–Z, rounded per round_amt.
        pts_phys = [((px - cx0) * scale, -(py - cy0) * scale) for px, py in poly]
        outline = _rounded_cell_points(np.asarray(pts_phys, dtype=np.float64), round_amt, inset_phys)
        if outline is None or len(outline) < 3:
            continue
        Xc = float(np.mean(outline[:, 0]))
        Zc = float(np.mean(outline[:, 1]))
        d = float(rs[i]) * scale * max(0.01, relief)
        if d < 1e-4:
            continue

        # Per-solid layout transform place(u, w, y) → world (u,w = cross-section, y = depth).
        if is_flat:
            def place(u, w, y):
                return (u, y, w)
        elif layout == 'iso':
            tc = float(rng.uniform(-scatter, scatter)) * L
            def place(u, w, y, tc=tc):
                p = u * Riso + w * Uiso + (y + tc) * Diso
                return (p[0], p[1], p[2])
        else:  # cloud — scale the solid about the front camera by tt (apparent size const)
            tt = 1.0 + float(rng.uniform(-scatter, scatter))
            def place(u, w, y, tt=tt):
                return (Ccam[0] + tt * (u - Ccam[0]),
                        Ccam[1] + tt * (y - Ccam[1]),
                        Ccam[2] + tt * (w - Ccam[2]))

        b_apex = add_v(*place(Xc, Zc, -d))
        rings = []
        for h in heights:
            s = _profile(h, round_amt)
            Y = h * d
            rings.append([add_v(*place(Xc + s * (X - Xc), Zc + s * (Z - Zc), Y)) for (X, Z) in outline])
        f_apex = add_v(*place(Xc, Zc, d))

        flist = faces_by_color.setdefault(cols[i], [])
        m = len(outline)
        # back apex fan → first ring
        r0 = rings[0]
        for j in range(m):
            flist.append((b_apex, r0[(j + 1) % m], r0[j]))
        # quads between rings
        for a in range(len(rings) - 1):
            ra, rb = rings[a], rings[a + 1]
            for j in range(m):
                j2 = (j + 1) % m
                flist.append((ra[j], ra[j2], rb[j2], rb[j]))
        # last ring → front apex fan
        rl = rings[-1]
        for j in range(m):
            flist.append((f_apex, rl[j], rl[(j + 1) % m]))

    # ── write MTL + OBJ ──
    def _hx(h):
        h = h.lstrip('#')
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)

    mtl = out / 'mosaic_solids.mtl'
    color_names = {}
    with open(mtl, 'w') as f:
        for idx, hexc in enumerate(faces_by_color.keys()):
            name = f'c{idx}_{hexc.lstrip("#").lower()}'
            color_names[hexc] = name
            r_, g_, b_ = _hx(hexc)
            f.write(f'newmtl {name}\nKd {r_:.4f} {g_:.4f} {b_:.4f}\nKa 0 0 0\nd 1\nillum 1\n\n')

    obj = out / 'mosaic_solids.obj'
    with open(obj, 'w') as f:
        f.write(f'# Smalti mosaic 3D cell solids (round={round_amt:.2f}, units={unit})\n')
        f.write('mtllib mosaic_solids.mtl\n')
        # Write in the standard OBJ Y-up convention (our world is Z-up): world
        # (X,Y,Z) → OBJ (X, Z, -Y).  Blender's OBJ importer converts Y-up→Z-up on
        # load, bringing it back to our coordinates so it lines up with the camera.
        for (x, y, z) in verts:
            f.write(f'v {x:.4f} {z:.4f} {-y:.4f}\n')
        for hexc, flist in faces_by_color.items():
            f.write(f'usemtl {color_names[hexc]}\n')
            for face in flist:
                f.write('f ' + ' '.join(str(k) for k in face) + '\n')

    return [str(obj), str(mtl)]


# ── 3D Gaussian splats (.ply) ──────────────────────────────────────────────────

def write_gaussian_splat(
    layer_results,
    output_dir: str,
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str = 'cm',
    layout: str = 'cloud',
    depth: float = 0.4,
    camera_dist: float = 2.0,
    seed: int = 1,
    softness: float = 1.0,     # gaussian size vs dot radius (airbrush feel)
) -> list[str]:
    """Standard 3D Gaussian Splat PLY (binary, SH degree 0).  Each mosaic dot is a
    colored isotropic gaussian, placed per layout so the splat cloud resolves into
    the image from the matching camera.  Viewable in gsplat viewers and Blender
    Gaussian-Splat add-ons."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(_placed_points(layer_results, working_w, working_h, physical_long_edge,
                               layout=layout, spread=depth, camera_dist=camera_dist, seed=seed))
    N = len(rows)
    C0 = 0.28209479177387814
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
              ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
              ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4'),
              ('opacity', '<f4'),
              ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
              ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')]
    arr = np.zeros(N, dtype=fields)
    xyz = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float32)
    rgb = np.array([[r[3], r[4], r[5]] for r in rows], dtype=np.float32) / 255.0
    radii = np.array([r[6] for r in rows], dtype=np.float32)
    sigma = np.maximum(radii * 0.5 * max(0.05, softness), 1e-4)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['f_dc_0'] = (rgb[:, 0] - 0.5) / C0
    arr['f_dc_1'] = (rgb[:, 1] - 0.5) / C0
    arr['f_dc_2'] = (rgb[:, 2] - 0.5) / C0
    arr['opacity'] = 6.0                       # sigmoid(6) ≈ 0.9975 (near-opaque)
    ls = np.log(sigma)
    arr['scale_0'] = ls; arr['scale_1'] = ls; arr['scale_2'] = ls
    arr['rot_0'] = 1.0                          # identity quaternion (isotropic)

    p = out / 'mosaic_splat.ply'
    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'comment Smalti 3D Gaussian splats (layout={layout}, units={unit})\n'
    header += f'element vertex {N}\n'
    for name, _ in fields:
        header += f'property float {name}\n'
    header += 'end_header\n'
    with open(p, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())
    return [str(p)]


# ── Blender setup script (import + aligned camera, ready to keyframe) ───────────

def write_blender_setup(output_dir, kind, layout, working_w, working_h,
                        physical_long_edge, camera_dist=2.0, canvas_hex=None) -> list[str]:
    """Write smalti_blender_setup.py — run it in Blender's Scripting tab to import
    the exported file AND create a camera aligned so the mosaic resolves.  The
    camera is a normal object you can keyframe/animate from.

    `canvas_hex`, if given, also sets the World background colour to match the
    mosaic's canvas/surface colour — same reasoning as the export-side "skip a
    layer matching the canvas" rule: the surroundings should read as the same
    surface the dots are painted on, not Blender's default grey void."""
    out = Path(output_dir)
    cam = camera_setup(layout, working_w, working_h, physical_long_edge, camera_dist)
    fname = {'obj': 'mosaic_solids.obj', 'ply': 'mosaic_pointcloud.ply',
             'splat': 'mosaic_splat.ply'}.get(kind, 'mosaic_pointcloud.ply')
    import math as _m
    fit = max(cam['fit_w'], cam['fit_h'])
    if cam['type'] == 'PERSP':
        angle = 2.0 * _m.atan(0.5 * fit / cam['dist'])
        cam_lines = f"cam_data.type='PERSP'\ncam_data.angle={angle:.6f}"
    else:
        cam_lines = f"cam_data.type='ORTHO'\ncam_data.ortho_scale={cam['ortho_scale']:.4f}"

    # 'cloud' (anamorphic) uses a PERSPECTIVE camera — points are scattered across a
    # real depth range and only resolve into the image from that one exact viewpoint.
    # A directional Sun light (parallel rays, no position) can light a nearer point
    # from an angle that casts it as a shadow onto a farther one behind it, even
    # though both are perfectly visible to the camera — the shadow only makes sense
    # from the Sun's off-axis direction, not the camera's. A Point light placed AT
    # the exact camera location sidesteps this: light and "eye" are co-located, so
    # anything the camera can see was never in another point's shadow to begin with
    # (occluding the camera view = occluding the light, always together).
    # 'iso'/'flat' use an ORTHOGRAPHIC camera, where "same direction as camera" IS
    # already how the camera itself projects (parallel rays) — a Sun aligned to that
    # same direction is the correct, equivalent light for those layouts.
    if cam['type'] == 'PERSP':
        light_lines = (
            "light_data = bpy.data.lights.new(\"Smalti Key Light\", type='POINT')\n"
            "light_data.energy = 5000.0\n"
            "light = bpy.data.objects.new(\"Smalti Key Light\", light_data)\n"
            "bpy.context.scene.collection.objects.link(light)\n"
            "light.location = cam.location"
        )
    else:
        light_lines = (
            "light_data = bpy.data.lights.new(\"Smalti Key Light\", type='SUN')\n"
            "light_data.energy = 3.0\n"
            "light = bpy.data.objects.new(\"Smalti Key Light\", light_data)\n"
            "bpy.context.scene.collection.objects.link(light)\n"
            "light.rotation_euler = cam.rotation_euler"
        )

    if kind == 'obj':
        imp = "bpy.ops.wm.obj_import(filepath=path)"
    else:
        imp = ("try:\n    bpy.ops.wm.ply_import(filepath=path)\n"
               "except Exception:\n    bpy.ops.import_mesh.ply(filepath=path)")

    world_lines = ""
    if canvas_hex:
        cr, cg, cb = _hex_rgb(canvas_hex)
        world_lines = f'''
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("Smalti World")
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg is not None:
    bg.inputs[0].default_value = ({cr / 255.0:.4f}, {cg / 255.0:.4f}, {cb / 255.0:.4f}, 1.0)
world.color = ({cr / 255.0:.4f}, {cg / 255.0:.4f}, {cb / 255.0:.4f})
'''

    script = f'''"""Smalti — import + aligned camera.  Blender Scripting tab → Run.
Layout: {layout} · file: {fname}
For '{layout}' the camera is set to resolve the image; keyframe the camera to animate."""
import bpy, os, mathutils

DIR = r"{str(out)}"
path = os.path.join(DIR, "{fname}")

{imp}

cam_data = bpy.data.cameras.new("Smalti Camera")
{cam_lines}
cam = bpy.data.objects.new("Smalti Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
loc = mathutils.Vector({tuple(round(v, 5) for v in cam['location'])})
tgt = mathutils.Vector({tuple(round(v, 5) for v in cam['target'])})
cam.location = loc
cam.rotation_euler = (tgt - loc).normalized().to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

# Key light — lights whatever faces the camera, i.e. exactly the fronts of the
# circles/cells the camera resolves the image from (see light_lines' construction
# above for why this differs between perspective 'cloud' and orthographic 'iso'/'flat').
{light_lines}
{world_lines}
print("Smalti: imported", path, "and added resolving camera '%s' + camera-aligned key light." % cam.name)
'''
    sp = out / 'smalti_blender_setup.py'
    sp.write_text(script)
    return [str(sp)]
