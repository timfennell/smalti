"""
Smalti — photo mosaics for makers.
Run: python main.py  →  http://localhost:8484
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import tempfile
from pathlib import Path

from nicegui import ui, app, Client, core as _nicegui_core
from fastapi.responses import FileResponse, Response

from core.pipeline import PipelineConfig, PipelineResult, run_pipeline

# NiceGUI's underlying websocket transport (python-socketio/engine.io) defaults to a
# 1 MB max message size (max_http_buffer_size) — messages larger than that are
# silently REJECTED (the "Message too long" toast), not just slow. Live-preview
# payloads for a dense mosaic (tens of thousands of circles, incl. their positions,
# radii, colours, and — in Polygon mode — cell geometry) routinely reach 5-20+ MB,
# well past that default, so this was a real hard-failure mode, not just a
# performance concern. Raised generously since this only ever runs over localhost
# (native app / same-machine browser session), never a real network path where a
# huge buffer would be a DoS concern.
_nicegui_core.sio.eio.max_http_buffer_size = 200 * 1024 * 1024   # 200 MB

# Tile mode: single square tiles that vary continuously in size (same
# detail-adaptive method circle sizing uses, driven by the same r_min/r_max
# sliders), rotated to follow the local contour direction. Placement is a
# Centroidal Voronoi Diagram under a locally-rotated Manhattan metric
# (core/tile_cvd.py, after Hausner's "Simulating Decorative Mosaics"). Every
# export format renders real tiles: PNG/JPEG/TIFF, SVG, G-code, Plot SVG, and
# video/frame-sequence output.
TILE_MODE_ENABLED = True


def _res_dir() -> Path:
    """Directory holding bundled resources (assets/).  Works both when run from
    source and when frozen into a PyInstaller .app (files land in sys._MEIPASS)."""
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return Path(__file__).parent


def _kill_stale_instances() -> None:
    """Terminate the ONE previous top-level Smalti process, if any, before this one
    starts its server. Without this, a stale instance left over from a previous
    session (e.g. a force-quit or a crash that didn't clean up) keeps serving its
    OLD, in-memory code and state — launching the app again can just re-activate
    that existing window instead of starting the just-installed build, so a shipped
    fix can appear to not be there at all.

    Deliberately NOT implemented by pattern-matching `ps` output for "other
    processes that look like Smalti": macOS's process tree around a frozen .app
    (launcher/bootloader forking, multiprocessing re-executing this whole script in
    its --multiprocessing-fork child, etc.) is uncertain enough that broad matching
    risks misidentifying and killing this very process. Instead, each top-level
    launch records its own PID to a fixed file; a launch only ever kills the ONE
    specific PID the *previous* launch recorded there, never anything discovered by
    scanning the process list."""
    try:
        pid_file = Path.home() / 'Library' / 'Application Support' / 'Smalti' / 'smalti.pid'
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        my_pid = os.getpid()
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                if old_pid != my_pid:
                    os.kill(old_pid, 9)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        pid_file.write_text(str(my_pid))
    except Exception:
        pass


# True when running as the packaged/native Smalti.app (vs a browser/web session).
_APP_MODE = getattr(sys, 'frozen', False) or os.environ.get('SMALTI_APP') == '1'

# The packaged .app has no visible terminal — without this, print()s and tracebacks
# go nowhere retrievable, which made diagnosing an earlier "the app looked like it
# reset itself" report impossible after the fact. Redirect stdout/stderr AND uncaught
# exceptions to a rotating log file so a future incident leaves real evidence.
if _APP_MODE:
    import logging
    from logging.handlers import RotatingFileHandler
    _log_dir = Path.home() / 'Library' / 'Logs' / 'Smalti'
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / 'smalti.log'
    _log_handler = RotatingFileHandler(_log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    _log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[_log_handler])

    class _StreamToLog:
        # uvicorn's own logging setup (and other libraries) probe stdout/stderr for
        # normal file-object attributes like isatty()/encoding before they'll accept
        # it as a log stream — without these, uvicorn's dictConfig call raises and
        # the whole app fails to start.
        encoding = 'utf-8'

        def __init__(self, log_func):
            self._log_func = log_func
        def write(self, msg):
            msg = msg.rstrip()
            if msg:
                self._log_func(msg)
        def flush(self):
            pass
        def isatty(self):
            return False

    sys.stdout = _StreamToLog(logging.info)
    sys.stderr = _StreamToLog(logging.error)

    def _log_uncaught(exc_type, exc_value, exc_tb):
        import traceback
        logging.error('UNCAUGHT EXCEPTION:\n' + ''.join(
            traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.excepthook = _log_uncaught

    # sys.excepthook does NOT catch exceptions raised inside background threads
    # (a separate mechanism, threading.excepthook, since Python 3.8) — several
    # worker threads in this app already catch their own exceptions internally,
    # but this is a safety net for any that don't.
    def _log_uncaught_thread(args):
        import traceback
        logging.error('UNCAUGHT THREAD EXCEPTION in %r:\n%s' % (
            args.thread, ''.join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))))
    threading.excepthook = _log_uncaught_thread

    print(f'Smalti (packaged) starting up — logging to {_log_path}')

_ASSETS_DIR = _res_dir() / 'assets'
# Serve bundled static assets (vendored PixiJS, logo) so the app needs no CDN.
if _ASSETS_DIR.is_dir():
    app.add_static_files('/assets', str(_ASSETS_DIR))

# App logo mark (the rounded mosaic-blob icon) — generated by the app's own engine.
_LOGO_PATH = _ASSETS_DIR / 'logo.svg'
try:
    _logo_raw = _LOGO_PATH.read_text()
    # Strip the fixed pixel size so it scales to whatever container it sits in.
    LOGO_MARK_SVG = _logo_raw.replace('width="512" height="512"', 'width="100%" height="100%"')
except Exception:
    LOGO_MARK_SVG = '<div style="width:100%;height:100%;border-radius:10px;background:#141821;"></div>'


# Processed video frames live here (server-side) during and after processing, so the
# browser can fetch them one at a time instead of holding the whole clip in memory.
_SCRUB = {'frames': [], 'meta': {}}


@app.get('/api/video_scrub')
async def serve_video_scrub(idx: int = -1):
    fr = _SCRUB['frames']
    if idx < 0:
        return Response(json.dumps({'count': len(fr), 'meta': _SCRUB['meta']}),
                        media_type='application/json')
    if not (0 <= idx < len(fr)):
        return Response(status_code=404)
    return Response(json.dumps({'frame': fr[idx], 'meta': _SCRUB['meta'], 'count': len(fr)},
                               separators=(',', ':')), media_type='application/json')


@app.get('/api/source_image')
async def serve_source_image():
    if S.image_path and os.path.exists(S.image_path):
        return FileResponse(S.image_path)
    return Response(status_code=404)


@app.get('/api/video_frame')
async def serve_video_frame(idx: int = 0):
    """One decoded frame of the loaded video as JPEG — used by the eyedropper's
    frame shuffler so colours can be picked from any frame."""
    if not (S.video_path and os.path.exists(S.video_path)):
        return Response(status_code=404)
    import cv2
    cap = cv2.VideoCapture(S.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return Response(status_code=404)
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type='image/jpeg')


@app.get('/api/video_playback')
async def serve_video_playback():
    """The processed mosaic_video.json — fetched by the browser's frame scrubber so
    playback stays client-side (no per-frame server round-trip, no WebSocket limit)."""
    if S.video_json_path and os.path.exists(S.video_json_path):
        return FileResponse(S.video_json_path, media_type='application/json')
    return Response(status_code=404)


@app.get('/api/video_render_frame')
async def serve_video_render_frame(idx: int = 0, mode: str = 'polygon'):
    """Server-render one processed frame in the true render mode (polygon/airbrush/tile) so
    the scrubber can preview modes the client-side circle renderer can't draw. Tile mode's
    render is capped to the same 900px preview resolution as everything else here (`s`
    below) — full-resolution tile computation (what still-image export uses) takes tens of
    seconds to minutes per frame, which even scrubbing one frame at a time can't hide."""
    import cv2
    from core.video_render import _frame_layers
    from core.raster_export import render_mosaic_bgr
    frames = _SCRUB['frames']; meta = _SCRUB['meta']
    if not meta or not (0 <= idx < len(frames)):
        print(f"[video_render_frame] 404: idx={idx} mode={mode} "
              f"meta_present={meta is not None} frame_count={len(frames)}", flush=True)
        return Response(status_code=404)
    Wg, Hg = meta['working_size']
    long_edge = max(Wg, Hg) or 1
    s = min(1.0, 900.0 / long_edge)          # cap preview resolution for speed
    ow = max(2, int(Wg * s)); oh = max(2, int(Hg * s))
    hidden = S.video_hidden or set()
    visible = set(i for i in range(len(meta['palette'])) if i not in hidden)
    layers = _frame_layers(frames[idx], meta['palette'], visible)
    eq_curve = list(S.airbrush_eq) if S.airbrush_eq else meta.get('airbrush_eq')
    r_min = meta['r_min']; r_max = meta['r_max']
    poly_inset = S.poly_inset; poly_smooth = S.poly_smooth; poly_gradient = S.poly_gradient
    canvas = meta['canvas']

    # This render is CPU-heavy (esp. airbrush's per-dot compositing) and Play calls this
    # endpoint back-to-back for every advancing frame.  Running it directly in this async
    # handler would block the WHOLE event loop for each call — including the websocket's
    # own ping/pong keepalive — which is what was dropping the connection ("Connection
    # lost") during airbrush playback.  run_in_executor moves it to a worker thread so the
    # event loop stays free to service the websocket while the render happens.
    def _render():
        return render_mosaic_bgr(
            layers, canvas, Wg, Hg, ow, oh, mode=mode,
            r_min=r_min, r_max=r_max, eq_curve=eq_curve,
            poly_inset=poly_inset, poly_smooth=poly_smooth, poly_gradient=poly_gradient,
        )
    try:
        bgr = await asyncio.get_event_loop().run_in_executor(None, _render)
    except Exception:
        # An uncaught exception here becomes a bare 500 with no body — the
        # scrubber's <img> has no onerror handler, so that failure was
        # previously invisible: the display just silently kept showing
        # whatever it last rendered (circles), with nothing pointing at why.
        import traceback
        print(f"[video_render_frame] mode={mode} idx={idx} failed:", flush=True)
        traceback.print_exc()
        return Response(status_code=500, content=traceback.format_exc(), media_type='text/plain')
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type='image/jpeg')


def _video_frame_count() -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(S.video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(1, n)
    except Exception:
        return 1

# ── App state ──────────────────────────────────────────────────────────────────

class State:
    image_path: str = ""
    image_name: str = ""
    media_mode: str = "image"        # 'image' | 'video' — set by the media picker
    palette: list[str] = ["#C8001E", "#005CA8", "#F5C400", "#1A7A3C", "#1C1C1C"]
    canvas_hex: str = "#F2EDE4"
    unit: str = "cm"

    # Physical measurements
    r_min: float = 0.15
    r_max: float = 1.2
    gap: float = 0.05
    physical_long_edge: float = 30.0
    dpi: float = 150.0

    color_softness: float = 80.0
    canvas_comp: float = 0.15   # canvas colour show-through compensation
    chroma_weight: float = 1.0
    detail_boost: float = 4.0
    lum_influence: float = 0.0
    max_working_px: int = 1800
    source_native_long: int = 0     # longest edge (px) of the currently loaded source's
                                     # native resolution — 0 = unknown. Drives the
                                     # Full/½/¼/⅛ working-resolution buttons.
    working_res_fraction: float = 1.0   # 1.0 / 0.5 / 0.25 / 0.125 — fraction of
                                         # source_native_long used for max_working_px
    video_exporting: bool = False   # re-entrancy guard for render_video_out — without
                                     # this, a fast double-click (or clicking again before
                                     # the button's "loading" state round-trips back to the
                                     # browser) starts a second ffmpeg process writing the
                                     # SAME output file concurrently, interleaving the two
                                     # encoded streams into one corrupted .mov.
    poisson_k: int = 30
    n_iterations: int = 15
    dither_strength: float = 1.0
    blend_radius: float = 8.0
    dither_kernel: str = 'ball'     # 'sierra' (Sierra-3) | 'ball' (legacy 1/d spread)

    mode: str = 'circle'   # 'circle' | 'airbrush' | 'polygon'
    # Airbrush EQ: blob scale per size band [XS, S, M, L, XL]
    airbrush_eq: list = None
    poly_inset: float = 0.08   # Voronoi cell inset ratio (fraction of each cell's equivalent radius)
    poly_gradient: bool = False  # gradient interpolation between adjacent polygon colors
    poly_smooth: float = 0.0    # Catmull-Rom tension: 0 = sharp facets, 1 = smooth blobs
    pre_quantize: bool = False
    stop_event: threading.Event | None = None
    worker_thread: threading.Thread | None = None
    # Durable (server-side, shared) job status — lets a FRESH page load detect and
    # reattach to an already-running or just-finished job instead of showing a blank
    # form.  A page's local `result_holder`/`error_holder`/progress callbacks are all
    # per-connection closures; if the browser/webview reloads mid-generation (e.g. its
    # WebView renderer gets reclaimed while backgrounded), the old page's references
    # are orphaned — the background thread keeps running invisibly with nowhere to
    # report to. These S fields are what a new page checks to recover that state.
    last_error: str | None = None
    progress_pct: float = 0.0
    progress_msg: str = ''
    job_kind: str = ''               # '' | 'still' | 'video_process' | 'video_export' —
                                      # which reattach UI a fresh page should restore
    video_export_path: str = ''      # durable result of the last video export (reattach)
    video_seq_format: str = 'video'  # export format for video mode's Export section — defaults
                                      # to the actual encoded video, not a frame sequence
    video_seq_exporting: bool = False

    export_per_layer: bool = True   # also write one file per palette colour
    reg_marks_enabled: bool = False  # draw registration marks at all 4 canvas corners,
                                      # on every 2D export (SVG/raster/G-code/Plot SVG) —
                                      # combined AND per-layer files, so separately cut/
                                      # printed layers can be physically aligned to each other
    reg_mark_dist: float = 1.0      # corner-to-mark-centre distance, in `unit`
    reg_mark_size: float = 0.3      # mark radius (circle) / half-length (cross, lines), in `unit`
    reg_mark_shape: str = 'circle'  # 'circle' | 'cross' | 'lines'
    # SVG airbrush combined export: True = single size-ordered pass across all colours
    # (small splats on top regardless of colour); False = one group per colour, colours
    # stacked in layer order (old behaviour — useful for per-colour selection in an
    # SVG editor, or a workflow that wants each colour's splats kept together).
    airbrush_svg_unified_order: bool = True
    export_format: str = 'svg'      # 'svg' | 'png' | 'jpeg' | 'tiff' | 'ply'
    export_long_px: int = 2000      # raster long-edge resolution
    export_3d_layout: str = 'cloud'  # 'flat' | 'cloud' | 'iso'
    export_3d_depth: float = 0.4     # cloud/iso depth spread
    export_3d_relief: float = 1.0    # OBJ solids: object depth = dot radius × this
                                     # (roundness ← poly_smooth, spacing ← poly_inset)
    export_json_passes: int = 2      # CNC JSON: paint passes per colour
                                     # (1 = fast serpentine · 2–3 = wet-paint-safe)
    pen_width_mm: float = 0.5        # G-code: pen tip width → spiral-fill spacing
    gcode_origin: str = 'bottom-left'  # where G-code (0,0) sits on the artwork
    # ── Video → mosaic (two-stage: process clip → render video) ──────────────
    video_path: str = ""             # uploaded source clip
    video_name: str = ""
    video_json_path: str = ""        # mosaic_video.json produced by stage 1
    video_out_long: int = 1280       # output video long-edge in px
    video_working_px: int = 900      # mosaic working resolution (density / speed)
    video_r_min_px: float = 3.0      # dot radius bounds in working px
    video_r_max_px: float = 14.0
    video_iters_first: int = 8       # optimisation iterations on frame 0
    video_iters_warm: int = 5        # optimization rounds per frame (both modes)
    video_fresh: bool = True         # True = fresh full process each frame (true per-frame
                                     # optimisation); False = warm-start (coherent, persists)
    video_stride: int = 1            # process every Nth frame
    video_max_frames: int = 0        # 0 = whole clip
    video_transparent: bool = False  # render with an alpha channel (ProRes 4444)
    video_hidden: object = None      # set of palette indices hidden in render (None = none)
    # Plain folder in $HOME — not a macOS TCC-protected location (Desktop/Documents/
    # Downloads are), so the app can write here without a privacy prompt.  Choosing a
    # folder via Browse grants access to protected locations if you prefer those.
    output_dir: str = str(Path.home() / "Smalti")
    last_result: object = None   # PipelineResult — kept for re-export
    processing: bool = False
    result: PipelineResult | None = None

S = State()

# ── Session save / load ─────────────────────────────────────────────────────────
# A .paintbot.json file captures everything needed to reopen a project and
# re-export in any mode/settings without re-running the pipeline: all settings,
# the source image (embedded base64 so the file is self-contained), and the
# generated circle data (positions/radii/colours per layer).

SESSION_FIELDS = [
    'image_name', 'palette', 'canvas_hex', 'unit',
    'r_min', 'r_max', 'gap', 'physical_long_edge', 'dpi',
    'color_softness', 'canvas_comp', 'chroma_weight', 'detail_boost',
    'lum_influence', 'max_working_px', 'poisson_k', 'n_iterations',
    'dither_strength', 'blend_radius', 'dither_kernel', 'mode', 'airbrush_eq',
    'poly_inset', 'poly_gradient', 'poly_smooth', 'pre_quantize',
    'export_per_layer', 'airbrush_svg_unified_order', 'export_format', 'export_long_px',
    'export_3d_layout', 'export_3d_depth', 'export_3d_relief',
    'reg_marks_enabled', 'reg_mark_dist', 'reg_mark_size', 'reg_mark_shape',
    'export_json_passes', 'pen_width_mm', 'gcode_origin', 'output_dir',
]


def _result_to_dict(R):
    if R is None:
        return None
    return {
        'layer_circles': [
            [[[float(x), float(y), float(r)] for (x, y, r) in circles], fill, stroke]
            for (circles, fill, stroke) in R.layer_circles
        ],
        'working_size': list(R.working_size),
        'original_size': list(R.original_size),
        'total_circles': R.total_circles,
        'circles_per_layer': R.circles_per_layer,
        'r_min_px': R.r_min_px,
        'r_max_px': R.r_max_px,
    }


def _dict_to_result(d):
    if not d:
        return None
    return PipelineResult(
        layer_circles=[
            ([tuple(c) for c in circles], fill, stroke)
            for (circles, fill, stroke) in d['layer_circles']
        ],
        output_paths=[],
        working_size=tuple(d['working_size']),
        original_size=tuple(d['original_size']),
        total_circles=d['total_circles'],
        circles_per_layer=d['circles_per_layer'],
        r_min_px=d.get('r_min_px', 1.0),
        r_max_px=d.get('r_max_px', 10.0),
    )


def serialize_session() -> dict:
    import base64
    img_b64 = None
    if S.image_path and os.path.exists(S.image_path):
        with open(S.image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('ascii')
    return {
        'app': 'smalti', 'version': 1,
        'settings': {k: getattr(S, k) for k in SESSION_FIELDS},
        'image_b64': img_b64,
        'result': _result_to_dict(S.last_result),
    }


def apply_session(data: dict):
    """Restore S from a loaded session dict. Returns True on success."""
    import base64
    if not isinstance(data, dict) or data.get('app') not in ('smalti', 'facet', 'paintbot'):
        raise ValueError('Not a Smalti session file.')
    for k, v in (data.get('settings') or {}).items():
        if k in SESSION_FIELDS:
            setattr(S, k, v)
    img_b64 = data.get('image_b64')
    if img_b64:
        suffix = os.path.splitext(S.image_name or 'image.png')[1] or '.png'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(base64.b64decode(img_b64))
        tmp.close()
        S.image_path = tmp.name
    R = _dict_to_result(data.get('result'))
    S.last_result = R
    S.result = R
    return True


def _preview_json_from_result(R) -> str:
    """Rebuild the updatePaintbot preview payload from a stored result so the
    canvas can be repopulated on load (no pipeline re-run), using CURRENT
    S.mode rather than whatever mode R was originally generated in — this is
    also what lets the render-mode toggle switch to Polygon/Tile after a
    completed run and get real geometry instead of a circle fallback (see
    make_mode_setter's call to this same function)."""
    import numpy as np
    from core.optimizer import _voronoi_polygons, _delaunay_neighbors, _preview_subsample, compute_preview_tiles
    # Build the colour table straight from R's own layers rather than indexing
    # into S.palette — S.palette is live app state and can drift from whatever
    # palette actually produced R (palette edited after generating, or after a
    # session save/load). A stale-palette lookup used to silently collapse any
    # non-matching layer onto index 0, which visually read as "only 1-2 colours
    # used, circles in a grid" (many layers all drawn in the same fallback
    # colour, and same-radius gap-fill circles in flat regions pack in a
    # near-hexagonal grid on their own, which made it look like a placement
    # bug rather than a colour one).
    palette = [fill for (_, fill, _) in R.layer_circles]
    xs, ys, rs, colors = [], [], [], []
    for ci, (circles, fill, _) in enumerate(R.layer_circles):
        for (x, y, r) in circles:
            xs.append(float(x)); ys.append(float(y)); rs.append(float(r)); colors.append(ci)
    W, H = R.working_size
    xa = np.array(xs, dtype=float); ya = np.array(ys, dtype=float)
    ra = np.array(rs, dtype=float); ca = np.array(colors, dtype=int)
    # Same reasoning as pipeline.py's preview call sites: Delaunay/Voronoi/tile
    # geometry are only read by the JS preview's matching mode branch, and
    # computing them over a large circle set (this restores a completed
    # session, so counts can be in the hundreds of thousands) is expensive/
    # large enough to be worth skipping when it won't even be rendered.
    tiles = tile_color_idx = None
    if S.mode == 'polygon':
        nbrs, polys = _delaunay_neighbors(xa, ya, W, H), _voronoi_polygons(xa, ya, W, H)
        pxs, pys, prs, pcolors = xa, ya, ra, ca
    elif S.mode == 'tile' and len(xa) >= 3:
        tiles, tile_color_idx = compute_preview_tiles(xa, ya, ra, ca, palette, W, H)
        nbrs, polys = None, None
        pxs, pys, prs, pcolors = _preview_subsample(xa, ya, ra, ca)
    else:
        nbrs, polys = None, None
        pxs, pys, prs, pcolors = _preview_subsample(xa, ya, ra, ca)
    return json.dumps({
        'xs': pxs.tolist(), 'ys': pys.tolist(), 'rs': prs.tolist(),
        'colors': pcolors.tolist(),
        'palette': palette, 'canvas_hex': S.canvas_hex,
        'W': W, 'H': H, 'mode': S.mode, 'phase': 'final',
        'r_min': float(R.r_min_px), 'r_max': float(R.r_max_px),
        'neighbors': nbrs,
        'polygons': polys,
        'tiles': tiles,
        'tile_color_idx': tile_color_idx,
    })


def _osascript_path(script: str):
    import subprocess, sys
    if sys.platform != 'darwin':
        return None
    try:
        r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, timeout=300)
        return r.stdout.strip() or None
    except Exception:
        return None


def choose_save_path(default_name: str, default_dir: str):
    q = '"' + default_dir.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return _osascript_path(
        f'set f to choose file name with prompt "Save Smalti session"'
        f' default name "{default_name}" default location POSIX file {q}\n'
        'POSIX path of f'
    )


def choose_open_path(default_dir: str):
    q = '"' + default_dir.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return _osascript_path(
        f'set f to choose file with prompt "Open Smalti session"'
        f' default location POSIX file {q}\n'
        'POSIX path of f'
    )


def reveal_in_finder(path: str):
    """Open Finder with `path` selected (a file) or shown (a folder)."""
    if sys.platform != 'darwin' or not path:
        return
    try:
        subprocess.Popen(['open', '-R', path])
    except Exception:
        pass


# ── Presets ────────────────────────────────────────────────────────────────────

PALETTE_PRESETS = {
    "CMYK":         ["#00AFEF", "#EC008C", "#FFE600", "#231F20"],
    "Screen print": ["#E63946", "#2B6CB0", "#2A9D8F", "#F4A261", "#1C1C1C"],
    "Warm earth":   ["#C0392B", "#E67E22", "#F1C40F", "#8B4513", "#1C1C1C"],
    "Cool":         ["#2980B9", "#1ABC9C", "#8E44AD", "#ECF0F1", "#1C1C1C"],
    "Monochrome":   ["#FFFFFF", "#AAAAAA", "#555555", "#111111"],
}

CANVAS_PRESETS = {
    "White paper":  "#FFFFFF",
    "Cream":        "#F2EDE4",
    "Kraft":        "#C8A97A",
    "Concrete":     "#9E9689",
    "Dark wood":    "#4A3728",
    "Slate":        "#4A5568",
    "Black":        "#111111",
}

UNIT_OPTIONS = [("px", "px"), ("mm", "mm"), ("cm", "cm"), ("in", "in")]

# Size-slider ranges: physical units share one range; px needs a much larger one.
# (min, max, step) for output-long-edge, min-diameter, max-diameter, gap.
PHYS_SIZE_RANGES = {
    'physical_long_edge': (5, 200, 1),
    'r_min': (0.04, 4.0, 0.01),
    'r_max': (0.2, 20.0, 0.1),
    'gap': (-0.5, 1.0, 0.01),
    'reg_mark_dist': (0.1, 15.0, 0.05),
    'reg_mark_size': (0.05, 5.0, 0.05),
}
PX_SIZE_RANGES = {
    'physical_long_edge': (16, 8000, 1),
    'r_min': (1, 200, 1),
    'r_max': (2, 400, 1),
    'gap': (-40, 80, 1),
    'reg_mark_dist': (4, 600, 1),
    'reg_mark_size': (2, 200, 1),
}

def _px_per_unit_ui(unit: str, dpi: float) -> float:
    return {'cm': dpi / 2.54, 'in': dpi, 'mm': dpi / 25.4, 'px': 1.0}[unit]

# ── CSS ────────────────────────────────────────────────────────────────────────

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0C0D10;
  --surface:  #13151A;
  --card:     #1A1D25;
  --card2:    #1F2330;
  --border:   #2A2E3D;
  --accent:   #7C6EFA;
  --accent2:  #FA6E9E;
  --green:    #4ADE80;
  --text:     #E8EAF0;
  --sub:      #717898;
  --sub2:     #717898;
  --rad:      14px;
  --rad-sm:   8px;
}

body {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}

.nicegui-content { padding: 0 !important; }

/* ── Layout ── */
.app-shell {
  display: grid;
  grid-template-columns: 360px 360px 1fr;
  grid-template-rows: 64px 1fr;
  grid-template-areas: "header header header" "sidebar algorithm main";
  min-height: 100vh;
}
/* Native app: no in-page header → drop the reserved header row entirely. */
.app-shell.no-header {
  grid-template-rows: 1fr;
  grid-template-areas: "sidebar algorithm main";
}

.app-header {
  grid-area: header;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 28px;
  gap: 14px;
}

.app-sidebar {
  grid-area: sidebar;
  background: var(--bg);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 20px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.app-algorithm {
  grid-area: algorithm;
  background: var(--bg);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 20px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.app-main {
  grid-area: main;
  padding: 20px 18px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* ── UI text-size control (fixed, outside the zoomed root so it never rescales) ── */
.ui-zoom-ctrl {
  position: fixed;
  bottom: 14px;
  right: 14px;
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 4px 6px 4px 10px;
  font-size: 11px;
  color: var(--sub);
  box-shadow: 0 2px 10px rgba(0,0,0,0.35);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
}
.ui-zoom-ctrl.visible {
  opacity: 1;
  pointer-events: auto;
}
.zoom-btn {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 5px;
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
}
.zoom-btn:hover { background: var(--accent); border-color: var(--accent); color: #fff; }
#ui-zoom-pct { min-width: 32px; text-align: center; color: var(--text); font-weight: 600; }

/* ── Cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--rad);
  padding: 16px;
}
.card-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--sub);
  margin-bottom: 12px;
}

/* ── Upload zone ── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: var(--rad);
  min-height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: border-color 0.15s;
  padding: 16px;
  text-align: center;
  color: var(--sub);
  font-size: 13px;
}
.upload-zone:hover { border-color: var(--accent); color: var(--text); }
.upload-done { border-color: var(--green) !important; color: var(--green) !important; }

/* ── Sliders ── */
.param-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.param-label {
  font-size: 12px;
  color: var(--text);
  width: 162px;
  flex-shrink: 0;
}
.param-value {
  font-size: 12px;
  color: var(--accent);
  min-width: 56px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}

/* ── Palette ── */
.swatch {
  width: 24px; height: 24px;
  border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.12);
  display: inline-block;
  flex-shrink: 0;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 10px 4px 6px;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.chip:hover { border-color: var(--accent2); }
.chip-x { color: var(--sub2); font-size: 15px; line-height: 1; }

/* ── Buttons ── */
.btn-primary {
  width: 100%;
  padding: 13px 0;
  border-radius: var(--rad-sm);
  border: none;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff;
  font-weight: 700;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.15s;
  letter-spacing: 0.03em;
}
.btn-primary:hover { opacity: 0.85; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-sm {
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--accent);
  background: var(--accent) !important;
  color: white;
  font-size: 11px;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn-sm:hover { opacity: 0.85; }

/* ── Progress ── */
.progress-track {
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: var(--border);
  overflow: hidden;
  margin: 8px 0;
}
.progress-fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  transition: width 0.3s ease;
}

/* ── Result tiles ── */
.layer-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: var(--rad-sm);
  background: var(--card2);
  margin-bottom: 6px;
}
.file-pill {
  font-family: 'Menlo', 'Monaco', monospace;
  font-size: 11px;
  color: var(--accent);
  background: rgba(124,110,250,0.1);
  border: 1px solid rgba(124,110,250,0.25);
  border-radius: 5px;
  padding: 2px 8px;
  word-break: break-all;
  margin: 2px 0;
}

/* ── Segment control (units) ── */
.seg-ctrl {
  display: inline-flex;
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: var(--rad-sm);
  overflow: hidden;
}
/* Locked while a generation runs — switching modes mid-optimization can crash. */
.seg-ctrl.locked { pointer-events: none; opacity: 0.45; }
.seg-btn {
  padding: 5px 14px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  color: var(--sub);
  border: none;
  background: transparent;
  transition: background 0.15s, color 0.15s;
}
.seg-btn.active {
  background: var(--accent);
  color: #fff;
}

/* ── Canvas preview dot ── */
.canvas-dot {
  width: 36px; height: 36px;
  border-radius: 50%;
  border: 2px solid var(--border);
  flex-shrink: 0;
  transition: background 0.2s;
}

/* ── Section divider ── */
.divider {
  height: 1px;
  background: var(--border);
  margin: 2px 0;
}

/* NiceGUI overrides */
.q-field__control { background: var(--card2) !important; border-radius: var(--rad-sm) !important; padding: 0 10px !important; }
.q-field__native { color: var(--text) !important; }
.q-slider .q-slider__track-container { opacity: 1; }
.q-checkbox__label { padding-left: 6px; }
</style>
"""

# ── Page ───────────────────────────────────────────────────────────────────────

CANVAS_JS = '''
<script>
(function () {
  // ── PixiJS state ─────────────────────────────────────────────────────────
  var pixiApp, pixiStage, bgGfx, mainGfx;
  var gradSprite = null, airSprite = null;
  var _gradCanvas = null, _gradCanvasTex = null;
  var _airCanvas  = null, _airCanvasTex  = null;
  var mountEl, hint;
  var canvasW = 800, canvasH = 450;
  var tx = 0, ty = 0, scale = 1;
  var dragging = false, lastX = 0, lastY = 0;
  var circleData = null;
  var initialized = false, pixiReady = false;

  // ── Helpers ──────────────────────────────────────────────────────────────
  function hexToRgb(hex) {
    return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
  }

  // Update a PixiJS Sprite to display the current contents of an offscreen canvas.
  // Always recreates the texture — PixiJS 8's Texture.from() caches by canvas reference,
  // so updating canvas pixels requires destroying the old texture to force a GPU re-upload.
  function syncCanvasSprite(sprite, offCanvas, storedTex) {
    if (storedTex && !storedTex.destroyed) storedTex.destroy(true);
    storedTex = PIXI.Texture.from(offCanvas);
    sprite.texture = storedTex;
    sprite.width   = offCanvas.width;
    sprite.height  = offCanvas.height;
    sprite.visible = true;
    return storedTex;
  }

  // Apply pan/zoom to the PixiJS stage container and trigger one GPU render.
  function applyTransform() {
    if (!pixiStage) return;
    pixiStage.x = tx; pixiStage.y = ty;
    pixiStage.scale.set(scale);
    if (pixiApp) pixiApp.render();
  }

  // ── PixiJS async init ────────────────────────────────────────────────────
  async function initPixi() {
    mountEl = document.getElementById('pb-canvas');
    hint    = document.getElementById('pb-hint');
    if (!mountEl) return;

    pixiApp = new PIXI.Application();
    await pixiApp.init({
      width:       canvasW,
      height:      canvasH,
      background:  0x13151A,
      antialias:   true,
      autoDensity: true,
      resolution:  window.devicePixelRatio || 1,
    });
    pixiApp.ticker.stop();  // manual render: we call pixiApp.render() ourselves

    var pc = pixiApp.canvas;
    pc.style.width       = '100%';
    pc.style.display     = 'block';
    pc.style.cursor      = 'grab';
    pc.style.touchAction = 'none';
    mountEl.appendChild(pc);

    // Scene graph: stage → pixiStage (pan/zoom container)
    //   bgGfx (0) → gradSprite (1) → airSprite (2) → mainGfx (3, topmost)
    pixiStage = new PIXI.Container();
    pixiApp.stage.addChild(pixiStage);

    bgGfx = new PIXI.Graphics();
    pixiStage.addChild(bgGfx);

    gradSprite = new PIXI.Sprite(PIXI.Texture.EMPTY);
    gradSprite.visible = false;
    pixiStage.addChild(gradSprite);

    airSprite = new PIXI.Sprite(PIXI.Texture.EMPTY);
    airSprite.visible = false;
    pixiStage.addChild(airSprite);

    mainGfx = new PIXI.Graphics();
    pixiStage.addChild(mainGfx);

    // ── Input events on the PixiJS canvas element ────────────────────────
    pc.addEventListener('wheel', function(e) {
      e.preventDefault();
      var rect = pc.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (canvasW / rect.width);
      var my = (e.clientY - rect.top)  * (canvasH / rect.height);
      var f = e.deltaY < 0 ? 1.12 : 1/1.12;
      tx = mx-(mx-tx)*f; ty = my-(my-ty)*f; scale *= f;
      applyTransform();
    }, {passive:false});

    window.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      var rect = pc.getBoundingClientRect();
      tx += (e.clientX - lastX) * (canvasW / rect.width);
      ty += (e.clientY - lastY) * (canvasH / rect.height);
      lastX = e.clientX; lastY = e.clientY;
      applyTransform();
    });
    pc.addEventListener('mousedown', function(e) {
      dragging=true; lastX=e.clientX; lastY=e.clientY; pc.style.cursor='grabbing';
    });
    window.addEventListener('mouseup', function() {
      dragging=false; if (pc) pc.style.cursor='grab';
    });
    pc.addEventListener('dblclick', function() { resetView(); applyTransform(); });

    var lastTouchDist = 0;
    pc.addEventListener('touchstart', function(e) {
      if (e.touches.length===1) { dragging=true; lastX=e.touches[0].clientX; lastY=e.touches[0].clientY; }
      else if (e.touches.length===2) { dragging=false; lastTouchDist=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY); }
      e.preventDefault();
    },{passive:false});
    pc.addEventListener('touchmove', function(e) {
      var rect=pc.getBoundingClientRect(), rw=canvasW/rect.width, rh=canvasH/rect.height;
      if (e.touches.length===1 && dragging) {
        tx+=(e.touches[0].clientX-lastX)*rw; ty+=(e.touches[0].clientY-lastY)*rh;
        lastX=e.touches[0].clientX; lastY=e.touches[0].clientY; applyTransform();
      } else if (e.touches.length===2) {
        var d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);
        var mx=((e.touches[0].clientX+e.touches[1].clientX)/2-rect.left)*rw;
        var my=((e.touches[0].clientY+e.touches[1].clientY)/2-rect.top)*rh;
        var f=d/lastTouchDist;
        tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; scale*=f; lastTouchDist=d; applyTransform();
      }
      e.preventDefault();
    },{passive:false});
    pc.addEventListener('touchend', function() { dragging=false; });

    new ResizeObserver(resize).observe(mountEl);
    pixiReady = true;
    resize();
    draw();
  }

  function init() {
    if (initialized) return;
    // Guard: PIXI may not be loaded yet (CDN), or Vue may not have mounted the element.
    if (typeof PIXI === 'undefined' || !document.getElementById('pb-canvas')) {
      setTimeout(init, 80);
      return;
    }
    initialized = true;
    initPixi().catch(function(err) {
      console.error('[PaintBot] PixiJS init failed:', err);
      initialized = false;  // allow retry on next updatePaintbot
    });
  }

  function resize() {
    var wrap = document.getElementById('pb-wrap');
    if (!wrap || !pixiApp) return;
    var w = wrap.clientWidth || 800;
    var aspect = circleData ? (circleData.H / circleData.W) : 0.56;
    var h = Math.round(Math.min(w * aspect, 600));
    canvasW = w; canvasH = h;
    wrap.style.minHeight = h + 'px';
    pixiApp.renderer.resize(w, h);
    draw();
  }

  // Mirrors registration_marks.py's required_margin() so the preview's default
  // fit leaves enough room for outward (positive-dist) marks to stay visible.
  function regRequiredMargin(dist, size, shape) {
    if (dist <= 0) return 0;
    return shape === 'circle' ? dist + size / 2.0 : dist + size;
  }

  function resetView() {
    if (!circleData) return;
    var reg = window.pbRegMarks;
    var margin = (reg && reg.enabled) ? regRequiredMargin(reg.dist, reg.size, reg.shape) : 0;
    var sx = canvasW / (circleData.W + 2 * margin), sy = canvasH / (circleData.H + 2 * margin);
    scale = Math.min(sx, sy) * 0.97;
    tx = (canvasW - circleData.W * scale) / 2;
    ty = (canvasH - circleData.H * scale) / 2;
  }

  // previewMode overrides circleData.mode for the canvas display only.
  // null = follow circleData.mode; 'circle', 'airbrush', or 'polygon' = force that view.
  window.pbPreviewMode = null;
  window.pbPolyInset = 0.08;
  window.pbPolyGradient = false;
  window.pbPolySmooth = 0.0;
  window.pbAirbrushEQ = [1.2, 1.2, 1.2, 1.2, 1.2];
  // Registration marks: mirrors core/registration_marks.py's geometry exactly
  // (corner_points/mark_primitives) so the live preview matches every export
  // format pixel-for-pixel in layout, just scaled to the working canvas.
  window.pbRegMarks = {enabled:false, dist:0, size:0, shape:'circle'};
  window.wvCells = null;
  window.wvCellsKey = null;
  window.wvBuildingKey = null;
  window.gradCanvas = null;
  window.gradKey = null;
  window.pbDataVersion = 0;   // increments every updatePaintbot so gradCanvas always rebuilds
  window.pbSetPreviewMode = function(m) { window.pbPreviewMode = m; draw(); };
  window.pbResetView = function() { resetView(); applyTransform(); draw(); };
  // pbSyncState is pushed from Python every 100ms.  Only redraw when a value actually
  // changed — otherwise a large canvas re-renders 10×/sec while idle.  (draw() itself
  // coalesces to one render per animation frame, so bursts of changes are cheap.)
  window.pbSyncState = function(mode,eq,polyInset,polyGradient,polySmooth,
                                 regEnabled,regDist,regSize,regShape) {
    var eqStr=(eq||[]).join(',');
    var reg=window.pbRegMarks;
    var changed = window.pbPreviewMode!==mode || window.pbPolyInset!==polyInset ||
                  window.pbPolyGradient!==polyGradient || window.pbPolySmooth!==polySmooth ||
                  window._pbEqStr!==eqStr ||
                  reg.enabled!==regEnabled || reg.dist!==regDist ||
                  reg.size!==regSize || reg.shape!==regShape;
    window.pbPreviewMode=mode; window.pbPolyInset=polyInset; window.pbAirbrushEQ=eq;
    window.pbPolyGradient=polyGradient; window.pbPolySmooth=polySmooth; window._pbEqStr=eqStr;
    window.pbRegMarks = {enabled:regEnabled, dist:regDist, size:regSize, shape:regShape};
    if (changed && circleData && pixiReady) draw();
  };

  // Mirrors registration_marks.py's corner_points/_CORNER_DIRS/mark_primitives
  // exactly: positive dist -> outside the W x H rect, negative -> inside; the
  // 'lines' shape draws only the two arms pointing away from the artwork at
  // each corner (an L-bracket), not a full 4-arm cross.
  var REG_CORNER_DIRS = [[-1,-1],[1,-1],[-1,1],[1,1]]; // TL, TR, BL, BR
  // Drawn as a black line with a white halo (two stroke passes over the same
  // geometry) so marks stay visible whether they land on the light artwork
  // or the dark preview-pane background outside it -- the exported files
  // stay plain black-only, this halo is a preview-only readability aid.
  function drawRegMarks(gfx, W, H, dist, size, shape) {
    var strokeW = Math.max(1, size * 0.05);
    var haloW = strokeW + 2;
    var corners = [[-dist,-dist],[W+dist,-dist],[-dist,H+dist],[W+dist,H+dist]];
    if (shape === 'circle') {
      for (var i=0;i<4;i++) gfx.circle(corners[i][0], corners[i][1], size/2.0);
      gfx.stroke({color:0xffffff, width:haloW});
      for (var i=0;i<4;i++) gfx.circle(corners[i][0], corners[i][1], size/2.0);
      gfx.stroke({color:0x000000, width:strokeW});
      return;
    }
    var segs = [];
    for (var i=0;i<4;i++) {
      var cx=corners[i][0], cy=corners[i][1];
      if (shape === 'cross') {
        segs.push([cx-size,cy,cx+size,cy], [cx,cy-size,cx,cy+size]);
      } else if (shape === 'lines') {
        var sx=REG_CORNER_DIRS[i][0], sy=REG_CORNER_DIRS[i][1];
        var gap = size*0.35;
        segs.push([cx+sx*gap,cy,cx+sx*size,cy], [cx,cy+sy*gap,cx,cy+sy*size]);
      }
    }
    for (var i=0;i<segs.length;i++) { gfx.moveTo(segs[i][0],segs[i][1]); gfx.lineTo(segs[i][2],segs[i][3]); }
    gfx.stroke({color:0xffffff, width:haloW});
    for (var i=0;i<segs.length;i++) { gfx.moveTo(segs[i][0],segs[i][1]); gfx.lineTo(segs[i][2],segs[i][3]); }
    gfx.stroke({color:0x000000, width:strokeW});
  }

  function eqScale(r, rMin, rMax, eq) {
    var n=eq.length; if(n===1||rMax<=rMin) return eq[0];
    var t=Math.max(0,Math.min(1,(r-rMin)/(rMax-rMin)));
    var idx=t*(n-1), lo=Math.floor(idx), hi=Math.min(lo+1,n-1);
    return eq[lo]*(1-(idx-lo))+eq[hi]*(idx-lo);
  }

  function avgColor(ha,hb) {
    var ra=hexToRgb(ha),rb=hexToRgb(hb);
    return 'rgb('+Math.round((ra[0]+rb[0])/2)+','+Math.round((ra[1]+rb[1])/2)+','+Math.round((ra[2]+rb[2])/2)+')';
  }

  // ── Polygon helpers ───────────────────────────────────────────────────

  // Signed area (positive = CCW in math coords / CW on screen).
  function polyArea(pts) {
    var a = 0, n = pts.length;
    for (var i = 0; i < n; i++) { var j=(i+1)%n; a += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]; }
    return a * 0.5;
  }

  // Uniform inset of a convex polygon by distance d (Minkowski subtraction).
  // Each edge is shifted inward by d; new vertices are the intersections of adjacent offset lines.
  function polyInset(pts, d) {
    if (!pts || pts.length < 3 || d <= 0) return pts;
    var n = pts.length, sign = polyArea(pts) >= 0 ? -1 : 1;
    var edges = [];
    for (var i = 0; i < n; i++) {
      var ax=pts[i][0], ay=pts[i][1], bx=pts[(i+1)%n][0], by=pts[(i+1)%n][1];
      var edx=bx-ax, edy=by-ay, len=Math.sqrt(edx*edx+edy*edy);
      if (len < 1e-9) continue;
      var nx=sign*edy/len, ny=sign*(-edx)/len;
      edges.push([ax+nx*d, ay+ny*d, bx+nx*d, by+ny*d]);
    }
    if (edges.length < 3) return null;
    var result = [];
    for (var i = 0; i < edges.length; i++) {
      var e1=edges[i], e2=edges[(i+1)%edges.length];
      var rx=e1[2]-e1[0], ry=e1[3]-e1[1], sx=e2[2]-e2[0], sy=e2[3]-e2[1];
      var denom=rx*sy-ry*sx;
      if (Math.abs(denom) < 1e-6) {
        result.push([e1[2], e1[3]]);   // parallel edges: use e1 endpoint
      } else {
        var t=((e2[0]-e1[0])*sy-(e2[1]-e1[1])*sx)/denom;
        result.push([e1[0]+t*rx, e1[1]+t*ry]);
      }
    }
    return result.length >= 3 ? result : null;
  }

  // Min distance from point (cx,cy) to any edge of the polygon.
  function minEdgeDist(pts, cx, cy) {
    var minD = Infinity, n = pts.length;
    for (var i = 0; i < n; i++) {
      var ax=pts[i][0], ay=pts[i][1], bx=pts[(i+1)%n][0], by=pts[(i+1)%n][1];
      var edx=bx-ax, edy=by-ay, len=Math.sqrt(edx*edx+edy*edy);
      if (len < 1e-9) continue;
      var d = Math.abs(edy*(cx-ax) - edx*(cy-ay)) / len;
      if (d < minD) minD = d;
    }
    return minD === Infinity ? 10 : minD;
  }
  // ─────────────────────────────────────────────────────────────────────

  // Weld coincident consecutive vertices (distance < 0.5px) into one.
  // This is deterministic and position-based, so two adjacent cells that share
  // a vertex weld it identically — sharing is preserved (unlike angle-based
  // simplification which could drop a vertex in one cell but not its neighbour).
  function weldPoly(pts) {
    var n = pts.length;
    if (n < 3) return null;
    var result = [];
    for (var i = 0; i < n; i++) {
      var curr = pts[i], nx = pts[(i + 1) % n];
      var dx = nx[0]-curr[0], dy = nx[1]-curr[1];
      if (dx*dx + dy*dy >= 0.25) result.push(curr);   // keep if next is > 0.5px away
    }
    return result.length >= 3 ? result : null;
  }

  // Remove polygon vertices that subtend less than minDeg degrees from center (cx,cy).
  // Each pass finds the single narrowest slice below threshold and merges those two
  // vertices into their midpoint. Repeats until no slice is below threshold.
  // Handles wrap-around correctly — each merge reduces vertex count by exactly 1.
  function collapseNarrowSlices(pts, cx, cy, minDeg) {
    var minRad = minDeg * Math.PI / 180;
    for (var guard = 0; guard < 20; guard++) {
      var n = pts.length;
      if (n < 4) break;
      var worstAngle = minRad, worstIdx = -1;
      for (var i = 0; i < n; i++) {
        var v0 = pts[i], v1 = pts[(i+1)%n];
        var ax = v0[0]-cx, ay = v0[1]-cy, bx = v1[0]-cx, by = v1[1]-cy;
        var la = Math.sqrt(ax*ax+ay*ay), lb = Math.sqrt(bx*bx+by*by);
        var cosA = (la>1e-9 && lb>1e-9) ? (ax*bx+ay*by)/(la*lb) : 1;
        var angle = Math.acos(Math.max(-1, Math.min(1, cosA)));
        if (angle < worstAngle) { worstAngle = angle; worstIdx = i; }
      }
      if (worstIdx < 0) break;
      var j = (worstIdx+1)%n;
      var mid = [(pts[worstIdx][0]+pts[j][0])*0.5, (pts[worstIdx][1]+pts[j][1])*0.5];
      var result = [];
      for (var i = 0; i < n; i++) {
        if (i === worstIdx) result.push(mid);
        else if (i !== j) result.push(pts[i]);
      }
      if (result.length >= 3) pts = result; else break;
    }
    return pts.length >= 3 ? pts : null;
  }

  // Manually subdivide a cubic bezier into N line segments on a PixiJS Graphics object.
  // Cubic gives C2 parametric continuity at knot joins; avoids PixiJS 8 coarse tessellation.
  function cBez(g, x0,y0, cp1x,cp1y, cp2x,cp2y, x1,y1) {
    var N=12;
    for(var t=1;t<=N;t++){var u=t/N,v=1-u;
      g.lineTo(v*v*v*x0+3*v*v*u*cp1x+3*v*u*u*cp2x+u*u*u*x1,
               v*v*v*y0+3*v*v*u*cp1y+3*v*u*u*cp2y+u*u*u*y1);}
  }

  // Emit a circular arc as line segments from angle a0→a1 (short way) around (cx,cy).
  // Segment count scales with ARC LENGTH (r·angle) so tiny cells emit ~2-3 segments
  // instead of a fixed 16 — critical for large canvases with thousands of cells,
  // where fixed tessellation blows up the WebGL vertex buffer and crashes the tab.
  function emitArc(g, cx, cy, r, a0, a1) {
    var d=a1-a0;
    while(d>Math.PI) d-=2*Math.PI;
    while(d<-Math.PI) d+=2*Math.PI;
    var N=Math.max(2, Math.min(24, Math.ceil(r*Math.abs(d)/3)));
    for(var k=1;k<=N;k++){var a=a0+d*k/N; g.lineTo(cx+r*Math.cos(a), cy+r*Math.sin(a));}
  }

  // Rounded convex polygon via MORPHOLOGICAL OPENING (erode by r, dilate by r).
  //
  // This is the key to "the radius keeps growing by pulling the sides toward the
  // middle."  A simple corner fillet caps the radius at what the shortest edge
  // allows, so constrained corners stay angular.  Opening removes that cap:
  //
  //   1. Erode: offset every edge inward by r.  Each interior corner becomes the
  //      intersection of its two inward-offset edge lines — this point is the
  //      centre of that corner's rounding arc.
  //   2. If r is larger than a short edge can support, that edge's offset segment
  //      inverts (its two arc-centres cross).  We DROP the collapsed edge and
  //      recompute — the two neighbouring arcs merge into one that sweeps inward
  //      toward the cell centre.  Exactly the requested behaviour.
  //   3. Dilate: the final outline is straight segments on the ORIGINAL edges
  //      (tangent points = arc-centre ± r·edge-normal) joined by circular arcs of
  //      radius r centred on the eroded vertices.
  //   4. If everything collapses (r ≥ inradius) the cell rounds to a full circle
  //      inscribed at its centre.
  //
  // Straight portions stay on the exact Voronoi edges, so adjacent cells remain
  // interlocked; corners are true circular arcs → never pointed.
  function tracePath(g, pts, smooth) {
    var n=pts.length; if(n<3) return;
    if(smooth<=0){
      g.moveTo(pts[0][0],pts[0][1]);
      for(var i=1;i<n;i++) g.lineTo(pts[i][0],pts[i][1]);
      return;
    }
    // Signed area → orientation + scale; centroid for inward-normal test.
    var area2=0, cxg=0, cyg=0;
    for(var i=0;i<n;i++){var a=pts[i],b=pts[(i+1)%n];area2+=a[0]*b[1]-b[0]*a[1]; cxg+=a[0]; cyg+=a[1];}
    cxg/=n; cyg/=n;
    var scale=Math.sqrt(Math.abs(area2)*0.5);
    var r=smooth*scale*0.5;     // target corner radius; grows past edge limits via opening
                                 // (~inradius of a regular cell at smooth=1 → maximally round
                                 //  but still edge-touching; constrained cells collapse earlier)
    if(r<0.01){
      g.moveTo(pts[0][0],pts[0][1]);
      for(var i=1;i<n;i++) g.lineTo(pts[i][0],pts[i][1]);
      return;
    }
    // Per-edge inward unit normal + a point on its inward-offset line.
    // edge i spans pts[i]→pts[i+1]; store {px,py (offset line point), nx,ny (inward normal), dx,dy (dir)}
    function buildEdges(idxs){
      var E=[];
      for(var k=0;k<idxs.length;k++){
        var i=idxs[k], a=pts[i], b=pts[(i+1)%n];
        var dx=b[0]-a[0], dy=b[1]-a[1], len=Math.sqrt(dx*dx+dy*dy);
        if(len<1e-9) continue;
        dx/=len; dy/=len;
        var nx=-dy, ny=dx;                       // one normal candidate
        // point inward toward centroid
        if(nx*(cxg-a[0])+ny*(cyg-a[1])<0){nx=-nx; ny=-ny;}
        E.push({ax:a[0],ay:a[1], dx:dx,dy:dy, nx:nx,ny:ny, len:len,
                 px:a[0]+nx*r, py:a[1]+ny*r});
      }
      return E;
    }
    // Intersect two offset lines (each: point P + t*dir). Returns arc centre.
    function isect(e1,e2){
      var denom=e1.dx*e2.dy - e1.dy*e2.dx;
      if(Math.abs(denom)<1e-9) return null;      // parallel
      var t=((e2.px-e1.px)*e2.dy - (e2.py-e1.py)*e2.dx)/denom;
      return [e1.px+t*e1.dx, e1.py+t*e1.dy];
    }
    // Iteratively erode: drop edges whose along-edge span inverts.
    var idxs=[]; for(var i=0;i<n;i++) idxs.push(i);
    var centres=null, guard=0;
    while(idxs.length>=3 && guard++<n+2){
      var E=buildEdges(idxs), m=E.length;
      if(m<3){ idxs=[]; break; }
      var C=new Array(m), ok=true, worst=-1, worstVal=0;
      for(var k=0;k<m;k++){
        C[k]=isect(E[(k-1+m)%m], E[k]);          // centre at start of edge k (between edge k-1 and k)
        if(!C[k]){ ok=false; }
      }
      if(ok){
        // For each edge, project its two end-centres onto edge dir; if end<start it collapsed.
        for(var k=0;k<m;k++){
          var cS=C[k], cE=C[(k+1)%m];
          if(!cS||!cE){ok=false;break;}
          var span=(cE[0]-cS[0])*E[k].dx+(cE[1]-cS[1])*E[k].dy;
          if(span< -1e-6 && span<worstVal){worstVal=span; worst=k;}
        }
      }
      if(ok && worst<0){ centres=C; break; }      // stable erosion
      // Drop the most-collapsed edge (or, if intersection failed, the shortest edge).
      if(worst<0){
        var sh=0; for(var k=1;k<idxs.length;k++) if(E[k]&&E[k].len<E[sh].len) sh=k;
        worst=sh;
      }
      idxs.splice(worst,1);
    }
    // Fallback: fully collapsed → inscribe a circle at the centroid.
    if(!centres || centres.length<3 || idxs.length<3){
      var rin=Infinity;
      for(var i=0;i<n;i++){var a=pts[i],b=pts[(i+1)%n];
        var dx=b[0]-a[0],dy=b[1]-a[1],len=Math.sqrt(dx*dx+dy*dy); if(len<1e-9)continue;
        var dist=Math.abs(dx*(cyg-a[1])-dy*(cxg-a[0]))/len; if(dist<rin)rin=dist;}
      if(!isFinite(rin)||rin<0.5){ g.moveTo(pts[0][0],pts[0][1]); for(var i=1;i<n;i++)g.lineTo(pts[i][0],pts[i][1]); return; }
      g.moveTo(cxg+rin,cyg);
      emitArc(g,cxg,cyg,rin,0,Math.PI); emitArc(g,cxg,cyg,rin,Math.PI,2*Math.PI);
      return;
    }
    // Dilate: draw straight edges (on original lines) + corner arcs of radius r.
    var Efin=buildEdges(idxs), m=Efin.length;
    // Tangent point on edge k at centre C[k] (start) and C[k+1] (end): centre - r*normal.
    function tanStart(k){var c=centres[k],e=Efin[k]; return [c[0]-e.nx*r, c[1]-e.ny*r];}
    function tanEnd(k){var c=centres[(k+1)%m],e=Efin[k]; return [c[0]-e.nx*r, c[1]-e.ny*r];}
    var t0=tanStart(0);
    g.moveTo(t0[0],t0[1]);
    for(var k=0;k<m;k++){
      var te=tanEnd(k);
      g.lineTo(te[0],te[1]);                       // straight run along original edge k
      // Arc at centre C[k+1], from edge k's outward normal to edge (k+1)'s outward normal.
      var c=centres[(k+1)%m], e1=Efin[k], e2=Efin[(k+1)%m];
      var a0=Math.atan2(-e1.ny,-e1.nx), a1=Math.atan2(-e2.ny,-e2.nx);
      emitArc(g, c[0],c[1], r, a0, a1);
    }
  }

  // ── Main draw ─────────────────────────────────────────────────────────────
  // Coalesced redraw: every trigger (slider tick, state sync, async build) calls
  // draw(), which flags dirty and schedules a single render on the next animation
  // frame.  Without this, rapid slider drags on a large canvas queue dozens of
  // synchronous full-scene rebuilds that block the main thread and crash the tab.
  //
  // Rendering is MANUAL (pixiApp.ticker is stopped), so a render only happens when
  // _flushRender → renderScene → applyTransform → pixiApp.render() actually runs.
  // requestAnimationFrame is throttled/paused in background tabs and can be dropped,
  // which would strand _renderScheduled=true and freeze the preview forever.  A cheap
  // safety-net interval guarantees any pending dirty render still flushes; it does
  // nothing while idle (dirty=false), so it costs nothing when the app is at rest.
  var _renderScheduled=false, _renderDirty=false;
  function _flushRender() {
    _renderScheduled=false;
    if(!_renderDirty) return;
    _renderDirty=false;
    renderScene();
  }
  function draw() {
    _renderDirty=true;
    if(_renderScheduled) return;
    _renderScheduled=true;
    requestAnimationFrame(_flushRender);
  }
  setInterval(function(){ if(_renderDirty) _flushRender(); }, 150);

  // Debounce a rebuild trigger: rapid repeated calls (e.g. every tick of a dragged
  // slider) collapse into ONE call `delay` ms after the last one, instead of doing
  // the expensive work on every intermediate value.
  var _debounceTimers = {};
  function debounceCall(key, delay, fn) {
    if (_debounceTimers[key]) clearTimeout(_debounceTimers[key]);
    _debounceTimers[key] = setTimeout(function(){ delete _debounceTimers[key]; fn(); }, delay);
  }

  // Run an expensive per-item loop (radial-gradient fills, gradient triangle fans) in
  // yielded batches so it never blocks the main thread long enough to freeze the tab.
  // TIME-budgeted (not a fixed item count): each batch runs for up to ~8ms of wall time
  // then yields, so it self-adapts regardless of how expensive each item is — a fixed
  // item count still froze the tab on very high dot counts, where even a few hundred
  // radial-gradient fills can exceed a frame budget.  The generation token is re-checked
  // after every batch — if a newer rebuild has started, this one aborts instead of
  // racing/overwriting a fresher result.
  var _chunkGen = 0;
  function runChunked(n, batchSizeHint, work, onDone) {
    var myGen = ++_chunkGen;
    var i = 0;
    var BUDGET_MS = 8;
    (function step(){
      if (myGen !== _chunkGen) return;   // superseded by a newer rebuild — abort
      var t0 = performance.now();
      while (i < n) {
        work(i); i++;
        if ((i & 15) === 0 && performance.now() - t0 > BUDGET_MS) break;  // check every 16 items
      }
      if (i < n) {
        setTimeout(step, 0);
      } else if (myGen === _chunkGen) {
        onDone();
      }
    })();
  }

  function renderScene() {
    if (!pixiReady || !circleData) return;
    var xs=circleData.xs, ys=circleData.ys, rs=circleData.rs;
    var colors=circleData.colors, palette=circleData.palette;
    var n=xs.length;
    var mode=window.pbPreviewMode!==null?window.pbPreviewMode:circleData.mode;
    // Polygon geometry is only valid when the backend ships fresh polygons.  During
    // optimization iterations they arrive stale (see updatePaintbot), so render
    // circles instead — colours always align with the current circle positions.
    if(mode==='polygon' && circleData._polysStale) mode='circle';
    // Row-packed tiles are only computed once, for the finished result (see
    // pipeline.py) — too slow to run every optimization iteration.  Render
    // circles instead until they arrive.
    if(mode==='tile' && circleData._tilesStale) mode='circle';
    var rMin=circleData.r_min||1, rMax=circleData.r_max||10;
    var W=circleData.W, H=circleData.H;

    mainGfx.clear();
    // Only hide the sprite belonging to the OTHER render mode.  While this mode's own
    // chunked rebuild is still in progress (see runChunked below), its sprite is left
    // showing whatever it last rendered instead of blanking to avoid a flicker.
    if (mode!=='polygon') gradSprite.visible=false;
    if (mode!=='airbrush') airSprite.visible=false;

    // Background rectangle lives on bgGfx so gradient/airbrush sprites above it stay visible.
    bgGfx.clear();
    bgGfx.rect(0,0,W,H).fill(circleData.canvas_hex);

    if (mode==='airbrush') {
      // Airbrush uses radial gradients — render to an offscreen 2D canvas, then
      // upload as a PixiJS texture. Cached by data version + eq settings.  The fill
      // loop runs in yielded batches (runChunked) so thousands of radial-gradient
      // fills — the single most expensive operation Canvas2D has — never block the
      // main thread long enough to freeze the tab, e.g. while dragging an EQ slider.
      var eq=window.pbAirbrushEQ||[1.2,1.2,1.2,1.2,1.2];
      var airKey=(window.pbDataVersion||0)+':'+eq.join(',');
      if (window._airKey===airKey) {
        airSprite.visible=true;
      } else if (window._airBuildingKey!==airKey) {
        window._airBuildingKey=airKey;
        if (!_airCanvas) _airCanvas=document.createElement('canvas');
        _airCanvas.width=W; _airCanvas.height=H;
        var actx=_airCanvas.getContext('2d');
        var rgbs=palette.map(function(h){var c=hexToRgb(h);return c[0]+','+c[1]+','+c[2];});
        // Draw LARGEST splats first (bottom) and smallest LAST (top) — matching the
        // exported render.  Drawing in raw array order would let a big splat from one
        // colour fully bury a small detail dot from another, regardless of which one
        // should read as "on top"; size order guarantees small dots stay visible.
        var brs=new Array(n);
        for(var i=0;i<n;i++) brs[i]=rs[i]*eqScale(rs[i],rMin,rMax,eq);
        var orderIdx=Array.from({length:n}, function(_,k){return k;});
        orderIdx.sort(function(a,b){return brs[b]-brs[a];});
        runChunked(n, 400, function(k){
          var i=orderIdx[k];
          var x=xs[i],y=ys[i],br=brs[i],rgb=rgbs[colors[i]];
          var grd=actx.createRadialGradient(x,y,0,x,y,br);
          grd.addColorStop(0,'rgba('+rgb+',1.0)'); grd.addColorStop(0.3,'rgba('+rgb+',0.85)');
          grd.addColorStop(0.6,'rgba('+rgb+',0.50)'); grd.addColorStop(0.85,'rgba('+rgb+',0.15)');
          grd.addColorStop(1,'rgba('+rgb+',0.0)');
          actx.beginPath(); actx.arc(x,y,br,0,6.2831853); actx.fillStyle=grd; actx.fill();
        }, function(){
          _airCanvasTex=syncCanvasSprite(airSprite,_airCanvas,_airCanvasTex);
          window._airKey=airKey;
          window._airBuildingKey=null;
          airSprite.visible=true;
          draw();
        });
      }
      // Still building (key already in progress): leave the previous frame's sprite
      // showing as-is rather than blanking it, so the view doesn't flicker mid-drag.

    } else if (mode==='polygon') {
      var insetRatio=window.pbPolyInset!=null?window.pbPolyInset:0.08;
      var gradMode=!!window.pbPolyGradient;
      var smooth=window.pbPolySmooth!=null?window.pbPolySmooth:0;

      var rawPolys=circleData.polygons;
      if (!rawPolys) { applyTransform(); return; }

      // Async cell build: weld coincident vertices only, 200 cells per tick.
      // NO per-cell simplification — the Python side ships mathematically shared
      // circumcenter vertices, so adjacent cells interlock exactly.  Dropping
      // near-collinear vertices per-cell (old simplifyPoly) broke that sharing
      // and produced misaligned intersections.  The circular-arc fillet in
      // tracePath is robust to collinear/degenerate vertices, so raw polygons
      // are used directly.
      // Key MUST vary with the actual data (matching gKey/airKey below) — a
      // constant key here means the welded cell SHAPES only ever get built once
      // and then reused for every later update, while colors[]/xs[]/ys[] keep
      // changing every preview tick. Once circle count/positions shift (which
      // happens on nearly every optimization step), cells[i]'s shape stops
      // corresponding to colors[i]'s color, painting the wrong color into the
      // wrong cell all over the canvas.
      var cellKey=String(window.pbDataVersion||0);
      if(window.wvCellsKey!==cellKey&&window.wvBuildingKey!==cellKey){
        window.wvBuildingKey=cellKey;
        var _built=new Array(n), _ci=0, _n=n;
        var _rp=rawPolys;
        (function buildChunk(){
          var end=Math.min(_ci+200,_n);
          for(;_ci<end;_ci++){
            var raw=_rp[_ci];
            if(!raw||raw.length<3){_built[_ci]=null;continue;}
            _built[_ci]=weldPoly(raw);
          }
          if(_ci<_n) setTimeout(buildChunk,0);
          else if(window.wvBuildingKey===cellKey){
            window.wvCells=_built;window.wvCellsKey=cellKey;window.wvBuildingKey=null;
            draw();   // cells ready → schedule a render (heartbeat no longer polls for this)
          }
        })();
      }
      var cells=window.wvCells;
      if(!cells){applyTransform();return;}

      if(gradMode){
        // Gradient fills are computed on an offscreen 2D canvas (linear gradient per pie slice)
        // then uploaded to a PixiJS Sprite. PixiJS mainGfx draws the rounding/gap stroke on top.
        // The fill loop runs in yielded batches (runChunked) — with thousands of cells this
        // was a fully synchronous main-thread loop that could freeze the tab, e.g. right
        // after toggling gradients on at a high dot count.
        var gKey=(window.pbDataVersion||0)+':'+cellKey+':'+smooth.toFixed(3);
        if (window.gradKey===gKey) {
          gradSprite.visible=true;
        } else if (window.gradBuildingKey!==gKey) {
          window.gradBuildingKey=gKey;
          if(!_gradCanvas) _gradCanvas=document.createElement('canvas');
          _gradCanvas.width=W; _gradCanvas.height=H;
          var gctx=_gradCanvas.getContext('2d');
          gctx.fillStyle=circleData.canvas_hex; gctx.fillRect(0,0,W,H);
          var nbrs=circleData.neighbors||[];
          runChunked(n, 300, function(i){
            // Welded poly when available; fall back to raw Voronoi polygon so a
            // degenerate/welded-away cell never bleeds gradient into the grout.
            var poly=cells[i];
            if(!poly){ poly=circleData.polygons&&circleData.polygons[i]; }
            if(!poly||poly.length<3) return;
            var gcx=xs[i],gcy=ys[i],centerColor=palette[colors[i]],np=poly.length;
            var myNbrs=nbrs[i]||[];
            // Per-edge target colour = halfway between this cell and the neighbour
            // nearest that edge's midpoint (shared edges resolve to the same colour).
            var eColors=new Array(np);
            for(var e=0;e<np;e++){
              var emx=(poly[e][0]+poly[(e+1)%np][0])*0.5,emy=(poly[e][1]+poly[(e+1)%np][1])*0.5;
              var bstJ=-1,bstD2=1e18;
              for(var k=0;k<myNbrs.length;k++){var j=myNbrs[k],dx2=xs[j]-emx,dy2=ys[j]-emy,d2=dx2*dx2+dy2*dy2;if(d2<bstD2){bstD2=d2;bstJ=j;}}
              eColors[e]=bstJ>=0?avgColor(centerColor,palette[colors[bstJ]]):centerColor;
            }
            // Clip every pie slice to the ROUNDED cell outline so sharp triangle
            // corners never poke past the rounded silhouette into the grout.
            gctx.save();
            gctx.beginPath();
            tracePath(gctx,poly,smooth);
            gctx.closePath();
            gctx.clip();
            // Pie slices: centre → vertex[e] → vertex[e+1].  Linear gradient runs
            // from the centre (base colour) out to the edge midpoint (halfway colour).
            for(var e=0;e<np;e++){
              var v0=poly[e], v1=poly[(e+1)%np];
              var emx=(v0[0]+v1[0])*0.5, emy=(v0[1]+v1[1])*0.5;
              var grd=gctx.createLinearGradient(gcx,gcy,emx,emy);
              grd.addColorStop(0,centerColor);grd.addColorStop(1,eColors[e]);
              gctx.beginPath();gctx.moveTo(gcx,gcy);gctx.lineTo(v0[0],v0[1]);
              gctx.lineTo(v1[0],v1[1]);
              gctx.closePath();gctx.fillStyle=grd;gctx.fill();
            }
            gctx.restore();
          }, function(){
            window.gradCanvas=_gradCanvas;
            _gradCanvasTex=syncCanvasSprite(gradSprite,_gradCanvas,_gradCanvasTex);
            window.gradKey=gKey;
            window.gradBuildingKey=null;
            gradSprite.visible=true;
            draw();
          });
        }
        // Still building (key already in progress): leave the previous frame's sprite
        // showing as-is rather than blanking it, so the view doesn't flicker mid-drag.
        // Gap + rounding stroke over the gradient sprite using GPU mainGfx
        var gapWg=insetRatio*(rMin+rMax);
        var strokeWg=gapWg;
        if(strokeWg>0.3){
          for(var i=0;i<n;i++){var poly=cells[i];if(!poly)continue;tracePath(mainGfx,poly,smooth);mainGfx.closePath();}
          mainGfx.stroke({color:circleData.canvas_hex,width:strokeWg});
        }

      } else {
        // Flat polygon mode: one fill() per cell.
        // Bucket-accumulation (many paths → one fill) triggers PixiJS 8's non-zero
        // winding rule: cells with opposite winding orders cancel and show as dark holes.
        // Individual fills avoid this entirely. rawPolys already declared above.
        for(var i=0;i<n;i++){
          // Welded poly when available; fall back to raw Voronoi polygon so a
          // degenerate/welded-away cell never shows as a background hole.
          var poly=cells[i];
          if(!poly){ poly=rawPolys&&rawPolys[i]; }
          if(!poly||poly.length<3) continue;
          tracePath(mainGfx,poly,smooth);
          mainGfx.closePath();
          mainGfx.fill(palette[colors[i]]);
        }
        var gapW=insetRatio*(rMin+rMax);
        var strokeW=gapW;
        if(strokeW>0.3){
          for(var i=0;i<n;i++){
            var poly=cells[i];
            if(!poly){ poly=rawPolys&&rawPolys[i]; }
            if(!poly||poly.length<3) continue;
            tracePath(mainGfx,poly,smooth);mainGfx.closePath();
          }
          mainGfx.stroke({color:circleData.canvas_hex,width:strokeW});
        }
      }

    } else if (mode==='tile') {
      var insetRatio=window.pbPolyInset!=null?window.pbPolyInset:0.08;
      var smooth=window.pbPolySmooth!=null?window.pbPolySmooth:0;
      var tiles=circleData.tiles||[], tileColorIdx=circleData.tile_color_idx||[];
      var nt=tiles.length;
      // tileColorIdx entries are already resolved PALETTE indices (not point
      // indices) — tiles ship once, for the finished result only, so they never
      // need to be re-cross-referenced against the (possibly subsampled) circle
      // preview's colors[] array.
      for(var i=0;i<nt;i++){
        var poly=tiles[i];
        if(!poly||poly.length<3) continue;
        tracePath(mainGfx,poly,smooth);
        mainGfx.closePath();
        mainGfx.fill(palette[tileColorIdx[i]]);
      }
      var gapWt=insetRatio*(rMin+rMax);
      if(gapWt>0.3){
        for(var i=0;i<nt;i++){
          var poly=tiles[i];
          if(!poly||poly.length<3) continue;
          tracePath(mainGfx,poly,smooth);mainGfx.closePath();
        }
        mainGfx.stroke({color:circleData.canvas_hex,width:gapWt});
      }

    } else {
      // Circle mode: bucket by color, fill each batch in one GPU draw call
      var buckets={};
      for(var i=0;i<n;i++){var c=colors[i];if(!buckets[c])buckets[c]=[];buckets[c].push(i);}
      for(var col in buckets){
        var ids=buckets[col];
        for(var j=0;j<ids.length;j++){mainGfx.circle(xs[ids[j]],ys[ids[j]],rs[ids[j]]);}
        mainGfx.fill(palette[col]);
      }
    }

    var reg=window.pbRegMarks;
    if (reg && reg.enabled && reg.size>0) {
      drawRegMarks(mainGfx, W, H, reg.dist, reg.size, reg.shape);
    }

    applyTransform();
  }

  window.updatePaintbot = function(data) {
    var needReset=!circleData;
    // The optimizer omits polygons during iterations ('polygons':null) to save
    // bandwidth, but still moves/recolours circles.  The old polygons no longer
    // line up with the new colours/positions, so mark them STALE — polygon mode
    // renders circles instead until fresh polygons arrive (cleanup/final phase).
    var polysProvided = data.polygons!=null;
    if(data.polygons==null&&circleData&&circleData.polygons) data.polygons=circleData.polygons;
    if(data.neighbors==null&&circleData&&circleData.neighbors) data.neighbors=circleData.neighbors;
    var polyChanged=!circleData||(data.polygons!==circleData.polygons);
    // Same staleness handling as polygons above — tiles only ship on the 'final'
    // phase, so intermediate iteration frames carry data.tiles===undefined.
    var tilesProvided = data.tiles!=null;
    if(!tilesProvided&&circleData&&circleData.tiles){ data.tiles=circleData.tiles; data.tile_color_idx=circleData.tile_color_idx; }
    circleData=data;
    circleData._polysStale=!polysProvided;
    circleData._tilesStale=!tilesProvided;
    if(polyChanged){
      window.wvCells=null; window.wvCellsKey=null; window.wvBuildingKey=null;
      window.gradCanvas=null; window.gradKey=null;
    }
    window.pbDataVersion=(window.pbDataVersion||0)+1;
    if(!initialized) init();
    if(!pixiReady){
      // PixiJS still initializing — retry until ready, then draw.
      setTimeout(function(){
        if(!pixiReady){ window.updatePaintbot(data); return; }
        if(hint) hint.style.display='none';
        resize(); if(needReset) resetView(); draw();
      }, 100);
      return;
    }
    if(hint) hint.style.display='none';
    resize();
    if(needReset) resetView();
    draw();
  };

  // Trigger init — use readyState check in case DOMContentLoaded already fired.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 0);  // already loaded; queue microtask so Vue finishes mounting
  }
})();
</script>
'''

EYEDROPPER_JS = '''
<script>
(function() {
  // Returns a Promise<string|null> — hex color chosen, or null if cancelled.
  // In multi mode, returns a Promise<string[]> — every color added via "Add
  // color" before "Done"/Cancel/Esc closed the overlay (never null; empty
  // array if nothing was added).
  // opts = {video:bool, frames:int, multi:bool} — video mode shows a frame
  // shuffler; multi mode keeps the overlay open across multiple picks.
  window.paintbotPickColor = function(target, opts) {
    opts = opts || {};
    var isVideo = !!opts.video;
    var multi = !!opts.multi;
    var frameCount = Math.max(1, opts.frames || 1);
    var frameIdx = 0;
    return new Promise(function(resolve) {

      // ── Overlay ────────────────────────────────────────────────────────────
      var overlay = document.createElement('div');
      overlay.style.cssText = [
        'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.85);',
        'display:flex;flex-direction:column;align-items:center;justify-content:center;',
        'font-family:sans-serif;'
      ].join('');

      var label = target === 'palette' ? (multi ? 'Add color' : 'Add to palette') : 'Set as canvas color';
      var navBtn = 'padding:6px 12px;background:#333;color:#fff;border:1px solid #555;'
                 + 'border-radius:6px;font-size:12px;cursor:pointer;';
      var doneBtnHtml = multi
        ? ('<button id="pb-ed-done" style="padding:8px 20px;background:#39c;color:#fff;'
           + 'border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;">'
           + 'Done (<span id="pb-ed-count">0</span>)</button>')
        : '';
      overlay.innerHTML = [
        '<div style="color:#fff;font-size:13px;margin-bottom:10px;opacity:0.7;">',
        'Scroll to zoom · Drag to pan · <b>Click</b> to sample · Esc to ',
        multi ? 'finish' : 'cancel', '</div>',
        '<div id="pb-ed-nav" style="display:none;align-items:center;gap:8px;margin-bottom:10px;color:#fff;">',
        '<button id="pb-ed-prev" style="' + navBtn + '">◀ Prev</button>',
        '<button id="pb-ed-shuffle" style="' + navBtn + '">🔀 Shuffle</button>',
        '<button id="pb-ed-next" style="' + navBtn + '">Next ▶</button>',
        '<input id="pb-ed-frame" type="range" min="0" max="' + (frameCount - 1) + '" value="0" style="width:240px;">',
        '<span id="pb-ed-framelbl" style="font-family:monospace;font-size:12px;min-width:90px;">frame 0 / ' + (frameCount - 1) + '</span>',
        '</div>',
        '<div id="pb-ed-wrap" style="position:relative;flex:1;width:100%;max-width:1100px;',
        'max-height:calc(100vh - 140px);overflow:hidden;border-radius:10px;cursor:crosshair;">',
        '<canvas id="pb-ed-canvas" style="display:block;width:100%;height:100%;"></canvas>',
        '<div id="pb-ed-lupe" style="position:absolute;pointer-events:none;display:none;',
        'width:100px;height:100px;border-radius:50%;border:3px solid #fff;',
        'box-shadow:0 0 0 2px #000,0 4px 16px rgba(0,0,0,0.6);overflow:hidden;"></div>',
        '</div>',
        '<div id="pb-ed-picked" style="display:' + (multi ? 'flex' : 'none') + ';gap:5px;',
        'margin-top:10px;flex-wrap:wrap;max-width:700px;justify-content:center;"></div>',
        '<div style="display:flex;align-items:center;gap:12px;margin-top:12px;">',
        '<div id="pb-ed-swatch" style="width:36px;height:36px;border-radius:6px;',
        'border:2px solid #555;background:#888;"></div>',
        '<span id="pb-ed-hex" style="color:#fff;font-size:14px;font-weight:600;',
        'font-family:monospace;">#------</span>',
        '<button id="pb-ed-ok" style="padding:8px 20px;background:#6c6;color:#fff;',
        'border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;">' + label + '</button>',
        doneBtnHtml,
        '<button id="pb-ed-cancel" style="padding:8px 16px;background:#555;color:#fff;',
        'border:none;border-radius:6px;font-size:13px;cursor:pointer;">' + (multi ? 'Close' : 'Cancel') + '</button>',
        '</div>',
      ].join('');
      document.body.appendChild(overlay);

      var canvas  = document.getElementById('pb-ed-canvas');
      var wrap    = document.getElementById('pb-ed-wrap');
      var lupe    = document.getElementById('pb-ed-lupe');
      var swatch  = document.getElementById('pb-ed-swatch');
      var hexSpan = document.getElementById('pb-ed-hex');
      var okBtn   = document.getElementById('pb-ed-ok');
      var cancelBtn = document.getElementById('pb-ed-cancel');
      var ctx = canvas.getContext('2d', {willReadFrequently: true});

      // Off-screen canvas holds the full-res image for pixel sampling
      var offscreen = document.createElement('canvas');
      var offCtx = offscreen.getContext('2d', {willReadFrequently: true});

      var img = new Image();
      var tx = 0, ty = 0, sc = 1, dragging = false, lastX = 0, lastY = 0;
      var currentHex = null;
      okBtn.disabled = true;
      okBtn.style.opacity = '0.4';

      function resize() {
        canvas.width  = wrap.clientWidth  || 800;
        canvas.height = wrap.clientHeight || 600;
        draw();
      }

      function fitImage() {
        var sx = canvas.width  / img.naturalWidth;
        var sy = canvas.height / img.naturalHeight;
        sc = Math.min(sx, sy) * 0.97;
        tx = (canvas.width  - img.naturalWidth  * sc) / 2;
        ty = (canvas.height - img.naturalHeight * sc) / 2;
      }

      function draw() {
        if (!img.complete || !img.naturalWidth) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(tx, ty);
        ctx.scale(sc, sc);
        ctx.drawImage(img, 0, 0);
        ctx.restore();
      }

      function canvasToImage(cx, cy) {
        // Convert canvas pixel coords to image pixel coords
        return { x: (cx - tx) / sc, y: (cy - ty) / sc };
      }

      function sampleColor(imgX, imgY) {
        var x = Math.round(Math.max(0, Math.min(img.naturalWidth  - 1, imgX)));
        var y = Math.round(Math.max(0, Math.min(img.naturalHeight - 1, imgY)));
        var px = offCtx.getImageData(x, y, 1, 1).data;
        return '#' + [px[0],px[1],px[2]].map(function(v){
          return v.toString(16).padStart(2,'0');
        }).join('').toUpperCase();
      }

      function updateLupe(e) {
        var rect = canvas.getBoundingClientRect();
        var cx = (e.clientX - rect.left) * (canvas.width  / rect.width);
        var cy = (e.clientY - rect.top)  * (canvas.height / rect.height);
        var im = canvasToImage(cx, cy);
        var hex = sampleColor(im.x, im.y);

        // Position loupe above-right of cursor (keep on screen)
        var lx = e.clientX - wrap.getBoundingClientRect().left + 12;
        var ly = e.clientY - wrap.getBoundingClientRect().top  - 112;
        if (lx + 100 > wrap.clientWidth)  lx = e.clientX - wrap.getBoundingClientRect().left - 112;
        if (ly < 0) ly = e.clientY - wrap.getBoundingClientRect().top + 16;
        lupe.style.left = lx + 'px';
        lupe.style.top  = ly + 'px';
        lupe.style.display = 'block';

        // Draw magnified patch into loupe via background
        var zoom = 6;
        var lupeR = 50;
        lupe.style.background = hex;
        // Draw a zoomed canvas crop into lupe using a second mini-canvas
        var lupeCanvas = lupe._canvas;
        if (!lupeCanvas) {
          lupeCanvas = document.createElement('canvas');
          lupeCanvas.width = 100; lupeCanvas.height = 100;
          lupeCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';
          lupe.appendChild(lupeCanvas);
          lupe._canvas = lupeCanvas;
        }
        var lc = lupeCanvas.getContext('2d');
        lc.save();
        lc.imageSmoothingEnabled = false;
        var srcX = im.x - lupeR/zoom;
        var srcY = im.y - lupeR/zoom;
        var srcW = 100/zoom;
        lc.clearRect(0,0,100,100);
        lc.drawImage(offscreen, srcX, srcY, srcW, srcW, 0, 0, 100, 100);
        // Crosshair
        lc.strokeStyle = 'rgba(255,255,255,0.9)';
        lc.lineWidth = 1;
        lc.beginPath(); lc.moveTo(50,44); lc.lineTo(50,56); lc.stroke();
        lc.beginPath(); lc.moveTo(44,50); lc.lineTo(56,50); lc.stroke();
        lc.restore();

        return hex;
      }

      var firstLoad = true;
      img.onload = function() {
        offscreen.width  = img.naturalWidth;
        offscreen.height = img.naturalHeight;
        offCtx.drawImage(img, 0, 0);
        if (firstLoad) { resize(); fitImage(); firstLoad = false; }  // keep pan/zoom across frames
        draw();
      };
      function loadFrame() {
        img.src = isVideo
          ? ('/api/video_frame?idx=' + frameIdx + '&_=' + Date.now())
          : ('/api/source_image?' + Date.now());
      }
      if (isVideo) {
        var nav = document.getElementById('pb-ed-nav');
        var frameSlider = document.getElementById('pb-ed-frame');
        var frameLbl = document.getElementById('pb-ed-framelbl');
        nav.style.display = 'flex';
        function setFrame(n) {
          frameIdx = Math.max(0, Math.min(frameCount - 1, n | 0));
          frameSlider.value = frameIdx;
          frameLbl.textContent = 'frame ' + frameIdx + ' / ' + (frameCount - 1);
          loadFrame();
        }
        document.getElementById('pb-ed-prev').onclick    = function() { setFrame(frameIdx - 1); };
        document.getElementById('pb-ed-next').onclick    = function() { setFrame(frameIdx + 1); };
        document.getElementById('pb-ed-shuffle').onclick = function() { setFrame(Math.floor(Math.random() * frameCount)); };
        frameSlider.oninput = function() { setFrame(parseInt(frameSlider.value)); };
      }
      loadFrame();

      // ── Events ──────────────────────────────────────────────────────────────
      canvas.addEventListener('wheel', function(e) {
        e.preventDefault();
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) * (canvas.width  / rect.width);
        var my = (e.clientY - rect.top)  * (canvas.height / rect.height);
        var factor = e.deltaY < 0 ? 1.12 : 1/1.12;
        tx = mx - (mx - tx) * factor;
        ty = my - (my - ty) * factor;
        sc *= factor;
        draw();
      }, {passive:false});

      canvas.addEventListener('mousedown', function(e) {
        if (e.button !== 0) return;
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.style.cursor = 'grabbing';
        e.preventDefault();
      });
      window.addEventListener('mousemove', function(e) {
        if (dragging) {
          var rect = canvas.getBoundingClientRect();
          var ratio = canvas.width / rect.width;
          tx += (e.clientX - lastX) * ratio;
          ty += (e.clientY - lastY) * ratio;
          lastX = e.clientX; lastY = e.clientY;
          draw();
        }
        // Update loupe on every move
        if (overlay.parentNode) updateLupe(e);
      });
      window.addEventListener('mouseup', function(e) {
        dragging = false;
        if (canvas) canvas.style.cursor = 'crosshair';
      });

      canvas.addEventListener('click', function(e) {
        if (dragging) return;
        var hex = updateLupe(e);
        currentHex = hex;
        swatch.style.background = hex;
        hexSpan.textContent = hex;
        okBtn.disabled = false;
        okBtn.style.opacity = '1';
        okBtn.style.background = hex;
        // Contrast text color
        var r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
        var lum = 0.299*r + 0.587*g + 0.114*b;
        okBtn.style.color = lum > 140 ? '#000' : '#fff';
      });

      canvas.addEventListener('mouseleave', function() { lupe.style.display='none'; });
      new ResizeObserver(resize).observe(wrap);

      // ── Buttons ─────────────────────────────────────────────────────────────
      var pickedColors = [];
      var pickedRow = document.getElementById('pb-ed-picked');
      var doneBtn = multi ? document.getElementById('pb-ed-done') : null;
      var countSpan = multi ? document.getElementById('pb-ed-count') : null;

      function addPicked(hex) {
        if (pickedColors.indexOf(hex) !== -1) return;
        pickedColors.push(hex);
        var sw = document.createElement('div');
        sw.style.cssText = 'width:22px;height:22px;border-radius:4px;'
                          + 'border:1px solid rgba(255,255,255,0.5);background:' + hex + ';';
        sw.title = hex;
        pickedRow.appendChild(sw);
        countSpan.textContent = pickedColors.length;
      }

      function done(color) {
        window.removeEventListener('keydown', onKey);
        document.body.removeChild(overlay);
        resolve(color);
      }
      okBtn.addEventListener('click', function() {
        if (!currentHex) return;
        if (multi) {
          addPicked(currentHex);
          okBtn.disabled = true;
          okBtn.style.opacity = '0.4';
        } else {
          done(currentHex);
        }
      });
      if (multi) {
        doneBtn.addEventListener('click', function() { done(pickedColors); });
      }
      cancelBtn.addEventListener('click', function() { done(multi ? pickedColors : null); });
      function onKey(e) {
        if (e.key === 'Escape') { done(multi ? pickedColors : null); return; }
        if (multi && e.key === 'Enter' && currentHex && !okBtn.disabled) {
          addPicked(currentHex);
          okBtn.disabled = true;
          okBtn.style.opacity = '0.4';
        }
      }
      window.addEventListener('keydown', onKey);
    });
  };
})();
</script>
'''

VIDEO_PLAYER_JS = '''
<script>
(function() {
  // Frames live on the SERVER; the browser fetches one at a time (bounded LRU cache),
  // so playback/scrubbing scale to any clip length without the tab running out of memory.
  var S = { count: 0, meta: null, i: 0, timer: null, fps: 12, userInteracted: false,
            renderMode: 'circle', hidden: {}, cache: {}, order: [], busy: false };
  window.__smv = S;
  var CACHE_MAX = 80;

  async function fetchFrame(i) {
    if (S.cache[i]) return S.cache[i];
    var r = await fetch('/api/video_scrub?idx=' + i);
    if (!r.ok) return null;
    var d = await r.json();
    if (d.meta) S.meta = d.meta;
    if (d.count) S.count = d.count;
    S.cache[i] = d.frame; S.order.push(i);
    while (S.order.length > CACHE_MAX) { delete S.cache[S.order.shift()]; }
    return d.frame;
  }

  function drawCircles(f) {
    var img = document.getElementById('smv-render');
    var canv = document.getElementById('pb-canvas');
    if (img) img.style.display = 'none';
    if (canv) canv.style.display = 'block';
    var m = S.meta; if (!m) return;
    var xs = f.x, ys = f.y, rs = f.r, cs = f.c;
    if (Object.keys(S.hidden).length) {
      xs = []; ys = []; rs = []; cs = [];
      for (var k = 0; k < f.c.length; k++) {
        if (!S.hidden[f.c[k]]) { xs.push(f.x[k]); ys.push(f.y[k]); rs.push(f.r[k]); cs.push(f.c[k]); }
      }
    }
    if (window.updatePaintbot) {
      window.updatePaintbot({
        xs: xs, ys: ys, rs: rs, colors: cs,
        palette: m.palette, canvas_hex: m.canvas,
        W: m.working_size[0], H: m.working_size[1],
        mode: 'circle', phase: 'scrub', r_min: m.r_min, r_max: m.r_max,
        neighbors: null, polygons: null
      });
    }
  }

  async function render(i) {
    if (!S.count) return;
    i = Math.max(0, Math.min(S.count - 1, i));
    S.i = i;
    var lbl = document.getElementById('smv-lbl');
    if (lbl) lbl.textContent = 'frame ' + (i + 1) + ' / ' + S.count;
    if (S.renderMode === 'polygon' || S.renderMode === 'airbrush' || S.renderMode === 'tile') {
      var img = document.getElementById('smv-render');
      var canv = document.getElementById('pb-canvas');
      if (img) {
        var reqMode = S.renderMode, reqIdx = i;
        img.onload = function() { img.style.display = 'block'; if (canv) canv.style.display = 'none'; };
        // Without this, a failed request (500/404) left the <img> untouched --
        // the display just silently kept showing whatever rendered last
        // (typically stale circles), with no indication anything went wrong.
        img.onerror = function() {
          console.error('[video_render_frame] failed for mode=' + reqMode + ' idx=' + reqIdx);
          var lbl2 = document.getElementById('smv-lbl');
          if (lbl2) lbl2.textContent = 'frame ' + (reqIdx + 1) + ' / ' + S.count + '  (render failed — see console)';
        };
        img.src = '/api/video_render_frame?idx=' + i + '&mode=' + S.renderMode + '&_=' + Date.now();
      }
      return;
    }
    var f = await fetchFrame(i);
    if (f) drawCircles(f);
  }

  window.smaltiSetHidden = function(arr) {
    S.hidden = {};
    (arr || []).forEach(function(c) { S.hidden[c] = true; });
    render(S.i);
  };
  window.smaltiSetRenderMode = function(m) { S.renderMode = m; render(S.i); };

  var _refreshT = null;
  window.smaltiRefresh = function() {
    if (!S.count) return;
    if (S.renderMode !== 'polygon' && S.renderMode !== 'airbrush' && S.renderMode !== 'tile') return;
    clearTimeout(_refreshT);
    _refreshT = setTimeout(function() { render(S.i); }, 300);
  };

  window.smaltiReset = function() {
    window.smaltiStop(); S.count = 0; S.i = 0; S.userInteracted = false;
    S.cache = {}; S.order = []; S.meta = null;
  };
  // Called (coalesced) by the server as frames finish — updates the count and, unless the
  // user is scrubbing, shows the newest frame.  Fetches only that one frame.
  window.smaltiSetCount = function(n) {
    S.count = n;
    var lbl = document.getElementById('smv-lbl');
    if (lbl && S.userInteracted) lbl.textContent = 'frame ' + (S.i + 1) + ' / ' + S.count;
    if (!S.userInteracted) render(n - 1);
  };
  window.smaltiShowFrame = function(i) { S.userInteracted = true; window.smaltiStop(); render(i | 0); };
  window.smaltiScrub = function(frac) {
    S.userInteracted = true; window.smaltiStop();
    render(Math.round(frac * Math.max(0, S.count - 1)));
  };
  window.smaltiNext = function() { S.userInteracted = true; window.smaltiStop(); render((S.i + 1) % S.count); };
  window.smaltiPrev = function() { S.userInteracted = true; window.smaltiStop(); render((S.i - 1 + S.count) % S.count); };
  window.smaltiPlay = function(fps) {
    if (S.timer || !S.count) return;
    S.userInteracted = true;
    S.fps = fps || S.fps || 12;
    S.timer = setInterval(async function() {
      if (S.busy) return;                 // skip a tick rather than pile up fetches
      S.busy = true;
      await render((S.i + 1) % S.count);
      S.busy = false;
    }, 1000 / S.fps);
  };
  window.smaltiStop = function() { if (S.timer) { clearInterval(S.timer); S.timer = null; } };
  window.smaltiIsPlaying = function() { return !!S.timer; };
})();
</script>
'''

@ui.page('/')
async def index(client: Client):
    # NOTE: a job in progress (S.processing / S.worker_thread) is deliberately left
    # alone here. This page function reruns on every reconnect — including the
    # native WebView briefly reloading while backgrounded — and that reconnect IS
    # the user coming back, not the tab closing forever. Cancelling here would race
    # the "Reattach to an in-progress job" logic further down and always win,
    # silently killing any generate/process/export that outlives a single
    # connection. A genuinely abandoned job (tab closed for good) simply finishes
    # unobserved; its durable result/error sits in S until someone reconnects.

    ui.add_head_html(STYLE)
    # Vendored PixiJS (served from /assets) so the app runs fully offline; fall back
    # to the CDN only if the local copy is missing (e.g. a source checkout without it).
    if (_ASSETS_DIR / 'pixi.min.js').exists():
        ui.add_head_html('<script src="/assets/pixi.min.js"></script>')
    else:
        ui.add_head_html('<script src="https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.min.js"></script>')
    ui.add_body_html(CANVAS_JS)
    ui.add_body_html(EYEDROPPER_JS)
    ui.add_body_html(VIDEO_PLAYER_JS)

    # Shared reactive refs (use lists so closures can reassign)
    canvas_dot_el = [None]
    canvas_hex_label_el = [None]
    palette_container_el = [None]
    progress_fill_el = [None]
    status_el = [None]
    results_el = [None]
    unit_btns: dict[str, object] = {}

    # ── UI text-size control ─────────────────────────────────────────────────
    # A user on a 4K monitor found the default text too small. This
    # scales the whole app via CSS `zoom` on the app-shell (a sibling of this
    # control, so the control itself stays fixed-size) and remembers the
    # chosen level per-machine in localStorage.
    _ui_zoom_pct = [100]
    with ui.element('div').classes('ui-zoom-ctrl').props('id=ui-zoom-ctrl'):
        ui.html('<span style="opacity:0.75;">Text size</span>')
        zoom_minus_btn = ui.element('div').classes('zoom-btn')
        with zoom_minus_btn:
            ui.html('−')
        zoom_pct_label = ui.html('<span id="ui-zoom-pct">100%</span>')
        zoom_plus_btn = ui.element('div').classes('zoom-btn')
        with zoom_plus_btn:
            ui.html('+')

    async def _apply_zoom(pct: int):
        pct = max(70, min(160, pct))
        _ui_zoom_pct[0] = pct
        zoom_pct_label.set_content(f'<span id="ui-zoom-pct">{pct}%</span>')
        await client.run_javascript(f"""
            var el = document.getElementById('app-shell-root');
            if (el) el.style.zoom = {pct} / 100;
            try {{ localStorage.setItem('smalti_ui_zoom', '{pct}'); }} catch(e) {{}}
        """)

    async def _zoom_minus():
        await _apply_zoom(_ui_zoom_pct[0] - 10)

    async def _zoom_plus():
        await _apply_zoom(_ui_zoom_pct[0] + 10)

    zoom_minus_btn.on('click', _zoom_minus)
    zoom_plus_btn.on('click', _zoom_plus)

    # ── HEADER ────────────────────────────────────────────────────────────────
    # The native Smalti.app already shows the name in its window title bar + Dock,
    # so the in-page header is only shown in the web/browser version.
    with ui.element('div').classes('app-shell' + ('' if not _APP_MODE else ' no-header')).props('id=app-shell-root'):
        if not _APP_MODE:
            with ui.element('div').classes('app-header'):
                ui.html(f'<div style="width:38px;height:38px;flex-shrink:0;">{LOGO_MARK_SVG}</div>')
                with ui.element('div'):
                    ui.html('<div style="font-size:18px;font-weight:700;color:#E8EAF0;'
                            'letter-spacing:-0.01em;">Smalti</div>')
                    ui.html('<div style="font-size:12px;color:#717898;">Photo mosaics for makers</div>')

        # Direct grid children — created up front so they exist as `.app-shell`
        # grid items; cards built deeper in the script are relocated into them
        # via .move() once everything below is defined (grid-area placement is
        # independent of DOM order, but .move() needs both elements to exist).
        algorithm_el = ui.element('div').classes('app-algorithm')

        # ── SIDEBAR ───────────────────────────────────────────────────────────
        with ui.element('div').classes('app-sidebar'):

            # ── Media Upload (image or video) ────────────────────────────────
            _media_refresh = {'fn': lambda: None}   # set once run-btn/video-opts exist
            _vct_refresh = {'fn': lambda: None}     # refresh video colour toggles
            _reswork_refresh = {'fn': lambda: None}  # refresh Working-resolution buttons (Algorithm card)
            still_only_els = []                     # controls hidden while in video mode
            video_only_els = []                     # controls shown only while in video mode

            def _set_source_native(path: str, is_video: bool):
                """Record the loaded source's native resolution and re-derive
                max_working_px from the currently selected Full/½/¼/⅛ fraction."""
                w = h = 0
                try:
                    if is_video:
                        import cv2
                        cap = cv2.VideoCapture(path)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                    else:
                        from PIL import Image as _PILImage
                        with _PILImage.open(path) as im:
                            w, h = im.size
                except Exception:
                    w = h = 0
                S.source_native_long = max(w, h) if (w and h) else 0
                if S.source_native_long:
                    S.max_working_px = max(200, round(S.source_native_long * S.working_res_fraction))
                _reswork_refresh['fn']()

            with ui.element('div').classes('card'):
                ui.html('<div class="card-title">1. Source Media</div>')

                upload_zone_el = ui.element('div').classes('upload-zone').props('id=pb-upload-zone')
                with upload_zone_el:
                    upload_label = ui.html(
                        '<div>Drop an image or video here<br>'
                        '<span style="font-size:11px;opacity:0.6;">'
                        'or click here to select an image or video</span></div>'
                    )

                _VIDEO_EXTS = ('.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.mpg', '.mpeg', '.wmv')

                async def handle_upload(e):
                    try:
                        f = e.file                          # FileUpload object (NiceGUI 3.x)
                        suffix = Path(f.name).suffix or '.jpg'
                        content = await f.read()            # async read
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp.write(content)
                        tmp.close()
                        is_video = suffix.lower() in _VIDEO_EXTS
                        if is_video:
                            S.video_path = tmp.name; S.video_name = f.name; S.video_json_path = ''
                            S.media_mode = 'video'
                            _set_source_native(tmp.name, is_video=True)
                            if S.unit != 'px':
                                retune_units('px')
                                for k, b in unit_btns.items():
                                    b.classes(add='active') if k == 'px' else b.classes(remove='active')
                            upload_zone_el.classes(add='upload-done')
                            upload_label.set_content(
                                f'<div style="font-weight:600;">🎬 {f.name}</div>'
                                f'<div style="font-size:11px;opacity:0.6;">Video · {len(content)//1024//1024} MB — video mode</div>'
                            )
                            status_el[0].set_content(
                                '<div style="font-size:13px;color:var(--sub);">Video ready. Adjust settings, '
                                'then click <b>Process video</b>.</div>'
                            )
                        else:
                            S.image_path = tmp.name
                            S.image_name = f.name
                            S.media_mode = 'image'
                            _set_source_native(tmp.name, is_video=False)
                            upload_zone_el.classes(add='upload-done')
                            upload_label.set_content(
                                f'<div style="font-weight:600;">✓ {f.name}</div>'
                                f'<div style="font-size:11px;opacity:0.6;">Ready — {len(content)//1024} KB</div>'
                            )
                            status_el[0].set_content(
                                '<div style="font-size:13px;color:var(--sub);">Image ready. Click Generate to process.</div>'
                            )
                        _media_refresh['fn']()
                    except Exception as ex:
                        import traceback
                        status_el[0].set_content(
                            f'<div style="font-size:13px;color:#FA6E9E;">Upload error: {ex}</div>'
                        )

                # Kept only for drag-and-drop (the drop-zone JS feeds files to this hidden
                # input). Its visible chrome is hidden — browsing is done via the native
                # "Choose image or video…" picker below, so no redundant widget. Large
                # videos dropped here still go through the browser's upload path (unlike
                # the native picker), so very large files may be slow or fail — the native
                # picker remains the reliable path for big video files.
                uploader = ui.upload(on_upload=handle_upload, auto_upload=True).props(
                    'accept=".tif,.tiff,.jpg,.jpeg,.png,.heic,.heif,'
                    + ','.join(_VIDEO_EXTS) + '" flat'
                ).style('position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;')

                # Native file picker — handles BOTH images and large local videos (which the
                # browser uploader can't take).  Routes by extension into image / video mode.
                async def pick_media():
                    def _pick():
                        import subprocess, sys
                        start = str(Path(S.video_path or S.image_path).parent) \
                            if (S.video_path or S.image_path) else str(Path.home())
                        if sys.platform == 'darwin':
                            q = '"' + start.replace('\\', '\\\\').replace('"', '\\"') + '"'
                            script = ('set f to choose file with prompt "Choose an image or video"'
                                      f' default location POSIX file {q}\nPOSIX path of f')
                            try:
                                r = subprocess.run(['osascript', '-e', script],
                                                   capture_output=True, text=True, timeout=600)
                                return r.stdout.strip() or None
                            except Exception:
                                return None
                        try:
                            r = subprocess.run(
                                [sys.executable, '-c',
                                 'import tkinter,tkinter.filedialog as fd;'
                                 't=tkinter.Tk();t.withdraw();print(fd.askopenfilename() or "")'],
                                capture_output=True, text=True, timeout=600)
                            return r.stdout.strip() or None
                        except Exception:
                            return None
                    path = await asyncio.get_event_loop().run_in_executor(None, _pick)
                    if not path or not os.path.exists(path):
                        return
                    name = os.path.basename(path)
                    if os.path.splitext(path)[1].lower() in _VIDEO_EXTS:
                        S.video_path = path; S.video_name = name; S.video_json_path = ''
                        S.media_mode = 'video'
                        _set_source_native(path, is_video=True)
                        if S.unit != 'px':
                            retune_units('px')
                            for k, b in unit_btns.items():
                                b.classes(add='active') if k == 'px' else b.classes(remove='active')
                        sz = os.path.getsize(path) // (1024 * 1024)
                        upload_zone_el.classes(add='upload-done')
                        upload_label.set_content(
                            f'<div style="font-weight:600;">🎬 {name}</div>'
                            f'<div style="font-size:11px;opacity:0.6;">Video · {sz} MB — video mode</div>')
                        status_el[0].set_content(
                            '<div style="font-size:13px;color:var(--sub);">Video ready. Adjust settings, '
                            'then click <b>Process video</b>.</div>')
                    else:
                        S.image_path = path; S.image_name = name; S.media_mode = 'image'
                        _set_source_native(path, is_video=False)
                        upload_zone_el.classes(add='upload-done')
                        upload_label.set_content(
                            f'<div style="font-weight:600;">✓ {name}</div>'
                            f'<div style="font-size:11px;opacity:0.6;">Image — image mode</div>')
                        status_el[0].set_content(
                            '<div style="font-size:13px;color:var(--sub);">Image ready. Click Generate.</div>')
                    _media_refresh['fn']()

                # The drop zone is the sole click/drag target for the native picker —
                # no separate button needed.
                upload_zone_el.on('click', pick_media)

                # Wire drag-and-drop on the visual zone to the hidden Quasar file input.
                # Without this, dropping a file onto the page navigates the browser to it.
                #
                # Vue renders the DOM asynchronously — by the time this script runs, the
                # elements NiceGUI just told the browser about may not be mounted yet, so
                # a plain one-shot getElementById can silently find nothing and attach no
                # listeners at all (drag-and-drop then does nothing, with no error visible
                # anywhere). Retry on a short interval instead of assuming the DOM is ready.
                await client.run_javascript(f'''
                (function() {{
                  var attempts = 0;
                  function wire() {{
                    attempts++;
                    var zone = document.getElementById('pb-upload-zone');
                    var uploader = document.getElementById('c{uploader.id}');
                    if (!zone || !uploader) {{
                      if (attempts < 40) setTimeout(wire, 100);   // keep trying for ~4s
                      return;
                    }}
                    attachListeners(zone, uploader);
                  }}
                  function attachListeners(zone, uploader) {{
                  var fileInput = uploader.querySelector('input[type=file]');

                  // Prevent browser from opening the file on any drag over the page
                  ['dragenter','dragover','dragleave','drop'].forEach(function(evt) {{
                    document.body.addEventListener(evt, function(e) {{ e.preventDefault(); }}, false);
                  }});

                  zone.addEventListener('dragenter', function(e) {{
                    e.preventDefault();
                    zone.style.borderColor = 'var(--accent)';
                    zone.style.background  = 'rgba(124,110,250,0.12)';
                  }});
                  zone.addEventListener('dragleave', function(e) {{
                    zone.style.borderColor = '';
                    zone.style.background  = '';
                  }});
                  zone.addEventListener('dragover', function(e) {{ e.preventDefault(); }});
                  zone.addEventListener('drop', function(e) {{
                    e.preventDefault();
                    zone.style.borderColor = '';
                    zone.style.background  = '';
                    if (!fileInput || !e.dataTransfer.files.length) return;
                    // Assign the dropped files to the hidden input and fire change
                    var dt = new DataTransfer();
                    dt.items.add(e.dataTransfer.files[0]);
                    fileInput.files = dt.files;
                    fileInput.dispatchEvent(new Event('change', {{bubbles: true}}));
                  }});
                  }}
                  wire();
                }})();
                ''')

            # Restore a per-machine text-size preference saved in a previous session.
            try:
                _stored_zoom = await client.run_javascript(
                    "localStorage.getItem('smalti_ui_zoom') || '100'")
                await _apply_zoom(int(_stored_zoom))
            except Exception:
                pass

            # Only reveal the text-size control once the user has scrolled to the
            # bottom — of the page itself, or of one of the app's three columns
            # (each has its own overflow:auto, and depending on content/viewport
            # size either the page or a column ends up being the actual scroller)
            # — keeps it out of the way otherwise instead of floating permanently.
            await client.run_javascript("""
            (function() {
              var ctrl = document.getElementById('ui-zoom-ctrl');
              if (!ctrl) return;
              var cols = Array.prototype.slice.call(
                document.querySelectorAll('.app-sidebar, .app-algorithm, .app-main'));
              var NEAR_BOTTOM_PX = 24;
              function atBottom(el) {
                var scrollable = el.scrollHeight - el.clientHeight > NEAR_BOTTOM_PX;
                return scrollable && (el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_PX);
              }
              function pageAtBottom() {
                var doc = document.documentElement;
                var scrollable = doc.scrollHeight - window.innerHeight > NEAR_BOTTOM_PX;
                return scrollable &&
                  (doc.scrollHeight - window.scrollY - window.innerHeight <= NEAR_BOTTOM_PX);
              }
              function refresh() {
                var show = pageAtBottom() || cols.some(atBottom);
                ctrl.classList.toggle('visible', show);
              }
              cols.forEach(function(el) {
                el.addEventListener('scroll', refresh, { passive: true });
              });
              window.addEventListener('scroll', refresh, { passive: true });
              window.addEventListener('resize', refresh);
              refresh();
            })();
            """)

            # ── Canvas Color ─────────────────────────────────────────────────
            with ui.element('div').classes('card'):
                ui.html('<div class="card-title">2. Canvas / Surface Color</div>')
                with ui.element('div').style('display:flex;align-items:center;gap:12px;margin-bottom:12px;'):
                    canvas_dot_el[0] = ui.element('div').classes('canvas-dot').style(
                        f'background:{S.canvas_hex};'
                    )
                    with ui.element('div').style('flex:1;'):
                        ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:2px;">Current surface</div>')
                        canvas_hex_label_el[0] = ui.html(
                            f'<div style="font-size:13px;font-weight:600;">{S.canvas_hex}</div>'
                        )
                    async def pick_canvas_color():
                        is_video = (S.media_mode == 'video')
                        if not (S.image_path or (is_video and S.video_path)):
                            return
                        if is_video:
                            js = ('return window.paintbotPickColor("canvas", '
                                  f'{{video:true, frames:{_video_frame_count()}}})')
                        else:
                            js = 'return window.paintbotPickColor("canvas")'
                        color = await client.run_javascript(js, timeout=300.0)
                        if color and color.startswith('#'):
                            S.canvas_hex = color
                            canvas_dot_el[0].style(f'background:{color};')
                            canvas_hex_label_el[0].set_content(
                                f'<div style="font-size:13px;font-weight:600;">{color}</div>'
                            )
                    ui.button().props('icon=colorize').style(
                        'font-size:13px;width:32px;height:32px;min-width:32px;padding:0;'
                        'background:var(--accent) !important;'
                        'border:1px solid var(--accent);border-radius:6px;color:white;'
                        'cursor:pointer;'
                    ).on('click', pick_canvas_color).tooltip('Pick color from image')

                with ui.element('div').style('display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;'):
                    for lbl, hx in CANVAS_PRESETS.items():
                        def make_canvas_setter(h, l):
                            def setter():
                                S.canvas_hex = h
                                canvas_dot_el[0].style(f'background:{h};')
                                canvas_hex_label_el[0].set_content(
                                    f'<div style="font-size:13px;font-weight:600;">{h}</div>'
                                )
                            return setter
                        ui.button(lbl).on('click', make_canvas_setter(hx, lbl)).style(
                            'font-size:11px;padding:4px 9px;background:var(--accent) !important;'
                            'border:1px solid var(--accent);border-radius:6px;color:white;'
                        )

                custom_canvas = ui.input('Custom hex e.g. #A07850').props('dense outlined dark').style(
                    'font-size:12px;width:100%;'
                )
                def apply_custom_canvas():
                    h = custom_canvas.value.strip()
                    if len(h) in (4, 7) and h.startswith('#'):
                        S.canvas_hex = h
                        canvas_dot_el[0].style(f'background:{h};')
                        canvas_hex_label_el[0].set_content(
                            f'<div style="font-size:13px;font-weight:600;">{h}</div>'
                        )
                        # The "Current surface" row above now shows the applied color; clearing
                        # the box also avoids the field's label and typed text drawing on top
                        # of each other (the label doesn't float for a value typed then applied).
                        custom_canvas.set_value('')
                custom_canvas.on('keydown.enter', apply_custom_canvas)
                ui.html('<div style="font-size:11px;color:var(--sub);margin-top:6px;">'
                        'Canvas color shifts the color balance of all layers — '
                        'the engine compensates so circles appear correctly on this background.</div>')

            # ── Paint Palette ─────────────────────────────────────────────────
            with ui.element('div').classes('card'):
                with ui.element('div').style('display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;'):
                    ui.html('<div class="card-title" style="margin-bottom:0;">3. Color Palette  <span style="color:var(--sub2);font-weight:400;">(up to 100 colors)</span></div>')
                    async def pick_palette_color():
                        is_video = (S.media_mode == 'video')
                        if not (S.image_path or (is_video and S.video_path)):
                            return
                        if len(S.palette) >= 100:
                            return
                        if is_video:
                            js = ('return window.paintbotPickColor("palette", '
                                  f'{{video:true, frames:{_video_frame_count()}, multi:true}})')
                        else:
                            js = 'return window.paintbotPickColor("palette", {multi:true})'
                        colors = await client.run_javascript(js, timeout=600.0)
                        added = False
                        for color in (colors or []):
                            if len(S.palette) >= 100:
                                break
                            if color and color.startswith('#') and color not in S.palette:
                                S.palette.append(color)
                                added = True
                        if added:
                            refresh_palette()
                    ui.button().props('icon=colorize').style(
                        'font-size:13px;width:32px;height:32px;min-width:32px;padding:0;'
                        'background:var(--accent) !important;'
                        'border:1px solid var(--accent);border-radius:6px;color:white;'
                        'cursor:pointer;'
                    ).on('click', pick_palette_color).tooltip('Pick color from image')

                def refresh_palette():
                    palette_container_el[0].clear()
                    with palette_container_el[0]:
                        for i, h in enumerate(S.palette):
                            def make_remove(idx):
                                def remove():
                                    if len(S.palette) > 1:
                                        S.palette.pop(idx)
                                        refresh_palette()
                                return remove
                            with ui.element('div').classes('chip').on('click', make_remove(i)):
                                ui.element('div').classes('swatch').style(f'background:{h};')
                                ui.html(f'<span>{h}</span>')
                                ui.html('<span class="chip-x">×</span>')
                    _vct_refresh['fn']()   # keep video colour toggles in sync

                palette_container_el[0] = ui.element('div').style(
                    'display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;'
                )
                refresh_palette()

                with ui.element('div').style('display:flex;gap:7px;align-items:center;margin-bottom:10px;'):
                    new_color = ui.input(placeholder='#RRGGBB').props('dense outlined dark').style(
                        'flex:1;font-size:12px;'
                    )
                    def add_color():
                        h = new_color.value.strip()
                        if h.startswith('#') and len(S.palette) < 100:
                            S.palette.append(h)
                            new_color.set_value('')
                            refresh_palette()
                    ui.button('+').on('click', add_color).style(
                        'background:var(--accent) !important;color:white;border-radius:6px;'
                        'padding:0 14px;font-weight:700;font-size:16px;'
                    )

                ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:7px;">Auto-generate from image</div>')
                with ui.element('div').style('display:flex;gap:7px;align-items:center;margin-bottom:10px;'):
                    auto_n = ui.number(value=6, min=2, max=100, step=1).props('dense outlined dark').style(
                        'width:70px;font-size:12px;'
                    )
                    auto_btn = ui.button('Generate palette').classes('btn-sm').style(
                        'background:var(--accent) !important;color:white;'
                    )
                    auto_status = ui.html('').style('font-size:11px;color:var(--sub);')

                    async def auto_generate_palette():
                        is_video = (S.media_mode == 'video')
                        if not (S.image_path or (is_video and S.video_path)):
                            auto_status.set_content('<span style="color:#e55;">Choose an image or video first.</span>')
                            return
                        auto_btn.props('loading')
                        auto_status.set_content('Sampling clip…' if is_video else 'Analyzing image…')
                        n = int(auto_n.value or 6)
                        import asyncio

                        def _gen_video():
                            import cv2
                            from core.video_mosaic import build_clip_palette
                            cap = cv2.VideoCapture(S.video_path)
                            try:
                                return build_clip_palette(cap, n)
                            finally:
                                cap.release()

                        try:
                            if is_video:
                                colors = await asyncio.get_event_loop().run_in_executor(None, _gen_video)
                                src = 'clip'
                            else:
                                from core.palette_gen import generate_palette
                                colors = await asyncio.get_event_loop().run_in_executor(
                                    None, generate_palette, S.image_path, n
                                )
                                src = 'image'
                            S.palette.clear()
                            S.palette.extend(colors)
                            refresh_palette()
                            auto_status.set_content(f'<span style="color:#6c6;">Done — {len(colors)} colors from {src}.</span>')
                        except Exception as e:
                            auto_status.set_content(f'<span style="color:#e55;">Error: {e}</span>')
                        finally:
                            auto_btn.props(remove='loading')

                    auto_btn.on('click', auto_generate_palette)

                ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:7px;">Presets</div>')
                with ui.element('div').style('display:flex;flex-wrap:wrap;gap:4px;'):
                    for pname, pcols in PALETTE_PRESETS.items():
                        def make_preset(cols):
                            def setter():
                                S.palette.clear()
                                S.palette.extend(cols)
                                refresh_palette()
                            return setter
                        ui.button(pname).on('click', make_preset(pcols)).classes('btn-sm').style(
                            'background:var(--accent) !important;border-color:var(--accent) !important;'
                            'color:white !important;'
                        )

            # ── Video options (only visible once a video is loaded) ────────────
            video_opts_card = ui.element('div').classes('card')
            video_opts_card.set_visibility(S.media_mode == 'video')   # hidden by default
            with video_opts_card:
                ui.html('<div class="card-title">4. Video Options</div>')
                ui.html('<div style="font-size:10px;color:var(--sub2);margin-bottom:8px;">'
                        'Reuses your main settings — output size = <b>Output long edge</b> (px), '
                        'dot sizes = <b>Min/Max diameter</b>, colours = <b>Palette</b>. The controls '
                        'below are video-only.</div>')

                def _vnum(label, field, mn, mx, step, hint=''):
                    with ui.element('div').classes('param-row').style('margin-bottom:2px;'):
                        ui.html(f'<span class="param-label">{label}</span>')
                        n = ui.number(value=getattr(S, field), min=mn, max=mx, step=step).style(
                            'width:72px;').props('dense outlined dark precision=0 '
                            'input-style="font-size:12px;text-align:right;color:var(--text);"')
                        n.on('update:model-value',
                             lambda e, fld=field, lo=mn, hi=mx:
                                 setattr(S, fld, int(max(lo, min(hi, float(e.args if e.args not in (None,'') else getattr(S, fld)))))))
                    if hint:
                        ui.html(f'<div style="font-size:10px;color:var(--sub2);padding-left:2px;margin-bottom:8px;">{hint}</div>')

                with ui.element('div').style('margin-bottom:8px;'):
                    fresh_cb = ui.checkbox('Fresh dots each frame', value=S.video_fresh).props('dense dark color=deep-purple')
                    fresh_cb.on('update:model-value',
                                lambda e: setattr(S, 'video_fresh',
                                                  bool(e.args if not isinstance(e.args, (list, tuple)) else e.args[0])))
                    ui.html('<div style="font-size:10px;color:var(--sub2);padding-left:2px;">'
                            'ON = each frame is optimised independently — dots always match that '
                            'frame’s detail (more flicker, slower). OFF = warm-start from the '
                            'previous frame (smoother, but arrangements can persist between frames).</div>')

                _vnum('Iterations · per frame', 'video_iters_warm', 1, 20, 1,
                      'Optimization rounds each frame gets. More = cleaner dots, slower. '
                      '(The "Iterations" slider in the Algorithm card is for still images.)')
                _vnum('Frame stride', 'video_stride', 1, 30, 1,
                      'Process every Nth frame (2 halves the work & the output fps).')
                _vnum('Max frames (0 = all)', 'video_max_frames', 0, 100000, 10,
                      'Cap the number of frames — set small to preview the look first.')

            # ── Output Units ──────────────────────────────────────────────────
            with ui.element('div').classes('card') as units_output_card_el:
                output_size_title = ui.html('<div class="card-title">5. Output Size</div>')

                with ui.element('div').style('display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:12px;'):
                    ui.html('<span style="font-size:12px;color:var(--text);">Units</span>')
                    with ui.element('div').classes('seg-ctrl').style('flex-shrink:0;') as seg:
                        for uid, ulabel in UNIT_OPTIONS:
                            is_active = uid == S.unit
                            def make_unit_setter(u):
                                def setter():
                                    retune_units(u)   # converts sizes + re-ranges sliders
                                    for k, b in unit_btns.items():
                                        if k == u:
                                            b.classes(add='active')
                                        else:
                                            b.classes(remove='active')
                                return setter
                            btn = ui.element('button').classes(
                                f'seg-btn{"  active" if is_active else ""}'
                            ).on('click', make_unit_setter(uid))
                            with btn:
                                ui.html(ulabel)
                            unit_btns[uid] = btn

                size_widgets = {}   # attr -> (slider, number)  for unit re-ranging

                def labeled_phys(label, attr, mn, mx, step, fmt='{:.2f}', hint='', show_unit=False):
                    num_el = [None]
                    with ui.element('div').style('margin-bottom:10px;'):
                        with ui.element('div').classes('param-row').style('margin-bottom:2px;'):
                            ui.html(f'<span class="param-label">{label}</span>')
                            sl = ui.slider(min=mn, max=mx, value=getattr(S, attr), step=step).style('flex:1;').props('dark color=deep-purple')
                            num_el[0] = ui.number(value=getattr(S, attr), min=mn, max=mx, step=step).style(
                                'width:64px;'
                            ).props('dense outlined dark input-style="font-size:12px;text-align:right;color:var(--text);"')
                            if show_unit:
                                ui.html(f'<span style="font-size:11px;color:var(--sub2);min-width:20px;">{S.unit}</span>')
                        if hint:
                            ui.html(f'<div style="font-size:10px;color:var(--sub2);line-height:1.4;padding-left:2px;">{hint}</div>')
                    def on_slider(e):
                        v = float(e.args)
                        setattr(S, attr, v)
                        num_el[0].set_value(v)
                    def on_number(e):
                        lo = sl._props.get('min', mn); hi = sl._props.get('max', mx)
                        try:
                            v = float(e.args)
                        except (TypeError, ValueError):
                            return
                        v = max(lo, min(hi, v))
                        setattr(S, attr, v)
                        sl.set_value(v)
                    sl.on('update:model-value', on_slider)
                    num_el[0].on('update:model-value', on_number)
                    if attr in PHYS_SIZE_RANGES:
                        size_widgets[attr] = (sl, num_el[0])
                    return sl

                def retune_units(new_unit):
                    """Switch units: convert the four size values to the new unit (real
                    physical conversion, incl. to/from px) and rebuild the size fields with
                    unit-appropriate ranges / controls."""
                    old = S.unit
                    if new_unit == old:
                        return
                    factor = _px_per_unit_ui(old, S.dpi) / _px_per_unit_ui(new_unit, S.dpi)
                    ranges = PX_SIZE_RANGES if new_unit == 'px' else PHYS_SIZE_RANGES
                    for attr, (mn, mx, step) in ranges.items():
                        v = max(mn, min(mx, getattr(S, attr) * factor))
                        v = round(v) if step >= 1 else round(v, 3)
                        setattr(S, attr, v)
                    S.unit = new_unit
                    size_fields.refresh()
                    reg_marks_fields.refresh()

                def labeled_int(label, attr, mn, mx, step, hint='', register=None):
                    num_el = [None]
                    with ui.element('div').style('margin-bottom:10px;'):
                        with ui.element('div').classes('param-row').style('margin-bottom:2px;'):
                            ui.html(f'<span class="param-label">{label}</span>')
                            sl = ui.slider(min=mn, max=mx, value=getattr(S, attr), step=step).style('flex:1;').props('dark color=deep-purple')
                            num_el[0] = ui.number(value=getattr(S, attr), min=mn, max=mx, step=step).style(
                                'width:64px;'
                            ).props('dense outlined dark input-style="font-size:12px;text-align:right;color:var(--text);" precision=0')
                        if hint:
                            ui.html(f'<div style="font-size:10px;color:var(--sub2);line-height:1.4;padding-left:2px;">{hint}</div>')
                    def on_slider(e):
                        v = int(e.args)
                        setattr(S, attr, v)
                        num_el[0].set_value(v)
                    def on_number(e):
                        try:
                            v = int(float(e.args))
                        except (TypeError, ValueError):
                            return
                        v = max(mn, min(mx, v))
                        setattr(S, attr, v)
                        sl.set_value(v)
                    sl.on('update:model-value', on_slider)
                    num_el[0].on('update:model-value', on_number)
                    # Registers a reset-to-default closure with the caller's list
                    # (only the Algorithm panel's controls pass one) — labeled_int
                    # is shared with the Output size card, which doesn't.
                    if register is not None:
                        def _reset_int():
                            d = getattr(State, attr)
                            setattr(S, attr, d)
                            sl.set_value(d)
                            num_el[0].set_value(d)
                        register.append(_reset_int)
                    return sl

                @ui.refreshable
                def size_fields():
                    is_px = (S.unit == 'px')
                    R = PX_SIZE_RANGES if is_px else PHYS_SIZE_RANGES
                    if is_px:
                        # A fine slider makes no sense at pixel scale — use an exact number
                        # plus standard-resolution presets instead.
                        mn, mx, _ = R['physical_long_edge']
                        with ui.element('div').style('margin-bottom:10px;'):
                            with ui.element('div').classes('param-row').style('margin-bottom:4px;'):
                                ui.html('<span class="param-label">Output long edge</span>')
                                out_num = ui.number(value=S.physical_long_edge, min=mn, max=mx, step=1).style(
                                    'width:96px;').props('dense outlined dark precision=0 '
                                    'input-style="font-size:12px;text-align:right;color:var(--text);"')
                                ui.html('<span style="font-size:11px;color:var(--sub2);">px</span>')

                            def set_out(v):
                                v = int(max(mn, min(mx, v)))
                                S.physical_long_edge = v
                                out_num.set_value(v)
                            out_num.on('update:model-value',
                                       lambda e: set_out(float(e.args)) if e.args not in (None, '') else None)

                            with ui.element('div').style('display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;'):
                                for lbl, pxv in [('720', 1280), ('1080', 1920), ('1440', 2560),
                                                 ('4K', 3840), ('8K', 7680)]:
                                    ui.button(lbl, on_click=lambda _=None, p=pxv: set_out(p)).style(
                                        'font-size:11px;padding:3px 9px;background:var(--accent) !important;'
                                        'border:1px solid var(--accent);border-radius:6px;color:white;')
                                if S.media_mode == 'video' and S.video_path:
                                    def match_src():
                                        try:
                                            import cv2
                                            cap = cv2.VideoCapture(S.video_path)
                                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                            cap.release()
                                            set_out(max(w, h))
                                        except Exception:
                                            pass
                                    ui.button('Match source', on_click=match_src).style(
                                        'font-size:11px;padding:3px 9px;background:var(--accent) !important;'
                                        'border-radius:6px;color:white;')
                            ui.html('<div style="font-size:10px;color:var(--sub2);padding-left:2px;'
                                    'margin-top:4px;">Longest edge of the output in pixels '
                                    '(video output resolution).</div>')
                    else:
                        labeled_phys('Output long edge', 'physical_long_edge', *R['physical_long_edge'],
                                     fmt='{:.0f}', hint='Physical size of the longest edge of the final painting')

                    labeled_phys('Min diameter', 'r_min', *R['r_min'],
                                 hint='Smallest circle — used in fine-detail & highlight areas')
                    labeled_phys('Max diameter', 'r_max', *R['r_max'],
                                 hint='Largest circle — used in smooth & shadow areas')
                    labeled_phys('Circle gap', 'gap', *R['gap'],
                                 hint='Space between circle edges. Negative = splat overlap (airbrush mode)')

                size_fields()


            # ── Algorithm Params ───────────────────────────────────────────────
            with ui.element('div').classes('card') as algorithm_card_el:
                algorithm_title = ui.html('<div class="card-title">4. Algorithm</div>')

                # Every labeled_float/labeled_int control registers itself here so
                # "Reset to defaults" can restore both S's value and the widget's
                # on-screen display without a full card re-render.
                _algo_controls = []

                def labeled_float(label, attr, mn, mx, step, fmt='{:.1f}', unit_sfx='', hint=''):
                    val_el = [None]
                    with ui.element('div').style('margin-bottom:10px;'):
                        with ui.element('div').classes('param-row').style('margin-bottom:2px;'):
                            ui.html(f'<span class="param-label">{label}</span>')
                            sl = ui.slider(min=mn, max=mx, value=getattr(S, attr), step=step).style('flex:1;').props('dark color=deep-purple')
                            val_el[0] = ui.html(
                                f'<span class="param-value">{fmt.format(getattr(S, attr))}{unit_sfx}</span>'
                            )
                        if hint:
                            ui.html(f'<div style="font-size:10px;color:var(--sub2);line-height:1.4;padding-left:2px;">{hint}</div>')
                    def on_change(e):
                        setattr(S, attr, float(e.args))
                        val_el[0].set_content(
                            f'<span class="param-value">{fmt.format(float(e.args))}{unit_sfx}</span>'
                        )
                    sl.on('update:model-value', on_change)
                    def _reset_float():
                        d = getattr(State, attr)
                        setattr(S, attr, d)
                        sl.set_value(d)
                        val_el[0].set_content(f'<span class="param-value">{fmt.format(d)}{unit_sfx}</span>')
                    _algo_controls.append(_reset_float)
                    return sl

                with ui.element('div').style('margin-bottom:14px;'):
                    pq_chk = ui.checkbox('Pre-quantize to palette').props('dense color=purple').style(
                        'font-size:13px;color:var(--text);white-space:nowrap;'
                    )
                    pq_chk.value = S.pre_quantize
                    pq_chk.on('update:model-value', lambda e: setattr(S, 'pre_quantize', bool(e.args)))
                    def _reset_pq():
                        S.pre_quantize = State.pre_quantize
                        pq_chk.set_value(State.pre_quantize)
                    _algo_controls.append(_reset_pq)
                    ui.html(
                        '<div style="font-size:10px;color:var(--sub2);line-height:1.4;'
                        'padding-left:2px;margin-top:3px;">'
                        'Remap every source pixel to its nearest palette color before fitting circles — '
                        'sharpens color boundaries, forces all palette colors to appear where they match'
                        '</div>'
                    )

                labeled_float('Color softness', 'color_softness', 5, 80, 1, '{:.0f}', ' ΔE',
                              hint='How far each paint color\'s influence bleeds beyond its region')
                labeled_float('Shadow color', 'chroma_weight', 0.0, 1.0, 0.05, '{:.0%}',
                              hint='Reduces lightness weight in color matching — palette colors compete in shadows instead of black monopolizing dark areas')
                labeled_float('Canvas colour compensation', 'canvas_comp', 0.0, 0.8, 0.01, '{:.0%}',
                              hint='How much the canvas colour shows through the gaps on the finished piece. '
                                   'Higher = paint colours shift further to offset a non-neutral canvas so the '
                                   'image still reads correctly. 0% = ignore the canvas colour.')
                labeled_float('Detail boost', 'detail_boost', 0.5, 4.0, 0.1, '{:.1f}', '×',
                              hint='Amplifies edge detection — higher = smaller circles on fine detail')
                labeled_float('Luminosity → size', 'lum_influence', 0.0, 1.0, 0.05, '{:.0%}',
                              hint='0% = size from detail only · 100% = size from brightness only (dark=big, bright=small)')
                with ui.element('div').style('margin-bottom:12px;'):
                    ui.html('<span class="param-label">Working resolution</span>')
                    _RES_OPTS = [(1.0, 'Full'), (0.5, '½'), (0.25, '¼'), (0.125, '⅛')]
                    with ui.element('div').classes('seg-ctrl').style('margin-top:4px;') as reswork_seg:
                        reswork_btns: dict[float, object] = {}
                        def make_res_setter(frac):
                            def setter():
                                S.working_res_fraction = frac
                                if S.source_native_long:
                                    S.max_working_px = max(200, round(S.source_native_long * frac))
                                _refresh_reswork()
                            return setter
                        for frac, flabel in _RES_OPTS:
                            b = ui.element('div').classes('seg-btn').style('flex:1;text-align:center;')
                            b.on('click', make_res_setter(frac))
                            with b:
                                ui.html(flabel)
                            reswork_btns[frac] = b
                    reswork_note = ui.html('')
                    def _refresh_reswork():
                        for frac, b in reswork_btns.items():
                            b.classes(add='active') if frac == S.working_res_fraction else b.classes(remove='active')
                        if S.source_native_long:
                            px = S.max_working_px
                            reswork_note.set_content(
                                f'<div style="font-size:10px;color:var(--sub2);line-height:1.4;padding-left:2px;'
                                f'margin-top:5px;">≈{px}px internal working size (source is {S.source_native_long}px). '
                                'Lower resolutions render faster but are less accurate.</div>'
                            )
                        else:
                            reswork_note.set_content(
                                '<div style="font-size:10px;color:var(--sub2);line-height:1.4;padding-left:2px;'
                                'margin-top:5px;">Lower resolutions render faster but are less accurate. '
                                'Load an image or video to see the exact working size.</div>'
                            )
                    _reswork_refresh['fn'] = _refresh_reswork
                    _refresh_reswork()
                    def _reset_reswork():
                        S.working_res_fraction = State.working_res_fraction
                        if S.source_native_long:
                            S.max_working_px = max(200, round(S.source_native_long * State.working_res_fraction))
                        else:
                            S.max_working_px = State.max_working_px
                        _refresh_reswork()
                    _algo_controls.append(_reset_reswork)
                labeled_int('Poisson candidates', 'poisson_k', 10, 60, 1,
                            hint='Attempts per circle placement. Higher = denser gap-fill, slower start',
                            register=_algo_controls)

                # Still-image iterations — video uses "Iterations · per frame" instead.
                with ui.element('div') as _iter_block:
                    ui.html('<div class="divider" style="margin:10px 0 8px;"></div>')
                    ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:8px;font-weight:600;">Optimization (Lloyd iterations)</div>')
                    labeled_int('Iterations', 'n_iterations', 0, 30, 1,
                                hint='Rounds of refinement. 0 = Poisson only (fast) · 8–16 = good · 20+ = slow',
                                register=_algo_controls)
                still_only_els.append(_iter_block)

                ui.html('<div class="divider" style="margin:10px 0 8px;"></div>')
                ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:8px;font-weight:600;">Optical Color Mixing</div>')
                labeled_float('Dither strength', 'dither_strength', 0.0, 1.0, 0.05, '{:.0%}',
                              hint='How much adjacent circles mix colors (pointillism). 0 = nearest color only')
                labeled_float('Blend radius', 'blend_radius', 1.5, 16.0, 0.5, '{:.1f}', '× r',
                              hint='How far color error spreads spatially when dithering (higher = more spread)')

                ui.html('<div style="font-size:11px;color:var(--sub);margin-top:4px;line-height:1.6;">'
                        '<b style="color:var(--text);">Optical mixing</b> places adjacent circles of '
                        'different colors so the eye blends them into intermediate tones not in your '
                        'palette — like pointillism. Higher dither strength = richer color range. '
                        'Blend radius controls how far the color error propagates spatially.'
                        '<br><br>'
                        'More iterations → circles converge toward lower perceptual error. '
                        '0 = Poisson-only (fast). 8–15 = good quality. 20+ = high quality (slow).'
                        '</div>')

                def reset_algorithm_defaults():
                    for reset_fn in _algo_controls:
                        reset_fn()
                ui.button('Reset to defaults', on_click=reset_algorithm_defaults).style(
                    'margin-top:14px;width:100%;font-size:12px;padding:7px 0;'
                    'background:transparent !important;border:1px solid var(--sub2);'
                    'border-radius:6px;color:var(--sub);'
                )


        # ── MAIN PANEL ────────────────────────────────────────────────────────
        with ui.element('div').classes('app-main') as main_el:

            # ── Live preview ──────────────────────────────────────────────────
            with ui.element('div').classes('card').style('max-width:860px;'):
                with ui.element('div').style('display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:8px;'):
                    ui.html('<div class="card-title" style="margin-bottom:0;">Live Preview</div>')
                    iter_badge = ui.html(
                        '<div style="font-size:11px;color:var(--sub);'
                        'background:var(--card2);border:1px solid var(--border);'
                        'border-radius:5px;padding:3px 10px;">Waiting…</div>'
                    )

                # Interactive canvas — zoom (scroll) · pan (drag) · double-click to reset
                ui.html(
                    '<div id="pb-wrap" style="position:relative;width:100%;background:#13151A;'
                    'border-radius:8px;overflow:hidden;min-height:240px;">'
                    '<div id="pb-canvas" style="display:block;width:100%;overflow:hidden;"></div>'
                    '<img id="smv-render" style="display:none;position:absolute;inset:0;width:100%;height:100%;'
                    'object-fit:contain;background:#13151A;"/>'
                    '<div id="pb-hint" style="position:absolute;inset:0;display:flex;flex-direction:column;'
                    'align-items:center;justify-content:center;color:#4A4F6A;font-family:Inter,sans-serif;'
                    'font-size:13px;pointer-events:none;">'
                    '<div>Preview will appear here during optimization</div>'
                    '<div style="font-size:11px;margin-top:6px;opacity:0.6;">'
                    'Scroll to zoom · Drag to pan · Double-click to reset</div>'
                    '</div></div>'
                )

                # ── Processed-video scrubber (hidden until a clip is processed) ────
                scrubber_row = ui.element('div').style(
                    'display:none;align-items:center;gap:10px;margin-top:12px;')
                with scrubber_row:
                    def _sbtn(txt):
                        return ui.button(txt).style(
                            'font-size:12px;padding:5px 10px;background:var(--accent) !important;'
                            'border:1px solid var(--accent);border-radius:6px;color:white;min-width:36px;')
                    prev_fbtn = _sbtn('◀')
                    play_fbtn = ui.button('▶ Play').style(
                        'font-size:12px;padding:5px 14px;background:var(--accent) !important;'
                        'color:white;border-radius:6px;font-weight:600;')
                    next_fbtn = _sbtn('▶')
                    # 0–1000 track mapped proportionally to the frame count in JS, so it
                    # always spans every frame without needing dynamic max updates.
                    scrub_slider = ui.slider(min=0, max=1000, value=0, step=1).style('flex:1;').props('dark color=deep-purple')
                    ui.html('<span id="smv-lbl" style="font-family:monospace;font-size:12px;'
                            'color:var(--sub);min-width:96px;text-align:right;">frame 0 / 0</span>')

                    scrub_slider.on('update:model-value',
                                    lambda e: client.run_javascript(
                                        f'window.smaltiScrub({(e.args[0] if isinstance(e.args,(list,tuple)) else e.args) or 0}/1000)'))
                    prev_fbtn.on('click', lambda: client.run_javascript('window.smaltiPrev()'))
                    next_fbtn.on('click', lambda: client.run_javascript('window.smaltiNext()'))

                    _playing = [False]
                    async def toggle_play():
                        if _playing[0]:
                            await client.run_javascript('window.smaltiStop()')
                            _playing[0] = False
                            play_fbtn.set_text('▶ Play')
                        else:
                            await client.run_javascript('window.smaltiPlay()')
                            _playing[0] = True
                            play_fbtn.set_text('⏸ Pause')
                    play_fbtn.on('click', toggle_play)

            # ── Generate card ─────────────────────────────────────────────────
            with ui.element('div').classes('card').style('max-width:860px;'):
                generate_title = ui.html('<div class="card-title">6. Generate</div>')

                # ── Render mode toggle ────────────────────────────────────────
                if S.airbrush_eq is None:
                    S.airbrush_eq = [1.2, 1.2, 1.2, 1.2, 1.2]

                mode_btns: dict[str, object] = {}
                mode_ctrl_el = [None]
                airbrush_eq_el = [None]
                poly_panel_el = [None]
                poly_gradient_row_el = [None]

                with ui.element('div').style('display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:10px;'):
                    ui.html('<span style="font-size:12px;color:var(--text);">Render mode</span>')
                    with ui.element('div').classes('seg-ctrl').style('flex-shrink:0;') as seg_ctrl:
                        mode_ctrl_el[0] = seg_ctrl
                        _mode_options = [('circle', 'Circle'), ('airbrush', 'Airbrush'), ('polygon', 'Polygon')]
                        if TILE_MODE_ENABLED:
                            _mode_options.append(('tile', 'Tile'))
                        for mid, mlabel in _mode_options:
                            is_active = mid == S.mode
                            def make_mode_setter(m):
                                async def setter():
                                    # Mode switching is locked while a generation is running:
                                    # re-rendering a rapidly-updating large point set in a new mode
                                    # mid-optimization can crash the renderer.
                                    if S.processing:
                                        return
                                    # State-only: the 100ms _sync_js_state timer pushes S.mode
                                    # to JS and pbSyncState redraws on change (same reliable path
                                    # as the sliders). Panels/buttons still update immediately.
                                    S.mode = m
                                    for k, b in mode_btns.items():
                                        if k == m:
                                            b.classes(add='active')
                                        else:
                                            b.classes(remove='active')
                                    if airbrush_eq_el[0]:
                                        airbrush_eq_el[0].set_visibility(m == 'airbrush')
                                    if poly_panel_el[0]:
                                        poly_panel_el[0].set_visibility(m in ('polygon', 'tile'))
                                    if poly_gradient_row_el[0]:
                                        poly_gradient_row_el[0].set_visibility(m == 'polygon')
                                    if airbrush_order_row_el[0]:
                                        airbrush_order_row_el[0].set_visibility(
                                            m == 'airbrush' and S.export_format == 'svg')
                                    # Update the video scrubber to preview in the real mode.
                                    if S.media_mode == 'video':
                                        # Fire-and-forget: smaltiSetRenderMode has no return
                                        # value we need, and awaiting it here was a real bug —
                                        # this app's websocket bridge times out an *awaited*
                                        # run_javascript at 1.0s under even moderate load (the
                                        # log is full of these for the unrelated _sync_js_state
                                        # timer too), and the bare `except: pass` that used to
                                        # sit here silently ate that timeout every time it hit.
                                        # The net effect: S.mode flipped to 'tile' server-side
                                        # (so the button visibly highlighted), but the JS-side
                                        # S.renderMode the scrubber's render() actually reads
                                        # never got updated, so it kept drawing plain circles —
                                        # exactly the "Tile is selected but I still see circles"
                                        # symptom, with no error surfaced anywhere.
                                        client.run_javascript(
                                            f'window.smaltiSetRenderMode && window.smaltiSetRenderMode("{m}")')
                                    elif S.last_result:
                                        # Switching to Polygon/Tile after generating in a
                                        # DIFFERENT mode has no real cell/tile geometry to
                                        # show yet (pipeline.py only computes the one the
                                        # run was actually started in) — the JS preview's
                                        # own staleness guard would otherwise silently fall
                                        # back to circles. Recompute on demand from the
                                        # same finished circle set and push a fresh payload,
                                        # same function session-restore already uses.
                                        # Tile mode's recompute is the real, full-resolution
                                        # tile pass (same one export uses, deliberately not
                                        # a faster approximation — see compute_preview_tiles'
                                        # docstring) so this can take well over a minute on
                                        # a large piece; the status line says so instead of
                                        # looking frozen.
                                        if m == 'tile':
                                            status_el[0].set_content(
                                                '<div style="font-size:13px;color:var(--sub);">'
                                                'Computing tile layout for preview — this runs the same '
                                                'full-resolution pass as export, so it can take a while…</div>')
                                        try:
                                            payload = await asyncio.get_event_loop().run_in_executor(
                                                None, _preview_json_from_result, S.last_result)
                                            if S.mode == m:   # user hasn't clicked elsewhere meanwhile
                                                client.run_javascript(f'window.updatePaintbot({payload})')
                                                if m == 'tile':
                                                    status_el[0].set_content('')
                                        except Exception as e:
                                            print(f"[mode-switch preview] failed to recompute {m}: {e!r}", flush=True)
                                            import traceback; traceback.print_exc()
                                            if S.mode == m:
                                                status_el[0].set_content(
                                                    f'<div style="font-size:13px;color:#FA6E9E;">'
                                                    f'Could not build the {m} preview: {e}</div>')
                                return setter
                            mbtn = ui.element('button').classes(
                                f'seg-btn{"  active" if is_active else ""}'
                            ).on('click', make_mode_setter(mid))
                            with mbtn:
                                ui.html(mlabel)
                            mode_btns[mid] = mbtn
                    ui.html(
                        '<span style="font-size:11px;color:var(--sub);">'
                        'Circle = solid discs &nbsp;·&nbsp; Airbrush = Gaussian splats &nbsp;·&nbsp; Polygon = faceted cells'
                        '</span>'
                    )

                # Airbrush EQ — blob scale per radius band, shown only in airbrush mode
                eq_band_labels = ['XS', 'S', 'M', 'L', 'XL']
                with ui.element('div').style(
                    'margin-bottom:14px;padding:10px 12px;'
                    'background:var(--card2);border-radius:8px;'
                ) as eq_panel:
                    airbrush_eq_el[0] = eq_panel
                    eq_panel.set_visibility(S.mode == 'airbrush')
                    with ui.element('div').style('display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'):
                        ui.html('<div style="font-size:11px;color:var(--sub);font-weight:600;">Blob size EQ &nbsp;<span style="font-weight:400;">(small → large circles)</span></div>')
                        async def reset_eq():
                            default = [1.2, 1.2, 1.2, 1.2, 1.2]
                            S.airbrush_eq[:] = default
                            for i, sl in enumerate(eq_sliders):
                                sl.set_value(default[i])
                            for lbl in eq_val_labels:
                                lbl.set_text('1.20')
                            eq_js = json.dumps(default)
                            await client.run_javascript(
                                f'window.pbAirbrushEQ={eq_js};'
                                'if(window.circleData) draw();'
                            )
                        ui.button('Reset').on('click', reset_eq).style(
                            'font-size:10px;padding:2px 8px;background:var(--accent) !important;'
                            'border:1px solid var(--accent);border-radius:4px;color:white;'
                        )
                    with ui.element('div').style('display:flex;gap:10px;align-items:flex-end;'):
                        eq_sliders = []
                        eq_val_labels = []
                        for band_i, label in enumerate(eq_band_labels):
                            with ui.element('div').style('display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;'):
                                ui.html(f'<div style="font-size:10px;color:var(--sub);">{label}</div>')
                                def make_eq_setter(idx, val_lbl):
                                    async def setter(e):
                                        raw = e.args
                                        try:
                                            v = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
                                        except (TypeError, ValueError):
                                            return
                                        S.airbrush_eq[idx] = v
                                        val_lbl.set_text(f'{v:.2f}')
                                        eq_js = json.dumps([round(x, 4) for x in S.airbrush_eq])
                                        # Debounce the actual (expensive, chunked) rebuild so a fast
                                        # slider drag doesn't restart it on every tick — only the
                                        # value assignment happens immediately.
                                        await client.run_javascript(
                                            f'window.pbAirbrushEQ={eq_js};'
                                            'if(window.circleData) debounceCall("airbrush-eq",120,draw);'
                                        )
                                    return setter
                                # vertical=True, no custom CSS — Quasar's native vertical
                                # slider increases when dragging upward.
                                sl = ui.slider(min=0.5, max=3.0, step=0.05, value=S.airbrush_eq[band_i]).style(
                                    'height:80px;'
                                ).props('vertical reverse')
                                val_lbl = ui.label(f'{S.airbrush_eq[band_i]:.2f}').style(
                                    'font-size:10px;color:var(--sub2);'
                                )
                                sl.on('update:model-value', make_eq_setter(band_i, val_lbl))
                                eq_sliders.append(sl)
                                eq_val_labels.append(val_lbl)

                # Voronoi cell outline — shown only in polygon mode
                with ui.element('div').style(
                    'margin-bottom:14px;padding:10px 12px;'
                    'background:var(--card2);border-radius:8px;'
                ) as poly_panel:
                    poly_panel_el[0] = poly_panel
                    poly_panel.set_visibility(S.mode in ('polygon', 'tile'))
                    ui.html('<div style="font-size:11px;color:var(--sub);font-weight:600;margin-bottom:8px;">Cell outline ratio</div>')
                    poly_num_el = [None]
                    with ui.element('div').classes('param-row'):
                        poly_sl = ui.slider(min=0.0, max=0.4, value=S.poly_inset, step=0.01).style('flex:1;').props('dark color=deep-purple')
                        poly_num_el[0] = ui.number(value=S.poly_inset, min=0.0, max=0.4, step=0.01).style('width:64px;').props(
                            'dense outlined dark input-style="font-size:12px;text-align:right;color:var(--text);"'
                        )
                    ui.html('<div style="font-size:10px;color:var(--sub2);margin-top:4px;">'
                            'Canvas outline as a fraction of each cell\'s size — adapts automatically to large and small cells. 0 = no gap.'
                            '</div>')
                    def _scrub_poly_refresh():
                        # Re-render the parked video scrub frame when polygon settings change
                        # (JS debounces, so dragging only triggers one server render).
                        if S.media_mode == 'video':
                            try:
                                client.run_javascript('window.smaltiRefresh && window.smaltiRefresh()')
                            except Exception:
                                pass

                    async def on_poly_slider(e):
                        raw = e.args
                        try:
                            v = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
                        except (TypeError, ValueError):
                            return
                        # Only update Python state + mirror the paired widget.  The 100ms
                        # _sync_js_state timer pushes the value to JS and pbSyncState redraws
                        # on change — avoids a per-drag-event run_javascript backlog.
                        S.poly_inset = v
                        poly_num_el[0].set_value(v)
                        _scrub_poly_refresh()
                    async def on_poly_number(e):
                        try:
                            v = max(0.0, min(0.4, float(e.args)))
                        except (TypeError, ValueError):
                            return
                        S.poly_inset = v
                        poly_sl.set_value(v)
                        _scrub_poly_refresh()
                    poly_sl.on('update:model-value', on_poly_slider)
                    poly_num_el[0].on('update:model-value', on_poly_number)

                    ui.separator().style('margin:10px 0;opacity:0.3;')
                    ui.html('<div style="font-size:11px;color:var(--sub);font-weight:600;margin-bottom:6px;">Curve smoothing</div>')
                    smooth_num_el = [None]
                    with ui.element('div').classes('param-row'):
                        smooth_sl = ui.slider(min=0.0, max=1.0, value=S.poly_smooth, step=0.05).style('flex:1;').props('dark color=deep-purple')
                        smooth_num_el[0] = ui.number(value=S.poly_smooth, min=0.0, max=1.0, step=0.05).style('width:64px;').props(
                            'dense outlined dark input-style="font-size:12px;text-align:right;color:var(--text);"'
                        )
                    ui.html('<div style="font-size:10px;color:var(--sub2);margin-bottom:4px;">0 = hard facets · 1 = smooth blobs (Catmull-Rom spline)</div>')
                    async def on_smooth_slider(e):
                        raw = e.args
                        try:
                            v = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
                        except (TypeError, ValueError):
                            return
                        # State-only; the 100ms sync timer pushes to JS and redraws on change.
                        S.poly_smooth = v
                        smooth_num_el[0].set_value(v)
                        _scrub_poly_refresh()
                    async def on_smooth_number(e):
                        try:
                            v = max(0.0, min(1.0, float(e.args)))
                        except (TypeError, ValueError):
                            return
                        S.poly_smooth = v
                        smooth_sl.set_value(v)
                        _scrub_poly_refresh()
                    smooth_sl.on('update:model-value', on_smooth_slider)
                    smooth_num_el[0].on('update:model-value', on_smooth_number)

                    ui.separator().style('margin:10px 0;opacity:0.3;')
                    with ui.element('div') as poly_gradient_row:
                        poly_gradient_row_el[0] = poly_gradient_row
                        poly_gradient_row.set_visibility(S.mode == 'polygon')
                        with ui.element('div').style('display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);font-weight:600;">Color gradients</div>')
                            grad_toggle = ui.checkbox(value=S.poly_gradient).props('dark color=deep-purple')
                        ui.html('<div style="font-size:10px;color:var(--sub2);margin-bottom:4px;">Blend adjacent colors across each cell — pie-slice gradient from center to each edge</div>')
                        async def on_grad_toggle(e):
                            raw = e.args
                            # State-only; the 100ms sync timer pushes to JS and redraws on change.
                            S.poly_gradient = bool(raw[0] if isinstance(raw, (list, tuple)) else raw)
                        grad_toggle.on('update:model-value', on_grad_toggle)

                status_el[0] = ui.html(
                    '<div style="font-size:13px;color:var(--sub);margin-bottom:8px;">'
                    'Upload an image and configure your palette to begin.</div>'
                )

                with ui.element('div').classes('progress-track'):
                    progress_fill_el[0] = ui.element('div').classes('progress-fill').style('width:0%;')

                run_btn = ui.button('Generate').style(
                    'width:100%;padding:13px 0;border-radius:8px;'
                    'background:linear-gradient(135deg,#7C6EFA,#FA6E9E) !important;'
                    'color:white;font-weight:700;font-size:14px;letter-spacing:0.03em;'
                )
                cancel_btn = ui.button('Cancel').style(
                    'width:100%;padding:10px 0;border-radius:8px;margin-top:8px;'
                    'background:transparent !important;border:1px solid var(--border);'
                    'color:var(--sub);font-weight:600;font-size:13px;'
                )
                cancel_btn.set_visibility(False)
                cancel_btn.on('click', lambda: S.stop_event.set() if hasattr(S, 'stop_event') and S.stop_event else None)

                def run_btn_set_busy():
                    run_btn.props('disabled')
                    run_btn.style('opacity:0.45;cursor:not-allowed;')
                    run_btn.set_text('Processing…' if S.media_mode == 'video' else 'Generating…')

                def run_btn_set_idle():
                    run_btn.props(remove='disabled')
                    run_btn.style('opacity:1;cursor:pointer;')
                    run_btn.set_text('Process video' if S.media_mode == 'video' else 'Generate')

                def update_media_ui():
                    is_vid = (S.media_mode == 'video')
                    try:
                        video_opts_card.set_visibility(is_vid)
                    except Exception:
                        pass
                    # Hide still-image-only controls (export format picker, per-layer toggle,
                    # still Iterations) while working on a video.
                    for _el in still_only_els:
                        try:
                            _el.set_visibility(not is_vid)
                        except Exception:
                            pass
                    # Show video-only controls (render output settings in Export) only
                    # while working on a video.
                    for _el in video_only_els:
                        try:
                            _el.set_visibility(is_vid)
                        except Exception:
                            pass
                    # video_export_row's own visibility depends on the sequence
                    # dropdown's format too (on_seq_fmt handles that case) — this
                    # only needs to force it OFF when leaving video mode entirely,
                    # since those settings are meaningless outside video mode.
                    if not is_vid:
                        try:
                            video_export_row.set_visibility(False)
                        except Exception:
                            pass
                    try:
                        run_btn.set_text('Process video' if is_vid else 'Generate')
                        save_btn.set_visibility(not is_vid)
                    except Exception:
                        pass
                    # Video Options is tile 4 when visible, bumping everything after it.
                    try:
                        n = 5 if is_vid else 4
                        algorithm_title.set_content(f'<div class="card-title">{n}. Algorithm</div>')
                        output_size_title.set_content(f'<div class="card-title">{n + 1}. Output Size</div>')
                        generate_title.set_content(f'<div class="card-title">{n + 2}. Generate</div>')
                        export_title.set_content(f'<div class="card-title">{n + 3}. Export</div>')
                    except Exception:
                        pass
                _media_refresh['fn'] = update_media_ui

                async def _process_video_run():
                    from core.video_mosaic import process_video
                    if not S.video_path or not os.path.exists(S.video_path):
                        status_el[0].set_content(
                            '<div style="font-size:13px;color:#FA6E9E;">⚠ Choose a video first.</div>')
                        return
                    S.processing = True
                    S.job_kind = 'video_process'
                    S.last_error = None
                    S.stop_event = threading.Event()
                    run_btn_set_busy()
                    cancel_btn.set_visibility(True)
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(add='locked')
                    progress_fill_el[0].style('width:0%;')
                    loop = asyncio.get_event_loop()
                    _client = client

                    def prog(idx, total, msg):
                        pct = (idx / total) if total else 0.0
                        S.progress_pct = pct
                        S.progress_msg = f'{msg} ({idx}/{total})'
                        async def _u():
                            progress_fill_el[0].style(f'width:{pct*100:.1f}%;')
                            status_el[0].set_content(
                                f'<div style="font-size:13px;color:var(--sub);">{msg} ({idx}/{total})</div>')
                        asyncio.run_coroutine_threadsafe(_u(), loop)

                    # Reset the server-side frame store + the browser player, and hide the
                    # scrubber until the first frame completes.
                    _SCRUB['frames'] = []
                    _SCRUB['meta'] = {}
                    try:
                        _client.run_javascript('window.smaltiReset && window.smaltiReset()')
                    except Exception:
                        pass
                    scrubber_row.style('display:none;')

                    # Coalesce rapid INTERMEDIATE previews (frame-1 building) for the live
                    # view.  COMPLETED frames ('final') are never dropped — they're added to
                    # the scrubber immediately so you can shuffle processed frames mid-run.
                    _pv_pending = [None]
                    _pv_scheduled = [False]

                    def _compact(m):
                        # Drop the heavy Voronoi/neighbour arrays and round coords — the
                        # preview + scrubber render circles, so full precision + polygons
                        # would just risk overflowing the socket at high dot counts.
                        keep = ('palette', 'canvas_hex', 'W', 'H', 'r_min', 'r_max',
                                'phase', 'frame', 'total')
                        d = {k: m[k] for k in keep if k in m}
                        d['xs'] = [round(v, 1) for v in m.get('xs', [])]
                        d['ys'] = [round(v, 1) for v in m.get('ys', [])]
                        d['rs'] = [round(v, 1) for v in m.get('rs', [])]
                        d['colors'] = m.get('colors', [])
                        d['mode'] = 'circle'
                        d['neighbors'] = None
                        d['polygons'] = None
                        return json.dumps(d, separators=(',', ':'))

                    _final_pending = [None]
                    _final_scheduled = [False]

                    def preview(payload):
                        try:
                            _m = json.loads(payload)
                        except Exception:
                            _m = {}
                        phase = _m.get('phase')
                        if phase == 'final':
                            # Store the finished frame SERVER-SIDE (this runs in the worker
                            # thread; list append is GIL-atomic).  No big per-frame socket push.
                            _SCRUB['frames'].append({
                                'x': _m.get('xs', []), 'y': _m.get('ys', []),
                                'r': _m.get('rs', []), 'c': _m.get('colors', [])})
                            if not _SCRUB['meta']:
                                _SCRUB['meta'] = {
                                    'palette': _m.get('palette'), 'canvas': _m.get('canvas_hex'),
                                    'working_size': [_m.get('W'), _m.get('H')],
                                    'r_min': _m.get('r_min'), 'r_max': _m.get('r_max'),
                                    'airbrush_eq': (list(S.airbrush_eq) if S.airbrush_eq else None)}
                            cnt = len(_SCRUB['frames'])
                            fno = int(_m.get('frame', cnt)); tot = _m.get('total', '?')
                            _final_pending[0] = (cnt, fno, tot)
                            if _final_scheduled[0]:
                                return
                            _final_scheduled[0] = True

                            async def _live():
                                await asyncio.sleep(0.25)   # coalesce live updates
                                item = _final_pending[0]; _final_pending[0] = None
                                _final_scheduled[0] = False
                                if item is None:
                                    return
                                c, f, t = item
                                try:
                                    _client.run_javascript(f'window.smaltiSetCount({c})')
                                except Exception:
                                    pass
                                scrubber_row.style('display:flex;')
                                iter_badge.set_content(
                                    f'<div style="font-size:11px;color:var(--accent);'
                                    f'background:rgba(124,110,250,0.1);border:1px solid rgba(124,110,250,0.3);'
                                    f'border-radius:5px;padding:3px 10px;">🎬 Frame {f}/{t}</div>')
                            asyncio.run_coroutine_threadsafe(_live(), loop)
                            return

                        cpayload = _compact(_m)

                        _lbl = f"Frame {_m.get('frame','?')}/{_m.get('total','?')}"
                        _pv_pending[0] = (cpayload, _lbl)
                        if _pv_scheduled[0]:
                            return
                        _pv_scheduled[0] = True

                        async def _flush():
                            await asyncio.sleep(0.2)
                            item = _pv_pending[0]; _pv_pending[0] = None; _pv_scheduled[0] = False
                            if item is None:
                                return
                            _jd, _label = item
                            try:
                                _client.run_javascript(f'window.updatePaintbot({_jd})')
                            except Exception:
                                pass
                            iter_badge.set_content(
                                f'<div style="font-size:11px;color:var(--accent);'
                                f'background:rgba(124,110,250,0.1);border:1px solid rgba(124,110,250,0.3);'
                                f'border-radius:5px;padding:3px 10px;">🎬 {_label}</div>')
                        asyncio.run_coroutine_threadsafe(_flush(), loop)

                    holder = {}
                    _stop = S.stop_event
                    def worker():
                        try:
                            holder['r'] = process_video(
                                S.video_path, S.output_dir,
                                palette=list(S.palette), canvas_hex=S.canvas_hex, mode=S.mode,
                                r_min_px=S.r_min / 2.0, r_max_px=S.r_max / 2.0,
                                max_working_px=S.max_working_px,
                                iters_first=S.n_iterations, iters_warm=S.video_iters_warm,
                                fresh=S.video_fresh,
                                frame_stride=S.video_stride, max_frames=S.video_max_frames,
                                poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                airbrush_eq=list(S.airbrush_eq) if S.airbrush_eq else None,
                                canvas_comp=S.canvas_comp, lum_influence=S.lum_influence,
                                detail_boost=S.detail_boost,
                                progress_cb=prog, preview_cb=preview, stop_event=_stop)
                        except Exception:
                            import traceback
                            holder['e'] = traceback.format_exc()
                            S.last_error = holder['e']   # durable: survives a page reload
                    t = threading.Thread(target=worker, daemon=True)
                    S.worker_thread = t
                    t.start()
                    while t.is_alive():
                        await asyncio.sleep(0.2)

                    S.processing = False
                    S.job_kind = ''
                    S.worker_thread = None
                    cancel_btn.set_visibility(False)
                    run_btn_set_idle()
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(remove='locked')
                    if _stop.is_set():
                        status_el[0].set_content('<div style="font-size:13px;color:var(--sub);">Cancelled.</div>')
                        return
                    if 'e' in holder:
                        status_el[0].set_content(
                            f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;'
                            f'white-space:pre-wrap;">Error: {holder["e"]}</div>')
                        return
                    S.video_json_path = holder.get('r') or ''
                    progress_fill_el[0].style('width:100%;')
                    # Frames are held server-side; finalise the scrubber count.
                    count = len(_SCRUB['frames'])
                    if count > 0:
                        scrubber_row.style('display:flex;')
                        try:
                            await client.run_javascript(f'window.smaltiSetCount({count})')
                        except Exception:
                            pass
                    status_el[0].set_content(
                        f'<div style="font-size:13px;color:var(--green);">✓ {count} frames processed — '
                        'scrub / ▶ Play below to review the movement, or <b>⬇ Render video</b> to encode.</div>')

                async def on_run():
                    try:
                        await _on_run_inner()
                    except Exception as _e:
                        import traceback as _tb
                        _msg = _tb.format_exc()
                        print(f"[on_run UNHANDLED] {_msg}", flush=True)
                        S.processing = False
                        S.job_kind = ''
                        S.worker_thread = None
                        try:
                            cancel_btn.set_visibility(False)
                            run_btn_set_idle()
                            if mode_ctrl_el[0]:
                                mode_ctrl_el[0].classes(remove='locked')
                            status_el[0].set_content(
                                f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;'
                                f'white-space:pre-wrap;">Startup error:\n{_msg}</div>'
                            )
                        except Exception:
                            pass

                async def _on_run_inner():
                    print(f"[on_run] processing={S.processing} mode={S.media_mode}", flush=True)
                    if S.processing:
                        return
                    if S.media_mode == 'video':
                        await _process_video_run()
                        return
                    if not S.image_path:
                        status_el[0].set_content(
                            '<div style="font-size:13px;color:#FA6E9E;">⚠ Upload an image first.</div>'
                        )
                        return

                    S.processing = True
                    S.job_kind = 'still'
                    S.stop_event = threading.Event()
                    run_btn_set_busy()
                    cancel_btn.set_visibility(True)
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(add='locked')   # lock render-mode switch
                    progress_fill_el[0].style('width:0%;')
                    if results_el[0] is not None:
                        results_el[0].clear()
                    iter_badge.set_content(
                        '<div style="font-size:11px;color:var(--sub);'
                        'background:var(--card2);border:1px solid var(--border);'
                        'border-radius:5px;padding:3px 10px;">Processing…</div>'
                    )

                    config = PipelineConfig(
                        image_path=S.image_path,
                        max_working_px=S.max_working_px,
                        canvas_hex=S.canvas_hex,
                        palette_hex=list(S.palette),
                        r_min_physical=S.r_min / 2.0,
                        r_max_physical=S.r_max / 2.0,
                        unit=S.unit,
                        physical_long_edge=S.physical_long_edge,
                        dpi=S.dpi,
                        gap_physical=S.gap,
                        color_softness=S.color_softness,
                        canvas_comp=S.canvas_comp,
                        chroma_weight=S.chroma_weight,
                        detail_boost=S.detail_boost,
                        lum_influence=S.lum_influence,
                        poisson_k=S.poisson_k,
                        n_iterations=S.n_iterations,
                        dither_strength=S.dither_strength,
                        blend_radius=S.blend_radius,
                        dither_kernel=S.dither_kernel,
                        output_dir=S.output_dir,
                        mode=S.mode,
                        airbrush_eq=list(S.airbrush_eq) if S.airbrush_eq else None,
                        poly_inset=S.poly_inset,
                        pre_quantize=S.pre_quantize,
                    )

                    loop = asyncio.get_event_loop()
                    _client = client   # capture NiceGUI client for background use
                    result_holder = [None]
                    error_holder = [None]

                    def progress_cb(msg: str, pct: float):
                        print(f"[progress_cb] {msg} {pct:.0%}", flush=True)
                        S.progress_msg = msg
                        S.progress_pct = pct
                        async def _u():
                            status_el[0].set_content(
                                f'<div style="font-size:13px;color:var(--sub);">{msg}</div>'
                            )
                            progress_fill_el[0].style(f'width:{pct*100:.1f}%;')
                        asyncio.run_coroutine_threadsafe(_u(), loop)

                    # Rate-limit preview updates: keep only the latest pending payload.
                    # Without this, fast optimizers flood the WebSocket and drop the connection.
                    _preview_pending = [None]   # [json_data] or None
                    _preview_scheduled = [False]

                    def preview_cb(json_data: str):
                        print(f"[preview_cb] got {len(json_data)} bytes", flush=True)
                        try:
                            _meta = json.loads(json_data)
                            _phase = _meta.get('phase', '')
                            if _phase == 'iter':
                                label = f"Iteration {_meta['iter']}/{_meta['total_iter']}"
                            elif _phase == 'cleanup':
                                label = 'Cleanup pass…'
                            elif _phase == 'final':
                                label = 'Final'
                            elif _phase == 'initial':
                                label = 'Placed'
                            else:
                                label = 'Placing…'
                        except Exception:
                            label = '…'
                        _preview_pending[0] = (json_data, label)
                        if not _preview_scheduled[0]:
                            _preview_scheduled[0] = True
                            async def _flush():
                                await asyncio.sleep(0.25)   # coalesce rapid updates
                                item = _preview_pending[0]
                                _preview_pending[0] = None
                                _preview_scheduled[0] = False
                                if item is None:
                                    return
                                _jd, _label = item
                                try:
                                    _client.run_javascript(f'window.updatePaintbot({_jd})')
                                    print(f"[preview_cb._u] run_javascript called OK", flush=True)
                                except Exception as e:
                                    print(f"[preview_cb._u] ERROR: {e}", flush=True)
                                iter_badge.set_content(
                                    f'<div style="font-size:11px;color:var(--accent);'
                                    f'background:rgba(124,110,250,0.1);'
                                    f'border:1px solid rgba(124,110,250,0.3);'
                                    f'border-radius:5px;padding:3px 10px;">{_label}</div>'
                                )
                            asyncio.run_coroutine_threadsafe(_flush(), loop)

                    _stop = S.stop_event

                    def worker():
                        try:
                            result_holder[0] = run_pipeline(config, progress_cb, preview_cb, _stop)
                        except InterruptedError:
                            pass  # cancelled cleanly
                        except Exception as ex:
                            import traceback
                            error_holder[0] = traceback.format_exc()
                            S.last_error = error_holder[0]   # durable: survives a page reload

                    S.last_error = None   # clear any previous run's error before starting a new one
                    t = threading.Thread(target=worker, daemon=True)
                    S.worker_thread = t
                    t.start()
                    while t.is_alive():
                        await asyncio.sleep(0.25)

                    S.processing = False
                    S.job_kind = ''
                    S.worker_thread = None
                    cancel_btn.set_visibility(False)
                    run_btn_set_idle()
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(remove='locked')   # unlock render-mode switch

                    if _stop.is_set():
                        status_el[0].set_content(
                            '<div style="font-size:13px;color:var(--sub);">Cancelled.</div>'
                        )
                        iter_badge.set_content(
                            '<div style="font-size:11px;color:var(--sub);'
                            'background:var(--card2);border:1px solid var(--border);'
                            'border-radius:5px;padding:3px 10px;">—</div>'
                        )
                        return

                    if error_holder[0]:
                        status_el[0].set_content(
                            f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;white-space:pre-wrap;">'
                            f'Error: {error_holder[0]}</div>'
                        )
                        iter_badge.set_content(
                            '<div style="font-size:11px;color:#FA6E9E;'
                            'background:rgba(250,110,158,0.1);border:1px solid rgba(250,110,158,0.3);'
                            'border-radius:5px;padding:3px 10px;">Error</div>'
                        )
                        return

                    R = result_holder[0]
                    S.result = R
                    progress_fill_el[0].style('width:100%;')
                    status_el[0].set_content(
                        f'<div style="font-size:13px;color:var(--green);">✓ Done — '
                        f'{R.total_circles:,} circles across {len(R.layer_circles)} layers.</div>'
                    )
                    iter_badge.set_content(
                        f'<div style="font-size:11px;color:var(--green);'
                        f'background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.3);'
                        f'border-radius:5px;padding:3px 10px;">'
                        f'✓ {R.total_circles:,} circles</div>'
                    )
                    S.last_result = R
                    show_results(R)

                run_btn.on('click', on_run)

            # ── State sync timer — pushes S.mode/eq/inset to JS every 100ms ──
            # This ensures mode/EQ/polygon controls respond instantly even while
            # the asyncio loop is busy processing optimizer preview frames.
            async def _sync_js_state():
                eq = S.airbrush_eq or [1.2, 1.2, 1.2, 1.2, 1.2]
                eq_js = json.dumps([round(x, 4) for x in eq])
                grad_js = str(S.poly_gradient).lower()
                # Registration marks' dist/size are stored in S.unit, same as
                # S.physical_long_edge — converting to WORKING-canvas pixels
                # (not export pixels) is what makes the preview mirror every
                # export format's mark layout at whatever scale the preview
                # canvas itself is currently showing.
                reg_enabled_js = 'false'
                reg_dist_px, reg_size_px = 0, 0
                if S.reg_marks_enabled and S.last_result and S.physical_long_edge:
                    long_px = max(S.last_result.working_size)
                    px_per_unit = long_px / S.physical_long_edge
                    reg_dist_px = round(S.reg_mark_dist * px_per_unit, 3)
                    reg_size_px = round(S.reg_mark_size * px_per_unit, 3)
                    reg_enabled_js = 'true'
                await client.run_javascript(
                    f'window.pbSyncState("{S.mode}",{eq_js},{S.poly_inset},{grad_js},{S.poly_smooth},'
                    f'{reg_enabled_js},{reg_dist_px},{reg_size_px},"{S.reg_mark_shape}")'
                )
            ui.timer(0.1, _sync_js_state)

            # ── Results ────────────────────────────────────────────────────────
            with ui.element('div').classes('card').style('max-width:860px;'):
                export_title = ui.html('<div class="card-title">7. Export</div>')

                # Output DPI (moved here from Units & Output Size — only converts
                # physical units to pixels, but shown regardless for simplicity).
                labeled_int('Output DPI', 'dpi', 72, 600, 1,
                            hint='Pixels per inch for SVG unit conversion')
                ui.html('<div class="divider" style="margin:8px 0 12px;"></div>')

                # ── Registration marks: alignment targets at all 4 canvas corners,
                # drawn identically on every 2D export (SVG/raster/G-code/Plot SVG,
                # combined AND per-layer/per-frame files, still image + video) so
                # separately cut or printed layers can be physically aligned to
                # each other — e.g. spray-paint stencils, multi-pass plots. ──────
                with ui.element('div').style('margin-bottom:14px;'):
                    reg_marks_chk = ui.checkbox('Registration marks', value=S.reg_marks_enabled).props(
                        'dense color=deep-purple').style('font-size:13px;color:var(--text);white-space:nowrap;')
                    ui.html(
                        '<div style="font-size:10px;color:var(--sub2);line-height:1.4;'
                        'padding-left:2px;margin-top:3px;">'
                        'Draws alignment marks at all 4 canvas corners on every export — combined '
                        'and per-layer/per-frame files alike — so separately cut or printed layers '
                        'can be aligned back against each other.</div>'
                    )

                    @ui.refreshable
                    def reg_marks_fields():
                        R = PX_SIZE_RANGES if S.unit == 'px' else PHYS_SIZE_RANGES
                        labeled_phys('Distance from corner', 'reg_mark_dist', *R['reg_mark_dist'],
                                     hint="How far the mark's centre sits from each canvas corner",
                                     show_unit=True)
                        labeled_phys('Mark size', 'reg_mark_size', *R['reg_mark_size'],
                                     hint='Circle diameter, or cross / corner-tick arm length',
                                     show_unit=True)
                        with ui.element('div').style('margin-bottom:10px;'):
                            ui.html('<span class="param-label">Shape</span>')
                            with ui.element('div').classes('seg-ctrl').style('margin-top:4px;'):
                                shape_btns = {}
                                def make_shape_setter(sid):
                                    def setter():
                                        S.reg_mark_shape = sid
                                        for k, b in shape_btns.items():
                                            b.classes(add='active') if k == sid else b.classes(remove='active')
                                    return setter
                                for sid, slabel in [('circle', 'Circles'), ('cross', 'Cross'),
                                                     ('lines', 'Corner ticks')]:
                                    b = ui.element('div').classes('seg-btn').style(
                                        'flex:1;text-align:center;white-space:nowrap;')
                                    b.on('click', make_shape_setter(sid))
                                    if sid == S.reg_mark_shape:
                                        b.classes(add='active')
                                    with b:
                                        ui.html(slabel)
                                    shape_btns[sid] = b

                    with ui.element('div') as reg_marks_row:
                        reg_marks_fields()
                    reg_marks_row.set_visibility(S.reg_marks_enabled)

                    def _on_reg_marks_change(e):
                        is_on = bool(e.args if not isinstance(e.args, (list, tuple)) else e.args[0])
                        S.reg_marks_enabled = is_on
                        reg_marks_row.set_visibility(is_on)
                        if is_on:
                            # Re-fit the preview so newly-visible outward marks
                            # aren't clipped by whatever pan/zoom was active.
                            client.run_javascript(
                                'setTimeout(function(){ if(window.pbResetView) window.pbResetView(); }, 150)')
                    reg_marks_chk.on('update:model-value', _on_reg_marks_change)
                ui.html('<div class="divider" style="margin:8px 0 12px;"></div>')

                # Relocate cards built earlier (for helper-function definition order)
                # into their final visual columns — grid-area placement is independent
                # of DOM order, so this only needs to happen once, after every card
                # referenced below already exists.
                algorithm_card_el.move(algorithm_el)
                units_output_card_el.move(algorithm_el)

                # Output folder
                with ui.element('div').style('display:flex;gap:7px;align-items:center;margin-bottom:10px;'):
                    out_input = ui.input(value=S.output_dir).props('dense outlined dark').style(
                        'font-size:12px;flex:1;'
                    )
                    out_input.on('blur', lambda: setattr(S, 'output_dir', out_input.value.strip() or S.output_dir))

                    async def browse_folder():
                        # macOS-native folder chooser via osascript — runs in a subprocess,
                        # so it avoids tkinter's "must run on the main thread" crash that was
                        # taking down the whole app.  The dialog also lets you create a new
                        # folder (the standard macOS "New Folder" button).
                        def _pick():
                            import subprocess, sys
                            start = S.output_dir if os.path.isdir(S.output_dir) else str(Path.home())
                            if sys.platform == 'darwin':
                                q = '"' + start.replace('\\', '\\\\').replace('"', '\\"') + '"'
                                script = (
                                    'set d to choose folder with prompt "Choose output folder"'
                                    f' default location POSIX file {q}\n'
                                    'POSIX path of d'
                                )
                                try:
                                    r = subprocess.run(['osascript', '-e', script],
                                                       capture_output=True, text=True, timeout=300)
                                    path = r.stdout.strip()
                                    return path or None   # empty on cancel
                                except Exception:
                                    return None
                            # Non-macOS fallback: tkinter on the main thread of a fresh process.
                            try:
                                r = subprocess.run(
                                    [sys.executable, '-c',
                                     'import tkinter,tkinter.filedialog as fd;'
                                     't=tkinter.Tk();t.withdraw();'
                                     'print(fd.askdirectory() or "")'],
                                    capture_output=True, text=True, timeout=300)
                                return r.stdout.strip() or None
                            except Exception:
                                return None
                        folder = await asyncio.get_event_loop().run_in_executor(None, _pick)
                        if folder:
                            S.output_dir = folder
                            out_input.set_value(folder)

                    ui.button('Browse…').on('click', browse_folder).style(
                        'font-size:12px;padding:6px 12px;background:var(--accent) !important;'
                        'border:1px solid var(--accent);border-radius:6px;color:white;white-space:nowrap;'
                    )

                # Export format + resolution.
                px_row_el = [None]
                three_d_row_el = [None]
                obj_extra_row_el = [None]
                json_row_el = [None]
                gcode_row_el = [None]
                gcode_origin_row_el = [None]
                airbrush_order_row_el = [None]
                _RASTER = ('png', 'jpeg', 'tiff')
                _THREED = ('ply', 'obj', 'splat')
                export_fmt_area = ui.element('div')
                with export_fmt_area:
                    with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-bottom:8px;'):
                        fmt_select = ui.select(
                            {'svg': 'SVG (vector)', 'png': 'PNG', 'jpeg': 'JPEG',
                             'tiff': 'TIFF (layered · LZW)',
                             'gcode': 'G-code (pen plotter · spiral fill)',
                             'plot_svg': 'Plot SVG (stroke-only · spiral fill)',
                             'json': 'JSON (paint robot / CNC)',
                             'ply': '3D point cloud (PLY + CSV)',
                             'obj': '3D cell solids (OBJ)',
                             'splat': '3D Gaussian splats (PLY)'},
                            value=S.export_format,
                        ).props('dense outlined dark').style('flex:1;font-size:12px;')

                        # Read the widget's own .value (reliable) — the raw event payload isn't.
                        def on_fmt(_e=None):
                            S.export_format = fmt_select.value
                            if px_row_el[0]:
                                px_row_el[0].set_visibility(S.export_format in _RASTER)
                            if three_d_row_el[0]:
                                three_d_row_el[0].set_visibility(S.export_format in _THREED)
                            if obj_extra_row_el[0]:
                                obj_extra_row_el[0].set_visibility(S.export_format == 'obj')
                            if json_row_el[0]:
                                json_row_el[0].set_visibility(S.export_format in ('json', 'gcode', 'plot_svg'))
                            if gcode_row_el[0]:
                                gcode_row_el[0].set_visibility(S.export_format in ('gcode', 'plot_svg'))
                            if gcode_origin_row_el[0]:
                                gcode_origin_row_el[0].set_visibility(S.export_format == 'gcode')
                            if airbrush_order_row_el[0]:
                                airbrush_order_row_el[0].set_visibility(
                                    S.mode == 'airbrush' and S.export_format == 'svg')
                        fmt_select.on_value_change(on_fmt)

                    with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-bottom:8px;') as px_row:
                        px_row_el[0] = px_row
                        ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">Long edge</div>')
                        long_px_num = ui.number(value=S.export_long_px, min=256, max=20000, step=100).props(
                            'dense outlined dark').style('width:110px;font-size:12px;')
                        long_px_num.on('update:model-value',
                                       lambda e: setattr(S, 'export_long_px',
                                                         int(max(256, min(20000, float(e.args or S.export_long_px))))))
                        ui.html('<div style="font-size:11px;color:var(--sub2);">px</div>')
                    px_row.set_visibility(S.export_format in _RASTER)

                    # Shared 3D layout settings (both point cloud and cell solids).
                    with ui.element('div').style('margin-bottom:8px;') as three_d_row:
                        three_d_row_el[0] = three_d_row
                        with ui.element('div').style('display:flex;align-items:center;gap:8px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">Layout</div>')
                            layout_sel = ui.select(
                                {'flat': 'Flat plane',
                                 'cloud': 'Anamorphic cloud (front)',
                                 'iso': 'Isometric cloud'},
                                value=S.export_3d_layout,
                            ).props('dense outlined dark').style('flex:1;font-size:12px;')
                            layout_sel.on_value_change(lambda: setattr(S, 'export_3d_layout', layout_sel.value))
                        with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-top:6px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">Cloud depth</div>')
                            depth_sl = ui.slider(min=0.0, max=0.9, value=S.export_3d_depth, step=0.05).props(
                                'dark color=deep-purple').style('flex:1;')
                            depth_sl.on('update:model-value',
                                        lambda e: setattr(S, 'export_3d_depth',
                                                          float(e.args[0] if isinstance(e.args, (list, tuple)) else e.args)))
                        ui.html('<div style="font-size:10px;color:var(--sub2);margin-top:4px;">'
                                'Cloud = points scatter in depth but resolve into the image from a set viewpoint '
                                '(Anamorphic = front camera · Isometric = 3/4 iso camera); from any other angle it’s a random cloud.</div>')
                    three_d_row.set_visibility(S.export_format in _THREED)

                    # Extra setting for the OBJ cell-solids only: object depth.
                    # (Roundness comes from Curve smoothing, spacing from Cell outline ratio.)
                    with ui.element('div').style('margin-bottom:8px;') as obj_extra_row:
                        obj_extra_row_el[0] = obj_extra_row
                        with ui.element('div').style('display:flex;align-items:center;gap:8px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">Object depth</div>')
                            relief_sl = ui.slider(min=0.1, max=3.0, value=S.export_3d_relief, step=0.1).props(
                                'dark color=deep-purple').style('flex:1;')
                            relief_sl.on('update:model-value',
                                         lambda e: setattr(S, 'export_3d_relief',
                                                           float(e.args[0] if isinstance(e.args, (list, tuple)) else e.args)))
                        ui.html('<div style="font-size:10px;color:var(--sub2);margin-top:4px;">'
                                'Front &amp; back thickness of each solid = dot size × this. '
                                'Shape (faceted ↔ rounded) follows <b>Curve smoothing</b>; gaps follow <b>Cell outline ratio</b>. '
                                'OBJ + MTL (per-colour materials).</div>')
                    obj_extra_row.set_visibility(S.export_format == 'obj')

                    # Draw-order passes — used by BOTH the CNC/paint-robot JSON (wet-paint
                    # dispersion) and G-code (avoids a plotter zig-zagging across the whole
                    # canvas, and keeps wet media from being revisited before it dries).
                    with ui.element('div').style('margin-bottom:8px;') as json_row:
                        json_row_el[0] = json_row
                        with ui.element('div').style('display:flex;align-items:center;gap:8px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">Paint passes</div>')
                            passes_sel = ui.select(
                                {1: '1 — single pass (fastest)',
                                 2: '2 — spaced + fill',
                                 3: '3 — most drying time'},
                                value=S.export_json_passes,
                            ).props('dense outlined dark').style('flex:1;font-size:12px;')
                            passes_sel.on_value_change(lambda: setattr(S, 'export_json_passes', int(passes_sel.value)))
                        ui.html('<div style="font-size:10px;color:var(--sub2);margin-top:4px;">'
                                'How many times the head/pen sweeps each colour. <b>2–3</b> draws spaced-out '
                                'shapes first, then fills the gaps once those have dried — instead of jumping '
                                'randomly between neighbours. <b>1</b> is a single fast serpentine pass.</div>')
                    json_row.set_visibility(S.export_format in ('json', 'gcode', 'plot_svg'))

                    # G-code / Plot SVG pen-plotter settings (same spiral-fill geometry,
                    # just serialized as G-code commands vs. stroke-only SVG paths).
                    with ui.element('div').style('margin-bottom:8px;') as gcode_row:
                        gcode_row_el[0] = gcode_row
                        with ui.element('div').style('display:flex;align-items:center;gap:8px;'):
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;flex:1;">Pen tip width</div>')
                            pen_num = ui.number(value=S.pen_width_mm, min=0.05, max=10.0, step=0.05).props(
                                'dense outlined dark').style('width:90px;font-size:12px;')
                            pen_num.on('update:model-value',
                                       lambda e: setattr(S, 'pen_width_mm',
                                                         float(max(0.05, min(10.0, float(e.args if e.args not in (None, '') else S.pen_width_mm))))))
                            ui.html('<div style="font-size:11px;color:var(--sub2);">mm</div>')
                        ui.html('<div style="font-size:10px;color:var(--sub2);margin-top:4px;">'
                                'One file per colour (swap pens between files). Each shape is a single '
                                'pen-down stroke: <b>circles</b> spiral edge→centre; <b>polygons</b> spiral from the '
                                'edge to their generating dot. Pen width sets the fill spacing for full coverage. '
                                '<b>Plot SVG</b> is the same spiral geometry as G-code, saved as a stroke-only '
                                '(no-fill) SVG instead — for plotter software that imports SVG directly. '
                                '<span style="color:#E0A050;">Circle, Polygon, or Tile render mode only.</span></div>')

                        with ui.element('div').style('margin-top:8px;') as gcode_origin_row:
                            gcode_origin_row_el[0] = gcode_origin_row
                            ui.html('<div style="font-size:11px;color:var(--sub);margin-bottom:5px;">'
                                    'G-code origin (0,0)</div>')
                            _ORIGIN_OPTIONS = [
                                ('top-left', 'Top Left'), ('top-right', 'Top Right'),
                                ('bottom-left', 'Bottom Left'), ('bottom-right', 'Bottom Right'),
                            ]
                            origin_btns = {}
                            with ui.element('div').classes('seg-ctrl'):
                                for oid, olabel in _ORIGIN_OPTIONS:
                                    is_active = oid == S.gcode_origin
                                    def make_origin_setter(o):
                                        def setter():
                                            S.gcode_origin = o
                                            for k, b in origin_btns.items():
                                                b.classes(add='active') if k == o else b.classes(remove='active')
                                        return setter
                                    obtn = ui.element('button').classes(
                                        f'seg-btn{"  active" if is_active else ""}'
                                    ).style('font-size:10px;').on('click', make_origin_setter(oid))
                                    with obtn:
                                        ui.html(olabel)
                                    origin_btns[oid] = obtn
                        gcode_origin_row.set_visibility(S.export_format == 'gcode')
                    gcode_row.set_visibility(S.export_format in ('gcode', 'plot_svg'))
                still_only_els.append(export_fmt_area)

                # ── Video export area (dropdown + button always visible in video
                # mode; encoding-specific settings nested inside, shown only once
                # "Video (ProRes .mov)" is picked — they're meaningless for a
                # still-image frame sequence). The dropdown/button must NOT be
                # nested inside the thing they control the visibility of, or
                # picking "Video" becomes impossible once anything else is
                # selected. ──────────────────────────────────────────────────
                video_seq_area = ui.element('div')
                video_seq_area.set_visibility(S.media_mode == 'video')
                video_only_els.append(video_seq_area)
                with video_seq_area:
                    video_export_row = ui.element('div')
                    video_export_row.set_visibility(S.video_seq_format == 'video')
                    with video_export_row:
                        with ui.element('div').style('margin-bottom:4px;'):
                            trans_cb = ui.checkbox('Transparent background', value=S.video_transparent).props(
                                'dense dark color=deep-purple')
                        ui.html('<div style="font-size:10px;color:var(--sub2);padding-left:2px;margin-bottom:10px;">'
                                '<b>ProRes 4444 .mov</b> — intra-frame, essentially artifact-free on flat-color, '
                                'hard-edge mosaic content (no H.264-style compression blockiness). ON adds an '
                                'alpha channel for a transparent background.</div>')

                        def _on_transparent_change(e):
                            is_trans = bool(e.args if not isinstance(e.args, (list, tuple)) else e.args[0])
                            S.video_transparent = is_trans
                        trans_cb.on('update:model-value', _on_transparent_change)

                        ui.html('<div style="font-size:10px;color:var(--sub);margin-bottom:4px;">Colours to render '
                                '<span style="color:var(--sub2);">(click a swatch to hide/show)</span></div>')

                        @ui.refreshable
                        def video_color_toggles():
                            with ui.element('div').style('display:flex;flex-wrap:wrap;gap:5px;'):
                                hidden = S.video_hidden or set()
                                for i, hx in enumerate(S.palette):
                                    is_hidden = i in hidden
                                    def toggle(idx=i):
                                        h = set(S.video_hidden or set())
                                        if idx in h:
                                            h.discard(idx)
                                        else:
                                            h.add(idx)
                                        S.video_hidden = h
                                        video_color_toggles.refresh()
                                        try:
                                            client.run_javascript(
                                                f'window.smaltiSetHidden && window.smaltiSetHidden({sorted(h)})')
                                        except Exception:
                                            pass
                                    # A plain clickable div — a q-btn would override the swatch
                                    # colour with its own background.
                                    sw = ui.element('div').on('click', lambda _=None, t=toggle: t()).style(
                                        f'width:26px;height:26px;border-radius:6px;background:{hx};cursor:pointer;'
                                        f'border:2px solid {"#FA6E9E" if is_hidden else "rgba(255,255,255,0.3)"};'
                                        f'opacity:{"0.35" if is_hidden else "1"};'
                                        f'display:flex;align-items:center;justify-content:center;')
                                    if is_hidden:
                                        with sw:
                                            ui.html('<span style="color:#fff;font-size:15px;font-weight:700;'
                                                    'text-shadow:0 0 3px #000;line-height:1;">×</span>')
                        _vct_refresh['fn'] = video_color_toggles.refresh
                        video_color_toggles()
                        ui.html('<div class="divider" style="margin:10px 0 6px;"></div>')

                    # ── Frame-sequence export (flip-book: one file per frame) ────────
                    ui.html('<div style="font-size:11px;color:var(--sub);font-weight:600;margin-bottom:6px;">'
                            'Export frame sequence</div>')
                    ui.html('<div style="font-size:10px;color:var(--sub2);margin-bottom:8px;">'
                            'Writes every frame as its own still-image file, in its own '
                            'numbered folder — for plotting or printing a physical flip-book '
                            'from the clip.</div>')
                    with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-bottom:8px;'):
                        seq_fmt_select = ui.select(
                            {'video': 'Video (ProRes .mov)',
                             'svg': 'SVG (vector)', 'png': 'PNG', 'jpeg': 'JPEG',
                             'tiff': 'TIFF (layered)',
                             'gcode': 'G-code (pen plotter)',
                             'plot_svg': 'Plot SVG (stroke-only)'},
                            value=S.video_seq_format,
                        ).props('dense outlined dark').style('flex:1;font-size:12px;')

                        with ui.element('div').style('display:flex;align-items:center;gap:8px;') as seq_gcode_origin_row:
                            ui.html('<div style="font-size:11px;color:var(--sub);white-space:nowrap;">'
                                    'Origin (0,0)</div>')
                            with ui.element('div').classes('seg-ctrl'):
                                seq_origin_btns = {}
                                for oid, olabel in _ORIGIN_OPTIONS:
                                    is_active = oid == S.gcode_origin
                                    def make_seq_origin_setter(o):
                                        def setter():
                                            S.gcode_origin = o
                                            for k, b in seq_origin_btns.items():
                                                b.classes(add='active') if k == o else b.classes(remove='active')
                                        return setter
                                    sobtn = ui.element('button').classes(
                                        f'seg-btn{"  active" if is_active else ""}'
                                    ).style('font-size:10px;').on('click', make_seq_origin_setter(oid))
                                    with sobtn:
                                        ui.html(olabel)
                                    seq_origin_btns[oid] = sobtn
                        seq_gcode_origin_row.set_visibility(S.video_seq_format == 'gcode')

                        def on_seq_fmt():
                            S.video_seq_format = seq_fmt_select.value
                            is_video_fmt = S.video_seq_format == 'video'
                            seq_gcode_origin_row.set_visibility(S.video_seq_format == 'gcode')
                            video_export_row.set_visibility(is_video_fmt)
                            seq_export_btn.set_text('⬇ Render video' if is_video_fmt else 'Export sequence')
                        seq_fmt_select.on_value_change(on_seq_fmt)

                    seq_export_btn = ui.button(
                        '⬇ Render video' if S.video_seq_format == 'video' else 'Export sequence'
                    ).style(
                        'font-size:12px;padding:5px 14px;background:var(--accent) !important;'
                        'color:white;border-radius:6px;font-weight:600;width:100%;margin-bottom:4px;'
                    )
                    with ui.element('div').style('display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;'):
                        seq_status_el = [ui.html('').style('display:inline;')]
                        seq_reveal_path = ['']
                        seq_reveal_btn = ui.html('Open in Finder').style(
                            'font-size:11px;color:var(--accent);text-decoration:underline;'
                            'cursor:pointer;white-space:nowrap;'
                        )
                        seq_reveal_btn.on('click', lambda: reveal_in_finder(seq_reveal_path[0]))
                    seq_reveal_btn.set_visibility(False)

                    async def export_sequence():
                        fmt = S.video_seq_format
                        if fmt == 'video':
                            # "Video (ProRes .mov)" isn't a per-frame still-image format —
                            # route straight to the existing, already-working
                            # video encoder instead of the frame-sequence path.
                            await render_video_out()
                            return
                        if fmt in ('gcode', 'plot_svg') and S.mode not in ('circle', 'polygon', 'tile'):
                            seq_status_el[0].set_content(
                                '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                                'G-code / Plot SVG need Circle, Polygon, or Tile render mode '
                                '(Airbrush has no outline to fill).</div>')
                            return
                        if not S.video_json_path or not os.path.exists(S.video_json_path):
                            seq_status_el[0].set_content(
                                '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                                'Process the video first (click Process video).</div>')
                            return
                        if S.video_seq_exporting:
                            seq_status_el[0].set_content(
                                '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                                'A sequence export is already running — please wait for it to finish.</div>')
                            return
                        S.video_seq_exporting = True
                        seq_export_btn.props('loading')
                        seq_reveal_btn.set_visibility(False)
                        try:
                            loop = asyncio.get_event_loop()
                            def prog(idx, total, msg):
                                async def _u():
                                    seq_status_el[0].set_content(
                                        f'<div style="font-size:11px;color:var(--sub);margin-top:6px;">{msg}</div>')
                                asyncio.run_coroutine_threadsafe(_u(), loop)
                            holder = {}
                            def worker():
                                try:
                                    from core.video_sequence_export import export_frame_sequence
                                    hidden = S.video_hidden or set()
                                    visible = [i for i in range(len(S.palette)) if i not in hidden]
                                    out_dir = os.path.join(S.output_dir, f'mosaic_video_{S.mode}_{fmt}_sequence')
                                    holder['r'] = export_frame_sequence(
                                        S.video_json_path, out_dir, fmt,
                                        mode=S.mode,
                                        eq_curve=list(S.airbrush_eq) if S.airbrush_eq else None,
                                        poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                        poly_gradient=S.poly_gradient, visible_colors=visible,
                                        physical_long_edge=S.physical_long_edge, unit=S.unit, dpi=S.dpi,
                                        long_px=int(S.export_long_px),
                                        pen_width_mm=S.pen_width_mm, passes=S.export_json_passes,
                                        gcode_origin=S.gcode_origin, per_layer=S.export_per_layer,
                                        reg_marks=S.reg_marks_enabled, reg_mark_dist=S.reg_mark_dist,
                                        reg_mark_size=S.reg_mark_size, reg_mark_shape=S.reg_mark_shape,
                                        progress_cb=prog,
                                    )
                                except Exception:
                                    import traceback
                                    holder['e'] = traceback.format_exc()
                            t = threading.Thread(target=worker, daemon=True)
                            t.start()
                            while t.is_alive():
                                await asyncio.sleep(0.2)
                            if 'e' in holder:
                                seq_status_el[0].set_content(
                                    f'<div style="font-size:11px;color:#FA6E9E;margin-top:6px;font-family:monospace;'
                                    f'white-space:pre-wrap;">{holder["e"]}</div>')
                                return
                            seq_status_el[0].set_content(
                                f'<div style="font-size:11px;color:var(--green);margin-top:6px;">✓ Frame sequence saved: '
                                f'<span style="color:var(--text);">{holder.get("r")}</span></div>')
                            seq_reveal_path[0] = holder.get('r') or ''
                            seq_reveal_btn.set_visibility(bool(seq_reveal_path[0]))
                        finally:
                            S.video_seq_exporting = False
                            seq_export_btn.props(remove='loading')
                    seq_export_btn.on('click', export_sequence)
                    ui.html('<div class="divider" style="margin:10px 0 6px;"></div>')

                with ui.element('div').style('display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;'):
                    save_btn = ui.button('⬇ Export').style(
                        'font-size:12px;padding:5px 14px;background:var(--accent) !important;'
                        'color:white;border-radius:6px;font-weight:600;'
                    )
                    with ui.element('div').style('display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;'):
                        save_status_el = [ui.html('').style('display:inline;')]
                        save_reveal_path = ['']
                        save_reveal_btn = ui.html('Open in Finder').style(
                            'font-size:11px;color:var(--accent);text-decoration:underline;'
                            'cursor:pointer;white-space:nowrap;'
                        )
                        save_reveal_btn.on('click', lambda: reveal_in_finder(save_reveal_path[0]))
                    save_reveal_btn.set_visibility(False)

                # Toggle: also write one file per palette colour (plus the combined/composite).
                with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-bottom:10px;') as per_layer_row:
                    still_only_els.append(per_layer_row)
                    per_layer_chk = ui.checkbox(value=S.export_per_layer).props('dense color=deep-purple')
                    per_layer_chk.on('update:model-value',
                                     lambda e: setattr(S, 'export_per_layer',
                                                       bool(e.args[0] if isinstance(e.args, (list, tuple)) else e.args)))
                    ui.html('<div style="font-size:11px;color:var(--sub);">Export a separate file per colour '
                            '<span style="color:var(--sub2);">(TIFF is always layered)</span></div>')

                # Toggle: airbrush SVG combined-file splat ordering. Only meaningful for
                # airbrush render mode + SVG export — hidden otherwise.
                with ui.element('div').style('display:flex;align-items:center;gap:8px;margin-bottom:10px;') as airbrush_order_row:
                    still_only_els.append(airbrush_order_row)
                    airbrush_order_row_el[0] = airbrush_order_row
                    airbrush_order_chk = ui.checkbox(value=S.airbrush_svg_unified_order).props('dense color=deep-purple')
                    airbrush_order_chk.on('update:model-value',
                                          lambda e: setattr(S, 'airbrush_svg_unified_order',
                                                            bool(e.args[0] if isinstance(e.args, (list, tuple)) else e.args)))
                    ui.html('<div style="font-size:11px;color:var(--sub);">Size-order splats (small on top) in combined SVG '
                            '<span style="color:var(--sub2);">— off groups splats by colour layer instead</span></div>')
                airbrush_order_row.set_visibility(S.mode == 'airbrush' and S.export_format == 'svg')

                # ── Session save / load ────────────────────────────────────────
                with ui.element('div').style('display:flex;gap:8px;margin-bottom:10px;'):
                    async def save_session_file():
                        default = (os.path.splitext(S.image_name or 'project')[0] or 'project') + '.smalti.json'
                        start = S.output_dir if os.path.isdir(S.output_dir) else str(Path.home())
                        path = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: choose_save_path(default, start))
                        if not path:
                            return
                        if not path.endswith('.json'):
                            path += '.smalti.json'
                        try:
                            data = await asyncio.get_event_loop().run_in_executor(None, serialize_session)
                            with open(path, 'w') as f:
                                json.dump(data, f)
                            ui.notify(f'Session saved: {os.path.basename(path)}', type='positive')
                        except Exception as ex:
                            ui.notify(f'Save failed: {ex}', type='negative')

                    async def load_session_file():
                        start = S.output_dir if os.path.isdir(S.output_dir) else str(Path.home())
                        path = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: choose_open_path(start))
                        if not path:
                            return
                        try:
                            with open(path) as f:
                                data = json.load(f)
                            apply_session(data)
                            ui.notify('Session loaded — reloading…', type='positive')
                            ui.navigate.reload()
                        except Exception as ex:
                            ui.notify(f'Load failed: {ex}', type='negative')

                    ui.button('Save session').on('click', save_session_file).style(
                        'font-size:12px;padding:5px 12px;background:var(--accent) !important;'
                        'border:1px solid var(--accent);border-radius:6px;color:white;')
                    ui.button('Load session').on('click', load_session_file).style(
                        'font-size:12px;padding:5px 12px;background:var(--accent) !important;'
                        'border:1px solid var(--accent);border-radius:6px;color:white;')

                results_el[0] = ui.element('div')

                async def render_video_out():
                    from core.video_render import render_video
                    if not S.video_json_path or not os.path.exists(S.video_json_path):
                        seq_status_el[0].set_content(
                            '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                            'Process the video first (click Process video).</div>')
                        return
                    # Re-entrancy guard: this check-and-set is synchronous (no `await`
                    # between them), so it's atomic against the single-threaded asyncio
                    # event loop even if a double-click already queued a second call
                    # before the button's "loading" prop round-trips back to the browser.
                    # Without it, two ffmpeg processes would write the same output file
                    # concurrently and interleave into one corrupted .mp4.
                    if S.video_exporting:
                        seq_status_el[0].set_content(
                            '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                            'A video export is already running — please wait for it to finish.</div>')
                        return
                    S.video_exporting = True
                    S.job_kind = 'video_export'
                    S.last_error = None
                    seq_export_btn.props('loading')
                    seq_reveal_btn.set_visibility(False)
                    try:
                        loop = asyncio.get_event_loop()
                        def prog(idx, total, msg):
                            S.progress_msg = msg
                            async def _u():
                                seq_status_el[0].set_content(
                                    f'<div style="font-size:11px;color:var(--sub);margin-top:6px;">{msg}</div>')
                            asyncio.run_coroutine_threadsafe(_u(), loop)
                        holder = {}
                        def worker():
                            try:
                                hidden = S.video_hidden or set()
                                visible = [i for i in range(len(S.palette)) if i not in hidden]
                                out_path = os.path.join(S.output_dir, f'mosaic_video_{S.mode}.mov')
                                holder['r'] = render_video(
                                    S.video_json_path, out_path,
                                    out_long_px=int(S.physical_long_edge) if S.unit == 'px' else 0,
                                    mode=S.mode,
                                    eq_curve=list(S.airbrush_eq) if S.airbrush_eq else None,
                                    poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                    poly_gradient=S.poly_gradient,
                                    visible_colors=visible, transparent=S.video_transparent,
                                    reg_marks=S.reg_marks_enabled, reg_mark_dist=S.reg_mark_dist,
                                    reg_mark_size=S.reg_mark_size, reg_mark_shape=S.reg_mark_shape,
                                    physical_long_edge=S.physical_long_edge,
                                    progress_cb=prog)
                            except Exception:
                                import traceback
                                holder['e'] = traceback.format_exc()
                                S.last_error = holder['e']   # durable: survives a page reload
                        t = threading.Thread(target=worker, daemon=True)
                        S.worker_thread = t
                        t.start()
                        while t.is_alive():
                            await asyncio.sleep(0.2)
                        if 'e' in holder:
                            seq_status_el[0].set_content(
                                f'<div style="font-size:11px;color:#FA6E9E;margin-top:6px;font-family:monospace;'
                                f'white-space:pre-wrap;">{holder["e"]}</div>')
                            return
                        kind = 'ProRes 4444 .mov (alpha)' if S.video_transparent else 'ProRes 4444 .mov'
                        S.video_export_path = holder.get('r') or ''
                        seq_status_el[0].set_content(
                            f'<div style="font-size:11px;color:var(--green);margin-top:6px;">✓ {kind} saved '
                            f'({S.mode}): <span style="color:var(--text);">{holder.get("r")}</span></div>')
                        seq_reveal_path[0] = S.video_export_path
                        seq_reveal_btn.set_visibility(bool(seq_reveal_path[0]))
                    finally:
                        S.video_exporting = False
                        S.job_kind = ''
                        S.worker_thread = None
                        seq_export_btn.props(remove='loading')

                async def save_svg():
                    if S.media_mode == 'video':
                        await render_video_out()
                        return
                    R = S.last_result
                    if not R:
                        ui.notify('Run the pipeline first.', type='warning')
                        return
                    save_btn.props('loading')
                    save_status_el[0].set_content('')
                    save_reveal_btn.set_visibility(False)
                    try:
                        W, H = R.working_size
                        eq = list(S.airbrush_eq) if S.airbrush_eq else [1.2] * 5
                        fmt = S.export_format
                        if fmt == 'svg':
                            from core.svg_writer import write_all_outputs
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_all_outputs(
                                    R.layer_circles, S.canvas_hex, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit, S.dpi, S.mode,
                                    r_min=R.r_min_px, r_max=R.r_max_px, eq_curve=eq,
                                    poly_inset=S.poly_inset, poly_gradient=S.poly_gradient,
                                    poly_smooth=S.poly_smooth, per_layer=S.export_per_layer,
                                    airbrush_unified_order=S.airbrush_svg_unified_order,
                                    reg_marks=S.reg_marks_enabled, reg_mark_dist=S.reg_mark_dist,
                                    reg_mark_size=S.reg_mark_size, reg_mark_shape=S.reg_mark_shape,
                                )
                            )
                        elif fmt == 'gcode':
                            if S.mode not in ('circle', 'polygon', 'tile'):
                                save_status_el[0].set_content(
                                    '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                                    'G-code needs Circle, Polygon, or Tile render mode (Airbrush has no outline to fill).</div>')
                                save_btn.props(remove='loading')
                                return
                            from core.gcode_export import write_gcode
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_gcode(
                                    R.layer_circles, S.canvas_hex, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit, S.dpi, S.mode,
                                    S.pen_width_mm, poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                    r_min=R.r_min_px, r_max=R.r_max_px, passes=S.export_json_passes,
                                    per_layer=S.export_per_layer, origin=S.gcode_origin,
                                    reg_marks=S.reg_marks_enabled, reg_mark_dist=S.reg_mark_dist,
                                    reg_mark_size=S.reg_mark_size, reg_mark_shape=S.reg_mark_shape,
                                )
                            )
                        elif fmt == 'plot_svg':
                            if S.mode not in ('circle', 'polygon', 'tile'):
                                save_status_el[0].set_content(
                                    '<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">'
                                    'Plot SVG needs Circle, Polygon, or Tile render mode (Airbrush has no outline to fill).</div>')
                                save_btn.props(remove='loading')
                                return
                            from core.plot_svg_export import write_plot_svg
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_plot_svg(
                                    R.layer_circles, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit, S.mode,
                                    S.pen_width_mm, poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                    r_min=R.r_min_px, r_max=R.r_max_px, passes=S.export_json_passes,
                                    per_layer=S.export_per_layer, canvas_hex=S.canvas_hex,
                                    reg_marks=S.reg_marks_enabled, reg_mark_dist=S.reg_mark_dist,
                                    reg_mark_size=S.reg_mark_size, reg_mark_shape=S.reg_mark_shape,
                                )
                            )
                        elif fmt == 'ply':
                            from core.model3d_export import write_pointcloud
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_pointcloud(
                                    R.layer_circles, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit,
                                    layout=S.export_3d_layout, depth=S.export_3d_depth,
                                    canvas_hex=S.canvas_hex,
                                )
                            )
                        elif fmt == 'obj':
                            from core.model3d_export import write_mesh_solids
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_mesh_solids(
                                    R.layer_circles, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit,
                                    r_min=R.r_min_px, r_max=R.r_max_px,
                                    # Roundness ← Curve smoothing, spacing ← Cell outline ratio,
                                    # so the solids match the on-screen polygon mosaic.
                                    round_amt=S.poly_smooth, relief=S.export_3d_relief,
                                    poly_inset=S.poly_inset,
                                    layout=S.export_3d_layout, scatter=S.export_3d_depth,
                                )
                            )
                        elif fmt == 'json':
                            from core.cnc_export import write_cnc_json
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_cnc_json(
                                    R.layer_circles, S.canvas_hex, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit, S.dpi,
                                    passes=S.export_json_passes,
                                )
                            )
                        elif fmt == 'splat':
                            from core.model3d_export import write_gaussian_splat
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_gaussian_splat(
                                    R.layer_circles, S.output_dir,
                                    W, H, S.physical_long_edge, S.unit,
                                    layout=S.export_3d_layout, depth=S.export_3d_depth,
                                )
                            )
                        else:
                            from core.raster_export import write_raster_outputs
                            paths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_raster_outputs(
                                    R.layer_circles, S.canvas_hex, S.output_dir,
                                    W, H, int(S.export_long_px), fmt, S.mode,
                                    r_min=R.r_min_px, r_max=R.r_max_px, eq_curve=eq,
                                    poly_inset=S.poly_inset, poly_smooth=S.poly_smooth,
                                    poly_gradient=S.poly_gradient,
                                    per_layer=S.export_per_layer,
                                    reg_marks=S.reg_marks_enabled,
                                    reg_mark_dist_px=(S.reg_mark_dist / S.physical_long_edge * int(S.export_long_px))
                                                     if (S.reg_marks_enabled and S.physical_long_edge) else 0.0,
                                    reg_mark_size_px=(S.reg_mark_size / S.physical_long_edge * int(S.export_long_px))
                                                     if (S.reg_marks_enabled and S.physical_long_edge) else 0.0,
                                    reg_mark_shape=S.reg_mark_shape,
                                )
                            )
                        # For any 3D export, also write a Blender setup script that
                        # imports the file and adds a camera aligned to resolve the image.
                        if fmt in ('ply', 'obj', 'splat'):
                            from core.model3d_export import write_blender_setup
                            bpaths = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: write_blender_setup(
                                    S.output_dir, fmt, S.export_3d_layout,
                                    W, H, S.physical_long_edge,
                                    canvas_hex=S.canvas_hex,
                                )
                            )
                            paths = paths + bpaths
                        file_pills = ''.join(
                            f'<div class="file-pill">{Path(p).name}</div>'
                            for p in paths
                        )
                        save_status_el[0].set_content(
                            f'<div style="margin-top:10px;">'
                            f'<div style="font-size:11px;color:var(--sub);margin-bottom:6px;">Files written ({S.mode} mode):</div>'
                            f'{file_pills}'
                            f'<div style="font-size:11px;color:var(--sub);margin-top:8px;">'
                            f'Folder: <span style="color:var(--text);">{S.output_dir}</span></div>'
                            f'</div>'
                        )
                        save_reveal_path[0] = paths[0] if paths else S.output_dir
                        save_reveal_btn.set_visibility(bool(save_reveal_path[0]))
                    except Exception as ex:
                        save_status_el[0].set_content(
                            f'<div style="font-size:11px;color:#FA6E9E;margin-top:6px;">Error: {ex}</div>'
                        )
                    finally:
                        save_btn.props(remove='loading')

                save_btn.on('click', save_svg)

                # Initialise media-mode UI (hides video options, sets button labels).
                update_media_ui()

                def show_results(R: PipelineResult):
                    results_el[0].clear()
                    with results_el[0]:
                        ui.html(
                            f'<div style="font-size:12px;color:var(--sub);margin-bottom:12px;">'
                            f'Original: {R.original_size[0]}×{R.original_size[1]}px &nbsp;·&nbsp; '
                            f'Working: {R.working_size[0]}×{R.working_size[1]}px</div>'
                        )
                        for i, (circles, hx, _) in enumerate(R.layer_circles):
                            with ui.element('div').classes('layer-row'):
                                ui.element('div').classes('swatch').style(f'background:{hx};')
                                ui.html(
                                    f'<span style="font-size:12px;">Layer {i+1} &nbsp;</span>'
                                    f'<span style="font-size:12px;color:var(--sub);">{hx}</span>'
                                )
                                ui.html(
                                    f'<span style="margin-left:auto;font-size:12px;'
                                    f'color:var(--accent);font-variant-numeric:tabular-nums;">'
                                    f'{len(circles):,} circles</span>'
                                )
                        save_status_el[0].set_content('')

                # ── Restore a loaded session ───────────────────────────────────
                # After apply_session() + reload, S.result holds the restored circle
                # data.  Repopulate the layer list now and push the preview once the
                # page connects (updatePaintbot retries until PixiJS is ready).
                if S.result:
                    show_results(S.result)
                    status_el[0].set_content(
                        f'<div style="font-size:13px;color:var(--green);">✓ Loaded — '
                        f'{S.result.total_circles:,} circles across {len(S.result.layer_circles)} layers.</div>'
                    )

                    async def _restore_preview():
                        # Rebuilding the preview in Tile mode runs the full-resolution tile
                        # layout (tens of seconds on a large piece). It MUST run off the event
                        # loop — computed inline it froze the whole UI (clicks ignored,
                        # websocket JS calls timing out) for the entire computation on every
                        # page load or session restore. Pushed fire-and-forget (no awaited
                        # round-trip, whose 1 s timeout a big payload can exceed).
                        try:
                            payload = await asyncio.get_event_loop().run_in_executor(
                                None, _preview_json_from_result, S.result)
                            client.run_javascript(f'window.updatePaintbot({payload})')
                        except Exception as e:
                            print(f"[restore preview] failed: {e!r}", flush=True)
                    ui.timer(0.4, _restore_preview, once=True)

                # ── Reattach to an in-progress (or just-finished) job ───────────────
                # If a PREVIOUS page connection started a generation and that connection
                # died before the job finished (e.g. the native window's WebView renderer
                # got reclaimed while backgrounded and reloaded), the background worker
                # thread keeps running invisibly — its own progress callbacks and result
                # holders were closures over the now-dead connection, with no way to reach
                # this fresh page. S.processing / S.worker_thread / S.progress_* /
                # S.last_error / S.result are the durable, server-side record this page
                # uses instead of silently showing a blank "not started" form. Mutually
                # exclusive with the "Restore a loaded session" block above: if the job
                # already finished by the time this page loaded, S.processing is already
                # False and S.result was already shown up there.
                if S.job_kind == 'video_process' and S.processing and S.worker_thread is not None:
                    status_el[0].set_content(
                        '<div style="font-size:13px;color:var(--sub);">Reattaching to video '
                        'processing already in progress (started from another view)…</div>')
                    run_btn_set_busy()
                    cancel_btn.set_visibility(True)
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(add='locked')

                    async def _reattach_video_process():
                        wt = S.worker_thread
                        while wt is not None and wt.is_alive():
                            status_el[0].set_content(
                                f'<div style="font-size:13px;color:var(--sub);">'
                                f'{S.progress_msg or "Working…"}</div>')
                            progress_fill_el[0].style(f'width:{S.progress_pct * 100:.1f}%;')
                            await asyncio.sleep(0.5)
                        S.processing = False
                        S.job_kind = ''
                        S.worker_thread = None
                        cancel_btn.set_visibility(False)
                        run_btn_set_idle()
                        if mode_ctrl_el[0]:
                            mode_ctrl_el[0].classes(remove='locked')
                        if S.last_error:
                            status_el[0].set_content(
                                f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;'
                                f'white-space:pre-wrap;">Error: {S.last_error}</div>')
                            return
                        progress_fill_el[0].style('width:100%;')
                        count = len(_SCRUB['frames'])
                        if count > 0:
                            scrubber_row.style('display:flex;')
                            try:
                                await client.run_javascript(f'window.smaltiSetCount({count})')
                            except Exception:
                                pass
                        status_el[0].set_content(
                            f'<div style="font-size:13px;color:var(--green);">✓ {count} frames processed — '
                            'scrub / ▶ Play below to review the movement, or <b>⬇ Render video</b> to encode.</div>')
                    ui.timer(0.1, _reattach_video_process, once=True)
                elif S.job_kind == 'video_export' and S.video_exporting and S.worker_thread is not None:
                    save_status_el[0].set_content(
                        '<div style="font-size:11px;color:var(--sub);margin-top:6px;">Reattaching to a '
                        'video export already in progress (started from another view)…</div>')
                    save_btn.props('loading')

                    async def _reattach_video_export():
                        wt = S.worker_thread
                        while wt is not None and wt.is_alive():
                            save_status_el[0].set_content(
                                f'<div style="font-size:11px;color:var(--sub);margin-top:6px;">'
                                f'{S.progress_msg or "Encoding…"}</div>')
                            await asyncio.sleep(0.5)
                        S.video_exporting = False
                        S.job_kind = ''
                        S.worker_thread = None
                        save_btn.props(remove='loading')
                        if S.last_error:
                            save_status_el[0].set_content(
                                f'<div style="font-size:11px;color:#FA6E9E;margin-top:6px;font-family:monospace;'
                                f'white-space:pre-wrap;">{S.last_error}</div>')
                            return
                        kind = 'ProRes 4444 .mov (alpha)' if S.video_transparent else 'ProRes 4444 .mov'
                        save_status_el[0].set_content(
                            f'<div style="font-size:11px;color:var(--green);margin-top:6px;">✓ {kind} saved '
                            f'({S.mode}): <span style="color:var(--text);">{S.video_export_path}</span></div>')
                        save_reveal_path[0] = S.video_export_path
                        save_reveal_btn.set_visibility(bool(save_reveal_path[0]))
                    ui.timer(0.1, _reattach_video_export, once=True)
                elif S.processing and S.worker_thread is not None:
                    status_el[0].set_content(
                        '<div style="font-size:13px;color:var(--sub);">Reattaching to a '
                        'generation already in progress (started from another view)…</div>')
                    run_btn_set_busy()
                    cancel_btn.set_visibility(True)
                    if mode_ctrl_el[0]:
                        mode_ctrl_el[0].classes(add='locked')

                    async def _reattach():
                        wt = S.worker_thread
                        while wt is not None and wt.is_alive():
                            status_el[0].set_content(
                                f'<div style="font-size:13px;color:var(--sub);">'
                                f'{S.progress_msg or "Working…"}</div>')
                            progress_fill_el[0].style(f'width:{S.progress_pct * 100:.1f}%;')
                            await asyncio.sleep(0.5)
                        S.processing = False
                        S.job_kind = ''
                        S.worker_thread = None
                        cancel_btn.set_visibility(False)
                        run_btn_set_idle()
                        if mode_ctrl_el[0]:
                            mode_ctrl_el[0].classes(remove='locked')
                        if S.last_error:
                            status_el[0].set_content(
                                f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;'
                                f'white-space:pre-wrap;">Error: {S.last_error}</div>')
                            return
                        if S.result:
                            progress_fill_el[0].style('width:100%;')
                            show_results(S.result)
                            status_el[0].set_content(
                                f'<div style="font-size:13px;color:var(--green);">✓ Done — '
                                f'{S.result.total_circles:,} circles across '
                                f'{len(S.result.layer_circles)} layers.</div>')
                            try:
                                payload = await asyncio.get_event_loop().run_in_executor(
                                    None, _preview_json_from_result, S.result)
                                client.run_javascript(f'window.updatePaintbot({payload})')
                            except Exception as e:
                                print(f"[reattach preview] failed: {e!r}", flush=True)
                    ui.timer(0.1, _reattach, once=True)
                elif S.last_error:
                    # Job already ENDED (with an error) before this page ever loaded —
                    # S.processing is already False by now, so neither block above fires.
                    # Without this, a failed job that finished between the old page dying
                    # and this fresh page loading shows a misleadingly blank "not started"
                    # form instead of the actual error.
                    status_el[0].set_content(
                        f'<div style="font-size:12px;color:#FA6E9E;font-family:monospace;'
                        f'white-space:pre-wrap;">Error: {S.last_error}</div>')



# When launched from the packaged Smalti.app (SMALTI_APP=1) run in a NATIVE desktop
# window (pywebview) instead of a browser tab.  Direct `python main.py` still opens
# a normal browser for development.  reload is always off — the uvicorn reloader
# spawns child processes that can wedge the port on shutdown.
if _APP_MODE:
    # multiprocessing's "spawn" start method re-executes this ENTIRE script from
    # scratch in the --multiprocessing-fork child (this whole `if _APP_MODE:` block
    # runs again there too), and empirically even a no-op call (nothing to kill,
    # PID file didn't exist yet) still destabilized startup — the exact mechanism
    # wasn't pinned down in the time available, so this is disabled rather than
    # shipped in a state that isn't understood. _kill_stale_instances() is left
    # defined below for whoever picks this back up.
    from nicegui import native as _native
    ui.run(
        title='Smalti',
        dark=True,
        port=_native.find_open_port(),   # dynamic port for the native/packaged app
        favicon=str(_LOGO_PATH) if _LOGO_PATH.exists() else '◉',
        native=True,
        window_size=(1500, 950),
        show=False,
        reload=False,
    )
else:
    ui.run(
        title='Smalti',
        dark=True,
        port=8484,
        favicon=str(_LOGO_PATH) if _LOGO_PATH.exists() else '◉',
        show=True,
        reload=False,
    )
