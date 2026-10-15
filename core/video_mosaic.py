"""
Video → mosaic, stage 1 of 2.

Turns an input video into a *frame sequence of Smalti mosaics*, written to a compact
JSON file (``mosaic_video.json``).  This is the slow, one-time processing stage; the
fast stage 2 (:mod:`core.video_render`) turns that JSON into an actual video file.

Temporal coherence: the palette is built once from frames sampled across the whole clip
so colours stay stable, and each frame WARM-STARTS from the previous frame's dots — the
optimiser re-settles the existing dots against the new frame instead of re-seeding from
scratch.  Dots therefore evolve continuously (with some intentional boil/flicker) and the
dot count stays fixed, so every dot is one trajectory across the clip.

JSON schema:
    {
      "app": "smalti", "format": "mosaic-video", "version": 1,
      "fps": 24.0, "mode": "polygon",
      "working_size": [W, H],          # px the geometry lives in
      "palette": ["#..", ...], "canvas": "#FFFFFF",
      "r_min": <px>, "r_max": <px>, "poly_inset": .., "poly_smooth": ..,
      "count": <dots per frame>,       # null if it varies (it shouldn't for warm-start)
      "frames": [ {"x":[..],"y":[..],"r":[..],"c":[..]}, ... ]
    }
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .image_processor import rgb_to_lab
from .palette_gen import _lab_to_hex, palette_from_lab


def build_clip_palette(cap, n_colors, sample_frames=12, max_px=400):
    """Sample frames spread across the clip and k-means a single LAB palette."""
    import cv2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), num=min(sample_frames, total)).astype(int)
    pool = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = max(h, w) / max_px
        if s > 1:
            rgb = cv2.resize(rgb, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
        lab = rgb_to_lab(rgb.astype(np.float32) / 255.0).reshape(-1, 3)
        pool.append(lab)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not pool:
        return ['#000000']
    lab = np.concatenate(pool, axis=0).astype(np.float32)
    return [_lab_to_hex(c) for c in palette_from_lab(lab, n_colors)]


def _flatten(layer_results):
    """layer_results (grouped by colour) → parallel x, y, r, c lists (c = colour index)."""
    xs, ys, rs, cs = [], [], [], []
    for li, (circles, _hex, _name) in enumerate(layer_results):
        for (x, y, r) in circles:
            xs.append(float(x)); ys.append(float(y)); rs.append(float(r)); cs.append(li)
    return xs, ys, rs, cs


def process_video(
    video_path: str,
    output_dir: str,
    *,
    palette=None,            # explicit list of hex; if None, auto-built from the clip
    n_colors: int = 8,
    canvas_hex: str = '#FFFFFF',
    mode: str = 'polygon',
    r_min_px: float = 3.0,
    r_max_px: float = 14.0,
    gap_px: float = 1.0,
    max_working_px: int = 900,
    iters_first: int = 8,    # optimisation iterations on the very first frame
    iters_warm: int = 3,     # iterations per subsequent (warm-started) frame
    fresh: bool = True,      # True = independent full process each frame (no warm-start):
                             #        true per-frame optimisation, more flicker.
                             # False = warm-start each frame from the previous (coherent,
                             #        but arrangements persist between frames).
    frame_stride: int = 1,   # process every Nth frame
    max_frames: int = 0,     # 0 = whole clip
    poly_inset: float = 0.08,
    poly_smooth: float = 0.0,
    airbrush_eq=None,
    canvas_comp: float = 0.0,
    lum_influence: float = 0.5,
    detail_boost: float = 1.2,
    progress_cb=None,        # (frame_index, total, message) -> None
    preview_cb=None,         # (payload_json_str) -> None — one call per finished frame
    stop_event=None,
) -> str:
    """Run stage 1 and write ``mosaic_video.json``.  Returns the JSON path."""
    import cv2
    from .pipeline import run_pipeline, PipelineConfig

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if palette is None:
        if progress_cb:
            progress_cb(0, total, "Building clip palette…")
        palette = build_clip_palette(cap, n_colors)

    def make_cfg(n_iters):
        return PipelineConfig(
            image_path='<frame>',
            max_working_px=max_working_px,
            canvas_hex=canvas_hex,
            palette_hex=list(palette),
            r_min_physical=r_min_px, r_max_physical=r_max_px,
            unit='px', physical_long_edge=0.0, dpi=96.0,   # long_edge set per-frame below
            gap_physical=gap_px,
            canvas_comp=canvas_comp,
            detail_boost=detail_boost,
            lum_influence=lum_influence,
            # Geometry is render-mode-independent, so always process as circles — this
            # avoids the pointless (and very slow) per-frame Voronoi that polygon mode
            # would otherwise compute.  The chosen render mode is applied at export.
            mode='circle',
            poly_inset=poly_inset,
            n_iterations=n_iters,
        )

    frames = []
    working_size = None
    prev_xyr = None
    fixed_count = None
    processed = 0
    fi = 0
    _prev_frame_hash = None
    while True:
        if stop_event is not None and stop_event.is_set():
            break

        # Read the next frame, skipping any that are byte-identical to the one
        # we just processed. OpenCV's ffmpeg backend can occasionally decode
        # the exact same frame twice in a row on variable-frame-rate source
        # video (common from phone/screen recordings) — a plain sequential
        # cap.read() doesn't always land on a genuinely new frame, which makes
        # the mosaic freeze on a "stuck" frame instead of advancing and shows
        # up as repeated frames in the exported video. Bounded (30 retries) so
        # a source clip that's genuinely static for a moment doesn't spin.
        ok = False
        for _ in range(31):   # 1 initial read + up to 30 duplicate-skip retries
            if frame_stride > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_hash = hashlib.blake2b(frame_bgr.tobytes(), digest_size=8).digest()
            if frame_hash != _prev_frame_hash:
                _prev_frame_hash = frame_hash
                break
            fi += 1   # duplicate decode — advance and retry
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        # fresh → full independent process every frame; warm → carry the previous frame.
        init = None if fresh else prev_xyr
        # Per-frame iteration count.  Fresh: every frame uses iters_warm (the "per frame"
        # control).  Warm: the first frame gets a fuller pass (iters_first), later frames
        # use iters_warm.
        if fresh:
            n_iters = iters_warm
        else:
            n_iters = iters_first if init is None else iters_warm
        cfg = make_cfg(n_iters)
        # px unit: set physical_long_edge = the working long edge so 1 working-px maps to
        # 1 px and r_min/r_max are interpreted directly in working pixels.
        cfg.physical_long_edge = float(min(max(W0, H0), max_working_px))

        total_out = min(total // max(frame_stride, 1), max_frames) if max_frames else (total // max(frame_stride, 1))
        frame_no = processed + 1
        if progress_cb:
            progress_cb(processed, total_out or frame_no, f"Processing frame {frame_no}…")

        # Forward the pipeline's preview to the UI, tagged with the current frame.
        # For the (slow, full-pipeline) FIRST frame we forward every step so it builds
        # live and never looks frozen.  For subsequent warm-start frames we forward only
        # the FINAL, fully-optimised state — otherwise the "initial" preview shows the
        # previous frame's dots at their old sizes (before this frame re-optimises them),
        # which reads as dots being mis-scaled.
        def _fwd(payload):
            if not preview_cb:
                return
            try:
                d = json.loads(payload)
                if frame_no > 1 and d.get('phase') != 'final':
                    return
                d['frame'] = frame_no
                d['total'] = total_out or frame_no
                preview_cb(json.dumps(d))
            except Exception:
                preview_cb(payload)

        result = run_pipeline(
            cfg,
            image_array=rgb,
            init_circles=init,
            preview_cb=_fwd if preview_cb else None,
            stop_event=stop_event,
        )
        if working_size is None:
            working_size = list(result.working_size)

        xs, ys, rs, cs = _flatten(result.layer_circles)
        if fixed_count is None:
            fixed_count = len(xs)
        elif len(xs) != fixed_count:
            fixed_count = None   # count drifted (shouldn't with warm-start, but be honest)
        frames.append({
            'x': [round(v, 2) for v in xs],
            'y': [round(v, 2) for v in ys],
            'r': [round(v, 2) for v in rs],
            'c': cs,
        })
        prev_xyr = list(zip(xs, ys, rs))

        processed += 1
        if progress_cb:
            progress_cb(processed, total_out or processed,
                        f"Frame {processed} done: {len(xs):,} dots")
        if max_frames and processed >= max_frames:
            break
        fi += frame_stride
    cap.release()

    out_fps = fps / max(frame_stride, 1)
    data = {
        'app': 'smalti', 'format': 'mosaic-video', 'version': 1,
        'fps': round(out_fps, 4), 'mode': mode,
        'working_size': working_size or [0, 0],
        'palette': list(palette), 'canvas': canvas_hex,
        'r_min': r_min_px, 'r_max': r_max_px,
        'poly_inset': poly_inset, 'poly_smooth': poly_smooth,
        'airbrush_eq': list(airbrush_eq) if airbrush_eq else None,
        'count': fixed_count,
        'frame_count': len(frames),
        'frames': frames,
    }
    p = out / 'mosaic_video.json'
    with open(p, 'w') as f:
        json.dump(data, f)
    return str(p)
