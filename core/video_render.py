"""
Mosaic frame sequence → video, stage 2 of 2.

Reads a ``mosaic_video.json`` produced by :mod:`core.video_mosaic` and renders every
frame's mosaic into output the editor can use:

  • ProRes 4444 .mov  — via ffmpeg when available, else an OpenCV H.264 fallback.
    ProRes is intra-frame and not block/DCT-quantized the way H.264 is, so flat-color,
    hard-edge mosaic content (exactly what this app renders) comes out essentially
    artifact-free — the compression-blockiness problem H.264 needs constant tuning to
    avoid just doesn't apply to it. 4444 also carries an alpha channel, so the same
    codec path covers the transparent-background case as a single .mov instead of a
    PNG sequence.
  • PNG image sequence — lossless fallback for transparent export when ffmpeg isn't
    available. Imports into DaVinci Resolve as an image sequence.

The dot geometry is render-mode-independent, so mode / eq_curve / poly_* / visible
colours / alpha are all applied here — the same processed clip can be re-rendered any
number of ways without reprocessing.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .raster_export import render_mosaic_bgr


def _find_ffmpeg():
    """Locate ffmpeg.  Prefer the copy bundled into the frozen .app (see
    Smalti.spec) so video export works regardless of what's installed on the
    machine running it. A .app launched from Finder also has a minimal PATH
    that omits Homebrew, so fall back to probing the usual install locations
    for source/dev runs."""
    if getattr(sys, 'frozen', False):
        bundled = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'ffmpeg'
        if bundled.exists():
            return str(bundled)
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    for p in ('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg'):
        if Path(p).exists():
            return p
    return None


def _frame_layers(frame, palette, visible=None):
    """Rebuild per-colour layer_results from a frame's parallel x/y/r/c arrays.
    `visible`: optional set/list of colour indices to include (others dropped)."""
    xs, ys, rs, cs = frame['x'], frame['y'], frame['r'], frame['c']
    buckets = [[] for _ in palette]
    for x, y, r, c in zip(xs, ys, rs, cs):
        if 0 <= c < len(palette) and (visible is None or c in visible):
            buckets[c].append((x, y, r))
    return [(buckets[i], palette[i], palette[i]) for i in range(len(palette))]


def render_video(
    json_path: str,
    output_path: str,
    *,
    out_long_px: int = 0,    # 0 = use the geometry's working long-edge (1:1)
    fps: float = 0.0,        # 0 = use fps stored in the JSON
    mode=None,               # override render mode ('circle'|'airbrush'|'polygon')
    eq_curve=None,           # override airbrush EQ curve
    poly_inset=None,         # override polygon cell inset ratio
    poly_smooth=None,        # override polygon curve smoothing
    poly_gradient=None,      # override polygon colour-gradient toggle
    visible_colors=None,     # iterable of palette indices to include (None = all)
    transparent=False,       # True → keep an alpha channel (ProRes 4444) / transparent background
    reg_marks=False,
    reg_mark_dist=0.0,       # in `unit`, same as physical_long_edge (a still export would use)
    reg_mark_size=0.0,
    reg_mark_shape='circle',
    physical_long_edge=0.0,  # only needed to scale reg_mark_dist/size — 0 disables marks
                              # (same value/unit S.physical_long_edge already carries)
    progress_cb=None,        # (frame_index, total, message) -> None
    stop_event=None,
) -> str:
    """Run stage 2.  Returns the output path (a .mov, or a folder of PNGs when
    transparent and ffmpeg isn't available)."""
    import cv2

    data = json.loads(Path(json_path).read_text())
    palette = data['palette']
    canvas_hex = data.get('canvas', '#FFFFFF')
    mode = mode or data.get('mode', 'polygon')
    Wg, Hg = data.get('working_size', [0, 0])
    r_min = data.get('r_min', 1.0)
    r_max = data.get('r_max', 10.0)
    poly_inset = data.get('poly_inset', 0.08) if poly_inset is None else poly_inset
    poly_smooth = data.get('poly_smooth', 0.0) if poly_smooth is None else poly_smooth
    poly_gradient = data.get('poly_gradient', False) if poly_gradient is None else poly_gradient
    eq_curve = eq_curve if eq_curve is not None else data.get('airbrush_eq')
    frames = data['frames']
    out_fps = fps or data.get('fps', 24.0)
    visible = set(visible_colors) if visible_colors is not None else None

    if not frames or Wg <= 0 or Hg <= 0:
        raise RuntimeError("mosaic_video.json has no renderable frames")

    # Output resolution (kept in aspect, forced even for codec friendliness).
    s = (out_long_px / float(max(Wg, Hg))) if out_long_px else 1.0
    out_w = max(2, int(round(Wg * s)) & ~1)
    out_h = max(2, int(round(Hg * s)) & ~1)

    reg_mark_dist_px = reg_mark_size_px = 0.0
    if reg_marks and physical_long_edge:
        reg_mark_dist_px = reg_mark_dist / physical_long_edge * out_w
        reg_mark_size_px = reg_mark_size / physical_long_edge * out_w

    def render_frame(frame):
        layers = _frame_layers(frame, palette, visible)
        return render_mosaic_bgr(
            layers, canvas_hex, Wg, Hg, out_w, out_h,
            mode=mode, r_min=r_min, r_max=r_max, eq_curve=eq_curve,
            poly_inset=poly_inset, poly_smooth=poly_smooth, poly_gradient=poly_gradient,
            alpha=transparent,
            reg_marks=reg_marks, reg_mark_dist_px=reg_mark_dist_px,
            reg_mark_size_px=reg_mark_size_px, reg_mark_shape=reg_mark_shape,
        )

    n = len(frames)
    ffmpeg = _find_ffmpeg()

    # ── No ffmpeg: fall back to the old zero-dependency paths ──────────────────
    if not ffmpeg:
        if transparent:
            folder = Path(output_path)
            if folder.suffix:                      # strip .mov etc → a folder
                folder = folder.with_suffix('')
            folder.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                if stop_event is not None and stop_event.is_set():
                    break
                bgra = render_frame(frame)          # (H, W, 4)
                cv2.imwrite(str(folder / f'frame_{i:05d}.png'), bgra)
                if progress_cb:
                    progress_cb(i + 1, n, f"PNG frame {i + 1}/{n}")
            return str(folder)

        out = Path(output_path)
        if out.suffix.lower() != '.mp4':
            out = out.with_suffix('.mp4')
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'avc1'), out_fps, (out_w, out_h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"could not open a video writer for {out}")
        try:
            for i, frame in enumerate(frames):
                if stop_event is not None and stop_event.is_set():
                    break
                writer.write(render_frame(frame))
                if progress_cb:
                    progress_cb(i + 1, n, f"Rendered frame {i + 1}/{n}")
        finally:
            writer.release()
        return str(out)

    # ── ffmpeg available: ProRes 4444 .mov, alpha or not ────────────────────────
    # ProRes is intra-frame (no interframe prediction to desync) and uses much
    # gentler quantization than H.264's DCT blocks, so it doesn't need any of the
    # level/tune/profile tuning H.264 needed to avoid blocky artifacts on flat-
    # color, hard-edge mosaic content — it's just clean by construction. 4444
    # also carries a real alpha channel, so the transparent case is the same
    # encode with a different pixel format instead of a separate PNG-sequence path.
    out = Path(output_path)
    if out.suffix.lower() != '.mov':
        out = out.with_suffix('.mov')
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write every frame to a lossless PNG in a temp folder first, then encode the
    # whole sequence in one ffmpeg pass — rather than streaming raw BGR(A) bytes
    # through a pipe into ffmpeg's stdin.
    #
    # The raw-pipe approach has no per-frame framing at all: ffmpeg's rawvideo
    # demuxer just consumes a flat byte stream at a fixed stride. If literally
    # anything perturbs the stream by even one byte for one frame — a hiccup in
    # the pipe, memory pressure, anything — every frame after that point decodes
    # from the wrong offset for the rest of the video, which looks exactly like
    # the corrupted/frozen-frame reports: fine up to some point, then garbled or
    # stuck. A PNG is self-contained and independently decoded; a bad one fails
    # or gets skipped on its own instead of shifting every subsequent frame.
    # Written next to the output video (overwritten fresh each export, kept
    # afterward for now) so a bad export's exact source PNGs can be inspected to
    # tell whether a glitch was already baked into a frame before ffmpeg ever
    # touched it, or was introduced during encoding.
    tmpdir = out.parent / f'{out.stem}_frames'
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, frame in enumerate(frames):
        if stop_event is not None and stop_event.is_set():
            break
        img = render_frame(frame)
        ok = cv2.imwrite(str(tmpdir / f'frame_{i:06d}.png'), img)
        if not ok:
            raise RuntimeError(f"failed to write frame {i} to a temp PNG")
        written += 1
        if progress_cb:
            progress_cb(i + 1, n, f"Rendering frame {i + 1}/{n}")

    if written == 0:
        return str(out)   # cancelled before any frame was rendered

    if progress_cb:
        progress_cb(written, n, f"Encoding {written} frames to ProRes 4444…")

    # Encode to a temp path and only rename to the real output on success.
    # A file written directly to `out` and interrupted mid-write (app force-quit,
    # OS "not responding" kill, etc.) would be left with an inconsistent
    # container — rejected outright by QuickTime and other strict decoders.
    # Renaming only after a clean exit means an interrupted export leaves no
    # output at all instead of a file that looks done but isn't.
    tmp_out = out.with_suffix('.encoding.mov')
    cmd = [ffmpeg, '-y', '-loglevel', 'warning',
           '-framerate', f'{out_fps}',
           '-i', str(tmpdir / 'frame_%06d.png'),
           '-an', '-c:v', 'prores_ks', '-profile:v', '4',
           '-pix_fmt', 'yuva444p10le' if transparent else 'yuv444p10le']
    cmd.append(str(tmp_out))
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg exited with code {result.returncode}:\n"
            f"{result.stderr.decode('utf-8', 'replace')[-4000:]}")
    os.replace(tmp_out, out)
    return str(out)
