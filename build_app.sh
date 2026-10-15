#!/bin/bash
# Build a self-contained, unsigned Smalti.app for Apple Silicon Macs.
# Recipients right-click → Open (or run the bundled first-time .command) to clear
# Gatekeeper quarantine.  Torch is excluded to keep the bundle small.
#
# Prerequisites: a venv with the requirements installed, plus PyInstaller:
#   python3 -m venv venv && venv/bin/pip install -r requirements.txt -r requirements-optional.txt
# Optional: `brew install ffmpeg` BEFORE building to bundle ffmpeg for ProRes video export
# (without it the app still builds; video export falls back to .mp4 / PNG sequences).
set -e
cd "$(dirname "$0")"

VENV=venv
PY=$VENV/bin/python
PYI=$VENV/bin/pyinstaller

[ -x "$PYI" ] || { echo "✗ $PYI not found — see the prerequisites at the top of this script"; exit 1; }

FFMPEG_ARGS=()
if FFMPEG_BIN="$(command -v ffmpeg)"; then
  FFMPEG_ARGS=(--add-binary "$FFMPEG_BIN:.")
  echo "▶ bundling ffmpeg from $FFMPEG_BIN"
else
  echo "▶ ffmpeg not found on PATH — building without it (video export will use the .mp4 / PNG-sequence fallbacks)"
fi

echo "▶ cleaning previous build…"
rm -rf build dist/Smalti.app

echo "▶ running PyInstaller…"
$PYI \
  --name Smalti \
  --windowed \
  --noconfirm \
  --clean \
  --icon assets/AppIcon.icns \
  --osx-bundle-identifier com.smalti.app \
  --collect-all nicegui \
  --collect-all pillow_heif \
  --collect-all numba \
  --collect-all llvmlite \
  --collect-all shapely \
  --collect-submodules scipy \
  --hidden-import webview.platforms.cocoa \
  --hidden-import core.video_mosaic \
  --hidden-import core.video_render \
  --hidden-import core.video_sequence_export \
  --hidden-import core.cnc_export \
  --hidden-import core.model3d_export \
  --hidden-import core.gcode_export \
  --hidden-import core.plot_svg_export \
  --add-data "assets:assets" \
  "${FFMPEG_ARGS[@]}" \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module torchaudio \
  --exclude-module functorch \
  --exclude-module torchgen \
  main.py

echo "▶ build complete → dist/Smalti.app"
