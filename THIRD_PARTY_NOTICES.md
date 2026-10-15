# Third-party software

Smalti's own source code is released under the [MIT License](LICENSE). It builds on
the open-source projects below, each under its own license. This is a summary; each
project's own license text is authoritative.

## Python packages (installed via `requirements.txt`)

| Package | License |
|---|---|
| [NiceGUI](https://nicegui.io) | MIT |
| [NumPy](https://numpy.org) | BSD-3-Clause |
| [SciPy](https://scipy.org) | BSD-3-Clause |
| [Numba](https://numba.pydata.org) / [llvmlite](https://llvmlite.readthedocs.io) | BSD-2-Clause (llvmlite bundles LLVM, Apache-2.0 with LLVM exception) |
| [OpenCV](https://opencv.org) (`opencv-python-headless`) | Apache-2.0 (the wheel also bundles LGPL FFmpeg libraries) |
| [Pillow](https://python-pillow.org) | MIT-CMU (HPND) |
| [pillow-heif](https://github.com/bigcat88/pillow_heif) | BSD-3-Clause (bundles LGPL libheif / libde265) |
| [Shapely](https://shapely.readthedocs.io) | BSD-3-Clause (bundles LGPL GEOS) |
| [svgwrite](https://github.com/mozman/svgwrite) | MIT |

## Optional packages (`requirements-optional.txt`)

| Package | License | Notes |
|---|---|---|
| [pywebview](https://pywebview.flowrl.com) | BSD-3-Clause | Native desktop window |
| [PyTorch](https://pytorch.org) | BSD-3-Clause | Optional Apple-GPU acceleration; **not** included in the prebuilt app |
| [PyInstaller](https://pyinstaller.org) | GPL-2.0-or-later with a special exception for the programs it builds | Build tool only; does not affect Smalti's license |

## Bundled in the repository

| Component | License |
|---|---|
| [PixiJS](https://pixijs.com) v8.19.0 (`assets/pixi.min.js`) | MIT (the minified bundle also embeds third-party components under their own licenses, e.g. BSD-3-Clause) |

## FFmpeg — prebuilt app only

The prebuilt `Smalti.app` (GitHub Releases) bundles an **unmodified FFmpeg 8.1.1** binary
(Homebrew build, configured with `--enable-gpl --enable-version3`), used only as a
separate command-line program to encode ProRes video. FFmpeg is licensed under the
**GNU General Public License v3 (or later) in this configuration** and is *not* covered
by Smalti's MIT license.

- FFmpeg project and license: <https://ffmpeg.org/legal.html>
- Corresponding source for the bundled version: <https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz>
- Build recipe used: the Homebrew `ffmpeg` formula, <https://formulae.brew.sh/formula/ffmpeg>

Smalti runs FFmpeg as a separate process and does not link against it. If you run
Smalti from source, no FFmpeg is bundled; it only needs to be on your `PATH` for video
export (see the README).
