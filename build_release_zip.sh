#!/bin/bash
# Build the shareable release zip: dist/Smalti-mac.zip
#
# Runs build_app.sh (a standalone, UNSIGNED Smalti.app for Apple Silicon), then packages
# it with a first-run helper, a short readme, the MIT license, and the third-party
# notices (which carry the FFmpeg GPL source offer — keep them in the zip).
set -e
cd "$(dirname "$0")"

APP="dist/Smalti.app"
STAGING="$(mktemp -d)/Smalti"
FINAL_ZIP="dist/Smalti-mac.zip"

bash ./build_app.sh

[ -d "$APP" ] || { echo "✗ $APP missing after build"; exit 1; }

echo "▶ packaging zip…"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/Smalti.app"
xattr -cr "$STAGING/Smalti.app" 2>/dev/null || true
cp LICENSE THIRD_PARTY_NOTICES.md "$STAGING/"

cat > "$STAGING/Open Smalti (first time).command" << 'EOF'
#!/bin/bash
# First-run helper for Smalti: removes the macOS "quarantine" flag that blocks
# unsigned apps, then launches Smalti.  You only need to run this once.
DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$DIR/Smalti.app"

if [ ! -d "$APP" ]; then
  osascript -e 'display alert "Smalti" message "Smalti.app must be in the SAME folder as this file. Put them together and try again."'
  exit 1
fi

echo "Preparing Smalti…"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null
xattr -cr "$APP" 2>/dev/null

echo "Launching Smalti…"
open "$APP" && {
  echo "Done. Smalti is opening — you can close this window."
  echo "From now on you can just double-click Smalti.app directly."
}
EOF
chmod +x "$STAGING/Open Smalti (first time).command"

cat > "$STAGING/READ ME FIRST.txt" << 'EOF'
Smalti — photo mosaics for makers
=================================
Source code, docs and updates: https://github.com/timfennell/smalti

REQUIREMENTS
  • A Mac with Apple Silicon (M1/M2/M3/M4). Intel Macs are not supported by this build.

FIRST TIME (one-time setup — needed because the app isn't signed by Apple)
  1. Keep "Smalti.app" and "Open Smalti (first time).command" in the SAME folder.
  2. Right-click "Open Smalti (first time).command"  →  Open  →  Open.
     (A small Terminal window appears, prepares the app, and launches Smalti.)
  3. That's it. From now on you can just double-click Smalti.app.

If macOS still says the app is blocked, open Terminal and paste:
     xattr -dr com.apple.quarantine "/path/to/Smalti.app"
  (drag the app onto the Terminal window to fill in the path), then press Return.

WHAT IT NEEDS
  • Nothing else — the app is self-contained and works offline.

LICENSE
  Smalti is MIT-licensed (see LICENSE). The bundled FFmpeg is GPL-licensed and is
  not covered by that license — see THIRD_PARTY_NOTICES.md.
EOF

rm -f "$FINAL_ZIP"
(cd "$(dirname "$STAGING")" && zip -r -X -q "$OLDPWD/$FINAL_ZIP" Smalti)
rm -rf "$(dirname "$STAGING")"

echo "▶ done → $FINAL_ZIP ($(du -h "$FINAL_ZIP" | cut -f1))"
