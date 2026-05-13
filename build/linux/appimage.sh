#!/usr/bin/env bash
# build/linux/appimage.sh — Build Quark AppImages for Linux
# Requires: appimagetool (https://github.com/AppImage/AppImageKit/releases)
# Usage: bash build/linux/appimage.sh [cpu|cuda]
set -euo pipefail

BUILD_TYPE="${1:-cpu}"
VERSION="0.1.0"
APP_NAME="Quark"

if [ "$BUILD_TYPE" = "cuda" ]; then
  FEATURES="backend-cpu backend-cuda"
  SUFFIX="linux-cuda-amd64"
else
  FEATURES="backend-cpu"
  SUFFIX="linux-cpu-amd64"
fi

echo "==> Building Quark (${BUILD_TYPE}) …"
cargo build --release --package quark-gui --features "$FEATURES"

APPDIR="dist/${APP_NAME}.AppDir"
rm -rf "$APPDIR"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/share/applications"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"

cp "target/release/quark-gui" "${APPDIR}/usr/bin/${APP_NAME}"
chmod +x "${APPDIR}/usr/bin/${APP_NAME}"

# Desktop entry
cat > "${APPDIR}/usr/share/applications/${APP_NAME}.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=Quark LLM
Comment=Train and run your own coding LLM
Exec=Quark
Icon=quark
Categories=Development;Science;
DESKTOP

cp "${APPDIR}/usr/share/applications/${APP_NAME}.desktop" "${APPDIR}/${APP_NAME}.desktop"

# Icon (copy if present, else create placeholder)
if [ -f "assets/quark.png" ]; then
  cp "assets/quark.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/quark.png"
  cp "assets/quark.png" "${APPDIR}/quark.png"
else
  # Minimal 1x1 placeholder so appimagetool does not fail
  printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82' \
    > "${APPDIR}/quark.png"
  cp "${APPDIR}/quark.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/quark.png"
fi

# AppRun symlink
cat > "${APPDIR}/AppRun" << 'APPRUN'
#!/bin/sh
exec "$(dirname "$0")/usr/bin/Quark" "$@"
APPRUN
chmod +x "${APPDIR}/AppRun"

OUTPUT="dist/${APP_NAME}-${VERSION}-${SUFFIX}.AppImage"
mkdir -p dist
ARCH=x86_64 appimagetool "${APPDIR}" "${OUTPUT}"
echo "==> Done: ${OUTPUT}"
