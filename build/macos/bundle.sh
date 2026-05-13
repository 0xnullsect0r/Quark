#!/usr/bin/env bash
# build/macos/bundle.sh — Build Quark.app and Quark.dmg for macOS
# Usage: bash build/macos/bundle.sh [--features "backend-cpu backend-wgpu"]
# Assumes cargo build --release --package quark-gui has already been run.
set -euo pipefail

APP_NAME="Quark"
VERSION="0.1.0"
BUNDLE_ID="com.quark.lm"
TARGET_DIR="target/release"
APP_DIR="dist/${APP_NAME}.app"

echo "==> Creating .app bundle…"
rm -rf "$APP_DIR"
mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"

cp "${TARGET_DIR}/quark" "${APP_DIR}/Contents/MacOS/${APP_NAME}"
chmod +x "${APP_DIR}/Contents/MacOS/${APP_NAME}"

# Copy icon if present
if [ -f "assets/quark.icns" ]; then
  cp "assets/quark.icns" "${APP_DIR}/Contents/Resources/${APP_NAME}.icns"
fi

cat > "${APP_DIR}/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>       <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>       <string>${BUNDLE_ID}</string>
  <key>CFBundleName</key>             <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>      <string>Quark LLM</string>
  <key>CFBundleVersion</key>          <string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundlePackageType</key>      <string>APPL</string>
  <key>CFBundleIconFile</key>         <string>${APP_NAME}</string>
  <key>NSHighResolutionCapable</key>  <true/>
  <key>LSMinimumSystemVersion</key>   <string>12.0</string>
  <key>NSHumanReadableCopyright</key> <string>Copyright © 2025 Quark Contributors. MIT License.</string>
</dict>
</plist>
PLIST

echo "==> Creating DMG…"
DMG_NAME="Quark-${VERSION}-macos.dmg"
STAGING="dist/dmg-staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -r "$APP_DIR" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create -volname "Quark ${VERSION}" \
  -srcfolder "$STAGING" \
  -ov -format UDZO \
  "dist/${DMG_NAME}"

rm -rf "$STAGING"
echo "==> Done: dist/${DMG_NAME}"
