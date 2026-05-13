#!/usr/bin/env bash
# build/linux/deb.sh — Build a .deb package for Quark on Debian/Ubuntu
# Assumes all three binaries have already been built with:
#   cargo build --release --package quark-gui --features backend-cpu
#   cargo build --release --package quark-chat --features backend-cpu
#   cargo build --release --package quark-code --features backend-cpu
# Usage: bash build/linux/deb.sh
set -euo pipefail

VERSION="${VERSION:-0.1.0}"
ARCH="amd64"
PKG_NAME="quark"
PKG_DIR="dist/${PKG_NAME}_${VERSION}_${ARCH}"
TARGET_DIR="target/release"

echo "==> Assembling .deb tree at ${PKG_DIR}…"
rm -rf "$PKG_DIR"
mkdir -p "${PKG_DIR}/DEBIAN"
mkdir -p "${PKG_DIR}/usr/bin"
mkdir -p "${PKG_DIR}/usr/share/applications"
mkdir -p "${PKG_DIR}/usr/share/doc/${PKG_NAME}"

# Binaries
install -m 755 "${TARGET_DIR}/quark"       "${PKG_DIR}/usr/bin/quark"
install -m 755 "${TARGET_DIR}/quark-chat"  "${PKG_DIR}/usr/bin/quark-chat"
install -m 755 "${TARGET_DIR}/quark-code"  "${PKG_DIR}/usr/bin/quark-code"

# Desktop entry for the GUI
cat > "${PKG_DIR}/usr/share/applications/quark.desktop" << 'DESKTOP'
[Desktop Entry]
Version=1.0
Type=Application
Name=Quark LLM
GenericName=LLM Trainer
Comment=Train and run your own Llama 4-style MoE coding LLM
Exec=quark %u
Icon=quark
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
Keywords=llm;ai;ml;training;coding;transformer;
StartupNotify=true
DESKTOP

# Copyright / changelog
cp LICENSE "${PKG_DIR}/usr/share/doc/${PKG_NAME}/copyright" 2>/dev/null || \
  echo "MIT License — see https://github.com/0xnullsect0r/Quark" \
    > "${PKG_DIR}/usr/share/doc/${PKG_NAME}/copyright"

# Calculate installed size (kB)
INSTALLED_SIZE=$(du -sk "${PKG_DIR}/usr" | awk '{print $1}')

# DEBIAN/control
cat > "${PKG_DIR}/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Quark Contributors <https://github.com/0xnullsect0r/Quark>
Installed-Size: ${INSTALLED_SIZE}
Depends: libgtk-3-0, libssl3 | libssl1.1
Homepage: https://github.com/0xnullsect0r/Quark
Section: devel
Priority: optional
Description: Quark LLM — train and run your own coding LLM
 Quark is a desktop application that lets you configure, train from scratch,
 or fine-tune a Llama 4-style Mixture-of-Experts (MoE) transformer LLM
 entirely on your own hardware.
 .
 Includes quark (GUI), quark-chat (terminal REPL), and quark-code (AI coding
 agent with full project context, slash commands, and MCP tool support).
EOF

# DEBIAN/postinst — update desktop database
cat > "${PKG_DIR}/DEBIAN/postinst" << 'POSTINST'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi
POSTINST
chmod 755 "${PKG_DIR}/DEBIAN/postinst"

# DEBIAN/postrm — clean up on uninstall
cat > "${PKG_DIR}/DEBIAN/postrm" << 'POSTRM'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi
POSTRM
chmod 755 "${PKG_DIR}/DEBIAN/postrm"

OUTPUT="dist/${PKG_NAME}_${VERSION}_${ARCH}.deb"
mkdir -p dist
echo "==> Building ${OUTPUT}…"
dpkg-deb --build --root-owner-group "$PKG_DIR" "$OUTPUT"
echo "==> Done: ${OUTPUT}"
echo "    Install with: sudo dpkg -i ${OUTPUT}"
