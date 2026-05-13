#!/usr/bin/env bash
# build/linux/rpm.sh — Build a .rpm package for Quark on Fedora/RHEL/openSUSE
# Assumes all three binaries have already been built with:
#   cargo build --release --package quark-gui --features backend-cpu
#   cargo build --release --package quark-chat --features backend-cpu
#   cargo build --release --package quark-code --features backend-cpu
# Requires: rpm-build
# Usage: bash build/linux/rpm.sh
set -euo pipefail

VERSION="${VERSION:-0.1.0}"
TARGET_DIR="target/release"
SPEC_SRC="installer/linux/quark.spec"

# Set up rpmbuild tree inside dist/
RPMBUILD_DIR="$(pwd)/dist/rpmbuild"
mkdir -p "${RPMBUILD_DIR}"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Copy binaries into BUILD so the spec can find them
cp "${TARGET_DIR}/quark"      "${RPMBUILD_DIR}/BUILD/quark"
cp "${TARGET_DIR}/quark-chat" "${RPMBUILD_DIR}/BUILD/quark-chat"
cp "${TARGET_DIR}/quark-code" "${RPMBUILD_DIR}/BUILD/quark-code"

# Desktop entry
cat > "${RPMBUILD_DIR}/BUILD/quark.desktop" << 'DESKTOP'
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

# Copy spec, substituting the version
sed "s/^Version:.*/Version:        ${VERSION}/" "${SPEC_SRC}" \
  > "${RPMBUILD_DIR}/SPECS/quark.spec"

echo "==> Running rpmbuild…"
rpmbuild -bb \
  --define "_topdir ${RPMBUILD_DIR}" \
  "${RPMBUILD_DIR}/SPECS/quark.spec"

# Collect output
RPM_FILE=$(find "${RPMBUILD_DIR}/RPMS" -name "quark-*.rpm" | head -1)
if [ -z "$RPM_FILE" ]; then
  echo "ERROR: rpmbuild did not produce a .rpm file" >&2
  exit 1
fi

mkdir -p dist
OUTPUT="dist/quark-${VERSION}-1.x86_64.rpm"
cp "$RPM_FILE" "$OUTPUT"
echo "==> Done: ${OUTPUT}"
echo "    Install with: sudo rpm -i ${OUTPUT}"
echo "    Or with dnf:  sudo dnf localinstall ${OUTPUT}"
