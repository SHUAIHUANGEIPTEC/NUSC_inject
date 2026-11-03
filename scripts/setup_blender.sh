#!/usr/bin/env bash
# Download and unpack Blender 3.6.23 LTS (headless, bundled Python 3.10 + numpy).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$ROOT_DIR/tools"
BLENDER_VER="3.6.23"
PKG="blender-${BLENDER_VER}-linux-x64"
URL="https://download.blender.org/release/Blender3.6/${PKG}.tar.xz"
TARGET="$TOOLS_DIR/blender"

mkdir -p "$TOOLS_DIR"

if [ -x "$TARGET/blender" ] && "$TARGET/blender" --version | grep -q "$BLENDER_VER"; then
    echo "Blender $BLENDER_VER already installed at $TARGET"
    exit 0
fi

ARCHIVE="$TOOLS_DIR/${PKG}.tar.xz"
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading $URL ..."
    wget -c -O "$ARCHIVE" "$URL"
fi

echo "Extracting $ARCHIVE ..."
tar -xJf "$ARCHIVE" -C "$TOOLS_DIR"
ln -sfn "$TOOLS_DIR/$PKG" "$TARGET"

# Headless system libs (no-op if already present; needs sudo)
if command -v apt-get >/dev/null 2>&1; then
    dpkg -s libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 libsm6 libice6 libxext6 \
        >/dev/null 2>&1 || \
        echo "NOTE: some X/GL libs missing; run: sudo apt-get install -y libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 libsm6 libice6 libxext6"
fi

"$TARGET/blender" --version | head -1
echo "Blender ready: $TARGET/blender"
