#!/usr/bin/env bash
set -eu
HERE="$(dirname $0)"
ROOT="$(realpath $HERE/..)"
DEST="${ROOT}/vendor/highfive"
HIGHFIVE_VERSION="v2.3.1"
HIGHFIVE_URL="https://github.com/BlueBrain/HighFive/archive/refs/tags/${HIGHFIVE_VERSION}.tar.gz"
echo "Downloading highfive ${HIGHFIVE_VERSION} to ${DEST}"
rm -rf "$DEST"
mkdir -p "$DEST"
curl --silent -L "$HIGHFIVE_URL" | tar zxf - -C "$DEST" --strip-components=1
cp $DEST/LICENSE $ROOT/LICENSE_highfive
echo "Done!"
