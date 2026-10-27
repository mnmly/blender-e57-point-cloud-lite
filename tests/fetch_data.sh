#!/bin/sh
# Download the libE57Format test corpus used by tests/test_e57.py into tests/data.
set -e
cd "$(dirname "$0")"
mkdir -p data
base=https://raw.githubusercontent.com/asmaloney/libE57Format-test-data/main
for p in 3rdParty/las2e57/ColourRepresentation.e57 reference/bunnyDouble.e57 reference/bunnyInt32.e57 \
         self/ColouredCubeDouble.e57 self/ColouredCubeFloat.e57 self/ZeroPoints.e57 self/ZeroPointsInvalid.e57 \
         self/InvalidCVHeader.e57 self/empty.e57 self/bad-crc.e57 self/NoPrototype.e57; do
    [ -f "data/$(basename "$p")" ] || curl -sfL -o "data/$(basename "$p")" "$base/$p"
done
