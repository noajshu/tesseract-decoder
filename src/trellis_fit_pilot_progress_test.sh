#!/usr/bin/env bash
set -euo pipefail

binary="$TEST_SRCDIR/$TEST_WORKSPACE/src/trellis_fit_pilot"
scratch=$(mktemp -d "${TEST_TMPDIR:-/tmp}/trellis-fit-progress.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

cat >"$scratch/model.dem" <<'EOF'
error(0.1) D0
error(0.00000001) D0 L0
detector D0
EOF
cat >"$scratch/counts.txt" <<'EOF'
0 5 1 3 1
1 2 1 1 2
EOF

"$binary" partial \
  "$scratch/model.dem" "$scratch/counts.txt" "$scratch/plain.json" \
  64 1 gradient observable train mass 2.0 \
  2>"$scratch/plain.stderr"
test ! -s "$scratch/plain.stderr"

TESSERACT_PROGRESS_INTERVAL_MS=1 "$binary" partial \
  "$scratch/model.dem" "$scratch/counts.txt" "$scratch/progress.json" \
  64 1 gradient observable train mass 2.0 \
  2>"$scratch/progress.stderr"
grep -qx 'TRELLIS_PROGRESS 0 2' "$scratch/progress.stderr"
grep -qx 'TRELLIS_PROGRESS 2 2' "$scratch/progress.stderr"
test -s "$scratch/progress.json"
