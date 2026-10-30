#!/bin/bash
# Render songs in parallel, at most 8 at once (each render holds ~1-4 GB while it runs).
#   tools/render_batch.sh [list]        # default list: songs.txt
# One song per line, "<mp3 file stem in $SONGS_DIR>|<folder under out/>[|<extra flags for this song>]";
# blank lines and # comments skipped.
# JOBS=2 tools/render_batch.sh runs fewer at once; it is never more than MAX_JOBS.
# Extra pop2quartet flags for every song: FLAGS="--lead-ahead 4" tools/render_batch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MAX_JOBS=8
JOBS=${JOBS:-$MAX_JOBS}; (( JOBS > MAX_JOBS )) && JOBS=$MAX_JOBS
export SONGS_DIR=${SONGS_DIR:-/mnt/c/Users/addis/Downloads} FLAGS=${FLAGS:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
mkdir -p out/.logs

one() {
  IFS='|' read -r f o extra <<< "$1"; s=$(date +%s); log="out/.logs/${o//\//_}.log"
  # shellcheck disable=SC2086
  if nice -n 19 .venv/bin/pop2quartet "$SONGS_DIR/$f.mp3" -o "out/$o" $FLAGS $extra < /dev/null > "$log" 2>&1; then
    cp "$log" "out/$o/run.log"; status=ok
  else status=FAILED; fi
  line=$(grep -E '^\[beats\] drum|^\[quality\]|played by|the tune sits high' "$log" \
    | sed -E 's/.*-> (steady|tracked)/\1 beat/; s/\[quality\] .*out_of_key ([0-9.]+).*total ([0-9.]+).*/out-of-key \1, score \2/; s/.*played by (.*)\)/\1/; s/.*the tune sits high.*/lowered/' | tr '\n' '|')
  printf "%-22s %s %s (%ss)\n" "$o" "$status" "$line" "$(( $(date +%s) - s ))"
}
export -f one
echo "rendering with $JOBS at once"
grep -vE '^\s*(#|$)' "${1:-songs.txt}" | xargs -P "$JOBS" -d '\n' -I{} bash -c 'one "$1"' _ {}
