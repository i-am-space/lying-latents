#!/bin/bash
# Parallel ranged download of Gemma Scope params.npz files, verified against HF's sha256.
# Single HF streams run at ~0.5 MB/s from this host; many concurrent ranges do not.
# Usage: fetch_saes.sh <dest_dir> <jobs> <path> [<path> ...]     (resumable: rerun to continue)
set -u
DEST=$1; JOBS=$2; shift 2
REPO=https://huggingface.co/google/gemma-scope-2b-pt-res/resolve/main
TOK=$(cat ~/.cache/huggingface/token 2>/dev/null)
CHUNK=$((64*1024*1024))
fetch_chunk() {  # url out start end
  local want=$(( $4 - $3 + 1 ))
  for try in $(seq 1 60); do
    [ -f "$2" ] && [ "$(stat -c %s "$2")" = "$want" ] && return 0
    curl -sL --max-time 900 --speed-limit 20000 --speed-time 60 -H "Authorization: Bearer $TOK" \
         -r "$3-$4" -o "$2.tmp" "$1" && [ "$(stat -c %s "$2.tmp")" = "$want" ] && mv "$2.tmp" "$2" && return 0
    sleep 3
  done
  echo "FAILED chunk $2" >&2; return 1
}
export -f fetch_chunk; export TOK
for P in "$@"; do
  out="$DEST/$P/params.npz"; mkdir -p "$DEST/$P/parts"
  [ -f "$DEST/$P/.verified" ] && { echo "$(date +%T) have $P"; continue; }
  hdr=$(curl -sI -H "Authorization: Bearer $TOK" "$REPO/$P/params.npz")
  size=$(echo "$hdr" | grep -i '^x-linked-size' | tr -d '\r' | awk '{print $2}')
  sha=$(echo "$hdr" | grep -i '^x-linked-etag' | tr -d '\r"' | awk '{print $2}')
  n=$(( (size + CHUNK - 1) / CHUNK ))
  echo "$(date +%T) start $P  $((size/1000000)) MB in $n chunks  sha=$sha"
  for i in $(seq 0 $((n-1))); do
    s=$((i*CHUNK)); e=$(( s + CHUNK - 1 )); [ $e -ge $size ] && e=$((size-1))
    printf '%s %s %s %s\n' "$REPO/$P/params.npz" "$DEST/$P/parts/$(printf %05d $i)" $s $e
  done | xargs -P "$JOBS" -L 1 bash -c 'fetch_chunk "$0" "$1" "$2" "$3"' || { echo "FAILED $P"; continue; }
  cat "$DEST/$P"/parts/* > "$out"
  got=$(sha256sum "$out" | awk '{print $1}')
  if [ "$got" = "$sha" ]; then rm -rf "$DEST/$P/parts"; touch "$DEST/$P/.verified"; echo "$(date +%T) VERIFIED $P"
  else echo "$(date +%T) SHA MISMATCH $P ($got != $sha); discarding parts"; rm -rf "$out" "$DEST/$P/parts"; fi
done
