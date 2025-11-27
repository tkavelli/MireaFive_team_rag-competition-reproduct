#!/usr/bin/env bash
set -euo pipefail

# Where to place outputs; mount a host path to /app/outputs to collect results.
OUT_DIR=${OUTPUT_DIR:-/app/outputs}
SRC_DIR=/app/outputs

assemble_index() {
  local idx_dir="$1"
  if [ -f "$idx_dir/faiss_index.bin" ]; then
    return 0
  fi
  if [ -f "$idx_dir/faiss_index.bin.part.aa" ]; then
    echo "[retriever] assembling faiss_index.bin from parts..."
    cat "$idx_dir"/faiss_index.bin.part.* > "$idx_dir/faiss_index.bin"
  fi
}

echo "[retriever] writing artifacts to: $OUT_DIR"
mkdir -p "$OUT_DIR"

assemble_index "$SRC_DIR/faiss_index_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00"

# Copy cached artifacts produced by the original retriever run.
cp -r "$SRC_DIR/faiss_index_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00" "$OUT_DIR"/
cp "$SRC_DIR/chunks_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00.csv" "$OUT_DIR"/
cp "$SRC_DIR/submission_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00.csv" "$OUT_DIR"/

echo "[retriever] done. Files copied:"
ls -1 "$OUT_DIR" | sed 's/^/  - /'

# If you want to recompute instead of copy, run inside the container for example:
# python3 pipelines/retrieve.py --queries data/questions_clean.csv --corpus data/websites.csv \
#   --out /app/outputs/submission.csv --chunk_size 768 --overlap 64 --model Qwen/Qwen3-Embedding-8B \
#   --use_hybrid --semantic_weight 0.70 --original_weight 1.00
