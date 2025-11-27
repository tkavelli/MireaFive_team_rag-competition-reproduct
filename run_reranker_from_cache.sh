#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${OUTPUT_DIR:-/app/outputs}
INDEX_DIR=${INDEX_DIR:-/app/outputs/faiss_index_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00}
RUN_TAG=${RUN_TAG:-qwen3_v6_pool50_int4_fast_resume}
POOL_SIZE=${POOL_SIZE:-50}
RERANK_TO=${RERANK_TO:-5}
EMBED_BS=${EMBED_BATCH_SIZE:-8}
RERANK_BS=${RERANKER_BATCH_SIZE:-6}

mkdir -p "$OUT_DIR/qwen3_rerank_runs"

echo "[reranker] index: $INDEX_DIR"
echo "[reranker] outputs: $OUT_DIR/qwen3_rerank_runs"

python3 scripts/run_qwen3_rerank_from_cache.py \
  --queries data/questions_clean.csv \
  --index-dir "$INDEX_DIR" \
  --embedding-model "Qwen/Qwen3-Embedding-8B" \
  --reranker-model "Qwen/Qwen3-Reranker-8B" \
  --pool-size "$POOL_SIZE" \
  --rerank-to "$RERANK_TO" \
  --embed-batch-size "$EMBED_BS" \
  --reranker-batch-size "$RERANK_BS" \
  --reranker-quantization int4 \
  --run-tag "$RUN_TAG" \
  --artifacts-root "$OUT_DIR/qwen3_rerank_runs" \
  ${OUTPUT:+--output "$OUTPUT"}

echo "[reranker] done. Results in $OUT_DIR/qwen3_rerank_runs/$RUN_TAG*"
