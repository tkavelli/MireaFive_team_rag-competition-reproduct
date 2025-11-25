#!/bin/bash
# Qwen3-Embedding-8B + RRF v2 Stable (Context7 optimizations without Flash Attention 2)
# Expected Hit@5: 42-45% (vs BGE-M3: 35.07%, Qwen3 v1: 38-41%)

echo "============================================"
echo "Qwen3-Embedding RRF v2 (Context7 Stable)"
echo "============================================"
echo "Model: Qwen/Qwen3-Embedding-8B (FP16)"
echo "MTEB Retrieval: 86.40 (vs BGE-M3: 80.76)"
echo "Chunk size: 2048 tokens"
echo "Batch size: 2 (reduced for large model)"
echo "Optimizations:"
echo "  - Last token pooling"
echo "  - Proper instruction formatting"
echo "  - Max length: 8192 (per docs)"
echo "  - Flash Attention 2 disabled (stable mode)"
echo "============================================"
echo ""

# Set environment variables (local cache to avoid read-only HF_HOME)
export HF_HUB_OFFLINE=0
export HF_HOME="$(pwd)/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "Model will be loaded from cache..."
echo "   Location: $TRANSFORMERS_CACHE"
echo "   Size: ~15GB (FP16)"
echo ""

# Kill any existing processes first
pkill -f "qwen3_rrf_v2_context7.csv" || true

# Run retrieval pipeline with stable parameters
python pipelines/retrieve.py \
  --queries questions_clean.csv \
  --corpus websites.csv \
  --out outputs/submission_qwen3_rrf_v2_stable.csv \
  --model "Qwen/Qwen3-Embedding-8B" \
  --chunk_version ch_v5 \
  --model_name qwen3_8b \
  --chunk_size 2048 \
  --overlap 200 \
  --batch_size 2 \
  --top_k 5 \
  --use_hybrid

echo ""
echo "============================================"
echo "Stable experiment completed"
echo "============================================"
echo "Output: outputs/submission_qwen3_rrf_v2_stable.csv"
echo ""
echo "Next steps:"
echo "  1. Validate submission format"
echo "  2. Upload to leaderboard"
echo "  3. Record Hit@5 in PROGRESS_LOG.md"
echo "  4. Compare with v1 results"
echo "============================================"
