#!/usr/bin/env python3
"""
Qwen3-only reranking launched from cached Qwen3 embeddings index.

1. Loads saved chunks/embeddings/FAISS index from `outputs/faiss_index_ch_v5_qwen3_8b/`.
2. Encodes queries using the same Qwen3-Embedding-8B model but does _not_ rebuild
   the chunk embeddings.
3. Searches the cached FAISS index, reranks top candidates with Qwen3-Reranker-8B,
   and writes a leaderboard submission.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.qwen3_embedding import Qwen3EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def log_memory(prefix: str = "") -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("%sGPU memory: allocated=%.2fGB reserved=%.2fGB", prefix, allocated, reserved)


class Qwen3Reranker:
    """Yes/no scoring reranker that follows the official Qwen3 guide."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        device: str | None = None,
        batch_size: int = 8,
        instruction: str | None = None,
        max_length: int = 8192,
        quantization: str = "none",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.instruction = instruction or "Given a web search query, retrieve relevant passages that answer the query"
        self.max_length = max_length
        self.quantization = quantization

        tokenizer_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HUB_CACHE")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            cache_dir=tokenizer_cache,
        )

        model_cache = tokenizer_cache
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "cache_dir": model_cache,
        }

        if self.quantization == "int4":
            if BitsAndBytesConfig is None:
                raise ImportError("transformers>=4.31 with bitsandbytes is required for --reranker-quantization=int4")
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device.startswith("cuda") else torch.float32
            if self.device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _prepare_inputs(self, prompts: List[str]) -> dict:
        max_prompt_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_prompt_length,
        )

        input_ids = []
        for ids in inputs["input_ids"]:
            input_ids.append(self.prefix_tokens + ids + self.suffix_tokens)

        padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        return {k: v.to(self.model.device) for k, v in padded.items()}

    @torch.no_grad()
    def _compute_scores(self, inputs: dict) -> List[float]:
        log_memory("Before forward: ")
        outputs = self.model(**inputs)
        log_memory("After forward: ")
        batch_scores = outputs.logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        stacked = torch.nn.functional.log_softmax(stacked, dim=1)
        return stacked[:, 1].exp().tolist()

    def rerank(self, query: str, documents: List[Tuple[str, dict]], top_k: int = 5) -> List[Tuple[str, dict, float]]:
        if not documents:
            return []

        prompts = [self._format_instruction(query, doc_text) for doc_text, _ in documents]

        scores: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self._prepare_inputs(batch_prompts)
            batch_scores = self._compute_scores(inputs)
            scores.extend(batch_scores)

        scored_docs: List[Tuple[str, dict, float]] = []
        for (doc_text, doc_metadata), score in zip(documents, scores):
            scored_docs.append((doc_text, doc_metadata, float(score)))

        scored_docs.sort(key=lambda x: x[2], reverse=True)
        return scored_docs[:top_k]


def load_index(index_dir: Path) -> tuple[pd.DataFrame, faiss.Index, dict[tuple[int, int], int]]:
    chunks_csv = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"
    faiss_path = index_dir / "faiss_index.bin"
    metadata_json = index_dir / "metadata.json"

    if not chunks_csv.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_csv}")
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    logger.info(f"Loading chunks from {chunks_csv}")
    chunks = pd.read_csv(chunks_csv)
    for column in ("web_id", "chunk_index"):
        if column in chunks.columns:
            chunks[column] = pd.to_numeric(chunks[column], errors="coerce").fillna(0).astype(int)
    if "token_count" in chunks.columns:
        chunks["token_count"] = pd.to_numeric(chunks["token_count"], errors="coerce").fillna(0).astype(int)
    else:
        logger.warning("token_count column missing in chunks.csv; estimating from characters")
        chunks["token_count"] = chunks["chunk_text"].astype(str).apply(lambda txt: max(len(txt.split()), 1))

    logger.info(f"Loading FAISS index from {faiss_path}")
    faiss_index = faiss.read_index(str(faiss_path))
    logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors")

    metadata = json.loads(metadata_json.read_text()) if metadata_json.exists() else {}
    lookup = {
        (int(web_id), int(chunk_index)): idx
        for idx, (web_id, chunk_index) in enumerate(zip(chunks["web_id"], chunks["chunk_index"]))
    }

    logger.info("Index metadata: %s", metadata)
    return chunks, faiss_index, lookup


def encode_queries(
    embedding_model: Qwen3EmbeddingModel, queries: List[str], batch_size: int
) -> np.ndarray:
    """Encode all queries (is_query=True) using the Qwen3 embedding wrapper."""
    logger.info("Encoding %d queries (batch %d)", len(queries), batch_size)
    return embedding_model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        is_query=True,
    )


def load_cached_query_embeddings(cache_path: Path, expected_qids: np.ndarray) -> np.ndarray | None:
    """Attempt to load cached query embeddings that match the provided q_ids order."""

    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_q_ids = data["q_ids"]
            cached_embeddings = data["embeddings"]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to read cached query embeddings at %s: %s", cache_path, exc)
        return None

    expected_qids = np.asarray(expected_qids)
    cache_len = cached_q_ids.shape[0]
    expected_len = expected_qids.shape[0]

    if cache_len < expected_len:
        logger.warning(
            "Cached embeddings shorter than expected (cache=%d, expected=%d)",
            cache_len,
            expected_len,
        )
        return None

    if np.array_equal(cached_q_ids[:expected_len], expected_qids):
        logger.info(
            "Loaded cached query embeddings from %s (%d/%d queries)",
            cache_path,
            expected_len,
            cache_len,
        )
        return np.ascontiguousarray(cached_embeddings[:expected_len], dtype=np.float32)

    logger.warning("Cached q_ids order does not match requested queries; ignoring cache")
    return None


def save_cached_query_embeddings(cache_path: Path, q_ids: np.ndarray, embeddings: np.ndarray) -> None:
    """Persist computed query embeddings for future runs."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        q_ids=np.asarray(q_ids, dtype=np.int64),
        embeddings=np.ascontiguousarray(embeddings, dtype=np.float32),
    )
    logger.info("Saved query embeddings cache to %s", cache_path)


def _estimate_tokens(entry: dict) -> int:
    tokens = int(entry.get("token_count") or 0)
    if tokens <= 0:
        tokens = max(len(str(entry.get("chunk_text", "")).split()), 1)
    return tokens


def assemble_document_context(
    web_id: int,
    candidates: List[dict],
    chunks_df: pd.DataFrame,
    chunk_lookup: dict[tuple[int, int], int],
    *,
    chunks_per_doc: int,
    include_neighbors: bool,
    max_doc_tokens: int,
    max_chunks_total: int,
    log_details: bool = False,
) -> tuple[str, List[str], str, int] | None:
    """Комбинирует несколько чанков одного документа в расширенный контекст."""

    if not candidates:
        return None

    sorted_candidates = sorted(candidates, key=lambda x: x["rank"])
    primary = sorted_candidates[: max(1, chunks_per_doc)]
    used_chunk_ids: set[str] = set()
    ordered_chunks: List[tuple[str, str]] = []
    total_tokens = 0

    def try_add(entry: dict) -> None:
        nonlocal total_tokens
        if len(ordered_chunks) >= max_chunks_total:
            return
        chunk_id = entry["chunk_id"]
        if chunk_id in used_chunk_ids:
            return
        tokens = _estimate_tokens(entry)
        if tokens <= 0:
            return
        if total_tokens + tokens > max_doc_tokens:
            return
        chunk_text = str(entry["chunk_text"]).strip()
        if not chunk_text:
            return
        ordered_chunks.append((chunk_id, chunk_text))
        used_chunk_ids.add(chunk_id)
        total_tokens += tokens

    for entry in primary:
        try_add(entry)

    if include_neighbors:
        for entry in primary:
            base_index = int(entry["chunk_index"])
            for offset in (-1, 1):
                neighbor_key = (web_id, base_index + offset)
                neighbor_pos = chunk_lookup.get(neighbor_key)
                if neighbor_pos is None:
                    continue
                neighbor_row = chunks_df.iloc[neighbor_pos]
                neighbor_entry = {
                    "chunk_id": neighbor_row["chunk_id"],
                    "chunk_text": neighbor_row["chunk_text"],
                    "chunk_index": neighbor_row["chunk_index"],
                    "token_count": neighbor_row.get("token_count", 0),
                }
                try_add(neighbor_entry)

    if not ordered_chunks:
        # fallback: добавляем лучший чанк как есть
        try_add(primary[0])

    if not ordered_chunks:
        return None

    combined_text = "\n\n".join(chunk_text for _, chunk_text in ordered_chunks).strip()
    source_chunk_ids = [chunk_id for chunk_id, _ in ordered_chunks]
    anchor_chunk_id = primary[0]["chunk_id"]

    if log_details:
        preview = combined_text.replace("\n", " ")[:160]
        logger.debug(
            "web_id=%s chunks=%s tokens≈%d preview=%s",
            web_id,
            source_chunk_ids,
            total_tokens,
            preview,
        )

    return combined_text, source_chunk_ids, anchor_chunk_id, total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank via Qwen3 from cached FAISS snapshot")
    parser.add_argument("--queries", default="questions_clean.csv", help="Path to queries CSV")
    parser.add_argument(
        "--index-dir",
        default="outputs/faiss_index_ch_v5_qwen3_8b",
        help="Directory with chunks.csv/faiss_index.bin/embeddings.npy",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model used to encode queries",
    )
    parser.add_argument(
        "--reranker-model",
        default="Qwen/Qwen3-Reranker-4B",
        help="Qwen3 reranker Hugging Face repo identifier",
    )
    parser.add_argument("--pool-size", type=int, default=100, help="How many candidates to rerank")
    parser.add_argument("--rerank-to", type=int, default=5, help="Top-k to keep after reranking")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size for query encoding")
    parser.add_argument("--sample-size", type=int, help="Optional number of queries for smoke tests")
    parser.add_argument(
        "--query-embeddings-cache",
        type=str,
        help="Optional path to .npz cache with query embeddings (speeds up repeated runs)",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=8,
        help="Batch size for the reranker forward pass",
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=8192,
        help="Maximum sequence length for reranker prompts",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=3,
        help="How many top-ranked chunks per document seed the document context",
    )
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=7200,
        help="Approximate token budget per document after concatenating neighbors",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=4,
        help="Hard limit on how many chunks (including neighbors) can form one document context",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log progress every N queries (set to 0 to disable interim logs)",
    )
    parser.add_argument(
        "--run-tag",
        default="qwen3_rerank",
        help="Logical name that prefixes artifacts for this run (will create unique folders)",
    )
    parser.add_argument(
        "--artifacts-root",
        default="outputs/qwen3_rerank_runs",
        help="Root directory where run-specific folders (indexes, caches, submissions) are created",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Explicit path for the submission CSV; if omitted, a timestamped file inside the run folder is used",
    )
    parser.add_argument(
        "--reranker-quantization",
        choices=["none", "int4"],
        default="none",
        help="Quantization strategy for reranker (int4 relies on bitsandbytes to fit 8B on 24GB GPUs)",
    )
    parser.add_argument(
        "--no-neighbors",
        action="store_true",
        help="Disable automatic inclusion of adjacent chunks when building document context",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging verbosity for this run",
    )
    parser.add_argument(
        "--log-context-details",
        action="store_true",
        help="Log chunk ids/token counts for each assembled document (debug only)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save intermediate submission rows every N queries (0 disables checkpointing)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Optional explicit path for checkpoint CSV (defaults to run_dir/checkpoint.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    index_dir = Path(args.index_dir)

    run_tag = args.run_tag.strip() if args.run_tag else "qwen3_rerank"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_tag}_{timestamp}"
    run_dir = Path(args.artifacts_root).joinpath(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Queries embedding cache defaults into the run folder, keeping every run isolated
    cache_path = (
        Path(args.query_embeddings_cache).expanduser().resolve()
        if args.query_embeddings_cache
        else run_dir.joinpath("query_embeddings.npz")
    )
    logger.info("Artifacts root for this run: %s", run_dir)

    # Allow overriding output path, otherwise create one per run for easy traceability
    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir.joinpath(f"submission_{run_id}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else run_dir.joinpath("checkpoint.csv")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
    logger.info("HF cache directory: %s", os.environ["HF_HOME"])

    queries_df = pd.read_csv(args.queries)
    if args.sample_size:
        queries_df = queries_df.head(args.sample_size).copy()
        logger.info("Using sample of %d queries", len(queries_df))

    chunks_df, faiss_index, chunk_lookup = load_index(index_dir)
    q_ids = queries_df["q_id"].astype(np.int64).to_numpy()

    query_embeddings: np.ndarray | None = None
    if cache_path:
        query_embeddings = load_cached_query_embeddings(cache_path, q_ids)

    if query_embeddings is None:
        # Step 1: Load embedding model and encode queries
        logger.info("=== STEP 1: Loading embedding model and encoding queries ===")
        embedding_model = Qwen3EmbeddingModel(
            model_name=args.embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_seq_length=8192,
        )

        query_embeddings = encode_queries(
            embedding_model,
            queries_df["query"].astype(str).tolist(),
            batch_size=args.embed_batch_size,
        )

        if cache_path:
            save_cached_query_embeddings(cache_path, q_ids, query_embeddings)

        # Step 2: Free embedding model memory
        logger.info("=== STEP 2: Freeing embedding model memory ===")
        del embedding_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                "GPU memory freed: %.2fGB",
                torch.cuda.memory_allocated() / 1024 ** 3,
            )
    else:
        logger.info("Skipping query encoding step because cached embeddings were found")

    # Step 3: Load reranker model
    logger.info("=== STEP 3: Loading reranker model ===")
    reranker = Qwen3Reranker(
        model_name=args.reranker_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        quantization=args.reranker_quantization,
    )

    topk = min(args.pool_size, faiss_index.ntotal)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    _, idxs = faiss_index.search(query_embeddings, topk)

    logger.info("=== STEP 4: Starting reranking process ===")
    logger.info(f"Processing {len(queries_df)} queries with pool_size={args.pool_size}, batch_size={reranker.batch_size}")

    submission_rows: List[dict[str, object]] = []
    iterator = tqdm(
        queries_df.itertuples(index=False),
        total=len(queries_df),
        desc="Reranking queries",
    )
    log_interval = max(args.log_interval, 0)
    include_neighbors = not args.no_neighbors

    processed_qids: set[int] = set()
    if args.checkpoint_interval > 0 and checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_csv(checkpoint_path)
            for _, checkpoint_row in checkpoint_df.iterrows():
                submission_rows.append({"q_id": int(checkpoint_row["q_id"]), "web_list": str(checkpoint_row["web_list"])})
                processed_qids.add(int(checkpoint_row["q_id"]))
            logger.info("Resumed from checkpoint with %d rows", len(processed_qids))
        except Exception as exc:
            logger.warning("Failed to load checkpoint %s: %s", checkpoint_path, exc)

    for q_idx, row in enumerate(iterator):
        query_id = int(row.q_id)
        query_text = str(row.query)

        if query_id in processed_qids:
            continue

        # Step 1: Collect all candidates from FAISS
        candidates_by_doc: dict[int, List[dict]] = {}
        for rank, doc_idx in enumerate(idxs[q_idx]):
            if doc_idx < 0:
                continue
            chunk_row = chunks_df.iloc[doc_idx]
            candidate = {
                "chunk_id": chunk_row["chunk_id"],
                "chunk_text": chunk_row["chunk_text"],
                "chunk_index": int(chunk_row["chunk_index"]),
                "token_count": int(chunk_row.get("token_count", 0)),
                "rank": rank,
            }
            web_id = int(chunk_row["web_id"])
            candidates_by_doc.setdefault(web_id, []).append(candidate)

        # Step 2: Собираем расширенный контекст на уровне документа
        document_contexts: List[Tuple[str, dict]] = []
        for web_id, doc_chunks in candidates_by_doc.items():
            assembled = assemble_document_context(
                web_id,
                doc_chunks,
                chunks_df,
                chunk_lookup,
                chunks_per_doc=args.chunks_per_doc,
                include_neighbors=include_neighbors,
                max_doc_tokens=args.max_doc_tokens,
                max_chunks_total=max(args.chunks_per_doc, args.max_chunks_per_doc),
                log_details=args.log_context_details and logger.isEnabledFor(logging.DEBUG),
            )
            if not assembled:
                continue
            combined_text, source_chunk_ids, anchor_chunk_id, approx_tokens = assembled
            doc_meta = {
                "web_id": web_id,
                "chunk_id": anchor_chunk_id,
                "source_chunk_ids": source_chunk_ids,
                "approx_tokens": approx_tokens,
            }
            document_contexts.append((combined_text, doc_meta))

        if not document_contexts:
            logger.warning("No document contexts assembled for query_id=%s", query_id)
            fallback_ids = [0] * args.rerank_to
            submission_rows.append({"q_id": query_id, "web_list": json.dumps(fallback_ids)})
            continue

        # Step 3: Rerank aggregated documents
        try:
            log_memory("Before rerank call: ")
            reranked = reranker.rerank(query_text, document_contexts, top_k=args.rerank_to)
        except RuntimeError as exc:
            log_memory("OOM/Runtime failure: ")
            logger.error(
                "Failed on query_id=%s with %d document contexts; first context tokens≈%s",
                query_id,
                len(document_contexts),
                document_contexts[0][1].get("approx_tokens") if document_contexts else "n/a",
            )
            raise

        # Step 4: Extract web_ids (already unique) - convert to int for JSON serialization
        web_ids = [int(doc_meta["web_id"]) for _, doc_meta, _ in reranked]
        while len(web_ids) < args.rerank_to:
            web_ids.append(web_ids[0] if web_ids else 0)
        submission_rows.append({"q_id": query_id, "web_list": json.dumps(web_ids)})

        if args.checkpoint_interval > 0 and (len(submission_rows) % args.checkpoint_interval == 0):
            pd.DataFrame(submission_rows).to_csv(checkpoint_path, index=False)
            logger.info("Checkpoint saved with %d rows at query_id=%s", len(submission_rows), query_id)

        if log_interval and (q_idx + 1) % log_interval == 0:
            logger.info(
                "Processed %d/%d queries (%.1f%%)",
                q_idx + 1,
                len(queries_df),
                (q_idx + 1) / len(queries_df) * 100,
            )

    logger.info("=== STEP 5: Reranking completed successfully ===")

    submission = pd.DataFrame(submission_rows)

    submission.to_csv(output_path, index=False)
    logger.info("Saved submission to %s", output_path)
    if args.checkpoint_interval > 0:
        pd.DataFrame(submission_rows).to_csv(checkpoint_path, index=False)


if __name__ == "__main__":
    main()
