"""
Qwen3 Embedding wrapper with official Context7 optimizations.

Requirements:
- transformers>=4.51.0
- sentence-transformers>=2.7.0
- flash-attn (optional, for +30-50% speed): pip install flash-attn --no-build-isolation

Features:
- Flash Attention 2 with fallback for maximum performance
- Memory-efficient processing with adaptive batch sizes
- Last token pooling as recommended in Qwen3 documentation
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Optional
import logging
import gc

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last token pooling as recommended in Qwen3 documentation.

    Args:
        last_hidden_states: Hidden states from the model
        attention_mask: Attention mask for the input

    Returns:
        Pooled embeddings using the last token
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Format query with instruction as recommended in Qwen3 documentation.

    Args:
        task_description: Task description
        query: Query text

    Returns:
        Formatted query with instruction
    """
    return f'Instruct: {task_description}\nQuery:{query}'


class Qwen3EmbeddingModel:
    """
    Stable wrapper for Qwen3-Embedding models using AutoModel with Context7 optimizations.
    Excludes Flash Attention 2 for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        max_seq_length: int = 8192,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize Qwen3 embedding model with stable settings.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_seq_length: Maximum sequence length (default: 8192 per docs)
            embedding_dim: Custom embedding dimension for MRL (optional)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.custom_embedding_dim = embedding_dim

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        logger.info(f"Loading Qwen3 stable model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Max sequence length: {max_seq_length}")

        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")

        # Load tokenizer and model with stable settings
        try:
            cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='left',
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # Try loading model with Flash Attention 2 first (official recommendation)
            # Requires: transformers>=4.51.0, flash-attn installed
            logger.info(f"Loading model with Flash Attention 2 optimization...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=None,  # single-GPU; avoids accelerate dependency
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"
                )
                self.model.to(self.device)
                logger.info("Flash Attention 2 enabled successfully!")
                logger.info("Expected: +30-50% speed, -20-30% VRAM usage")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
                logger.info("To install Flash Attention 2: pip install flash-attn --no-build-isolation")
                logger.info("Falling back to standard attention...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
                logger.info("Model loaded with standard attention (slower but compatible)")

            self.model.eval()
            self.model_embedding_dim = self.model.config.hidden_size

            # Use custom embedding dimension if specified (MRL support)
            self.embedding_dim = embedding_dim if embedding_dim else self.model_embedding_dim

            logger.info(f"Qwen3 stable model loaded successfully!")
            logger.info(f"Model embedding dimension: {self.model_embedding_dim}")
            if embedding_dim:
                logger.info(f"Custom embedding dimension (MRL): {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load Qwen3 stable model: {e}")
            raise

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int = 2,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings using Qwen3 model with stable optimizations.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize with proper padding
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings with memory optimization
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use last_token_pool instead of mean pooling (Context7 recommendation)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

                # Apply MRL if custom embedding dimension is specified
                if self.custom_embedding_dim and self.custom_embedding_dim < self.model_embedding_dim:
                    embeddings = embeddings[:, :self.custom_embedding_dim]

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                # Move to CPU immediately to free GPU memory
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings.numpy())

                # Clear inputs and outputs from memory
                del inputs, outputs, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            if show_progress and i % (batch_size * 4) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return np.vstack(all_embeddings)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
        normalize_embeddings: bool = True,
        is_query: bool = False,
        show_progress_bar: Optional[bool] = None,
        use_sentence_transformer: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings with Qwen3 Context7 optimizations.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Normalize embeddings to unit length
            is_query: Whether texts are queries (affects instruction formatting)
            use_sentence_transformer: Use SentenceTransformer with official prompts (recommended)

        Returns:
            Numpy array of embeddings
        """
        if show_progress_bar is not None:
            show_progress = show_progress_bar

        # Use SentenceTransformer with official prompts if requested
        if use_sentence_transformer:
            return self.encode_with_sentence_transformer(
                texts, batch_size=batch_size, is_query=is_query,
                normalize_embeddings=normalize_embeddings, show_progress=show_progress
            )

        # Fallback to manual instruction formatting (legacy method)
        if is_query:
            # Use proper instruction from Context7 documentation
            task_description = "Given a web search query, retrieve relevant passages that answer the query"
            formatted_texts = [get_detailed_instruct(task_description, text) for text in texts]
            logger.info("📝 Using manual instruction formatting (legacy method)")
        else:
            # Documents don't need instruction (per Context7 docs)
            formatted_texts = texts

        # For very large datasets, use memory-efficient batch processing
        if len(formatted_texts) > 1000 and isinstance(self.model, AutoModel):
            logger.info(f"Large dataset detected ({len(formatted_texts)} items), using memory-efficient processing")
            return self._encode_large_dataset_memory_efficient(formatted_texts, batch_size, show_progress, normalize_embeddings)

        return self._encode_texts(
            formatted_texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings
        )

    def _encode_large_dataset_memory_efficient(self, texts: List[str], batch_size: int, show_progress: bool, normalize_embeddings: bool) -> np.ndarray:
        """Memory-efficient encoding for very large datasets"""
        all_embeddings = []
        total_texts = len(texts)

        # Process in smaller batches for memory efficiency
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")

            # Process this batch
            batch_embeddings = self._encode_texts(
                batch_texts,
                batch_size=len(batch_texts),  # Process all remaining items in final batch
                show_progress=show_progress,
                normalize_embeddings=normalize_embeddings
            )

            all_embeddings.append(batch_embeddings)

            # Explicit cleanup after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return np.vstack(all_embeddings)

    def encode_with_sentence_transformer(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
        normalize_embeddings: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Alternative encoding method using SentenceTransformer with official prompts.

        This method follows the official Context7 recommendations:
        - Uses SentenceTransformer library for better performance
        - Built-in prompt_name="query" for queries (no manual formatting needed)
        - Documents encoded without prompts

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (SentenceTransformer can handle larger batches)
            is_query: Whether texts are queries (uses prompt_name="query")
            normalize_embeddings: Normalize embeddings to unit length
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"🔄 Using SentenceTransformer with official prompts (is_query={is_query})")

            # Load model with SentenceTransformer optimizations
            st_model = SentenceTransformer(
                self.model_name,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "trust_remote_code": False,
                    "low_cpu_mem_usage": True,
                    # Try Flash Attention 2 if available
                    "attn_implementation": "flash_attention_2"
                },
                tokenizer_kwargs={"padding_side": "left"},
            )

            # Use official prompt handling
            if is_query:
                logger.info("Using built-in prompt_name='query' for queries")
                embeddings = st_model.encode(
                    texts,
                    batch_size=batch_size,
                    prompt_name="query",  # Official built-in prompt
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            else:
                logger.info("Encoding documents without prompts (official recommendation)")
                embeddings = st_model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )

            logger.info(f"SentenceTransformer encoding completed: {embeddings.shape}")
            return embeddings

        except ImportError:
            logger.warning("SentenceTransformer not available, falling back to AutoModel")
            return self.encode(texts, batch_size=batch_size, is_query=is_query,
                            normalize_embeddings=normalize_embeddings, show_progress=show_progress)
        except Exception as e:
            logger.error(f"SentenceTransformer encoding failed: {e}")
            logger.info("Falling back to AutoModel method...")
            return self.encode(texts, batch_size=batch_size, is_query=is_query,
                            normalize_embeddings=normalize_embeddings, show_progress=show_progress)
