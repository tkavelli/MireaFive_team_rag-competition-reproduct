"""
Qwen3 Embedding wrapper using AutoModel with Context7 optimizations.
Fixes compatibility issues with Qwen3-Embedding-8B and implements best practices.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Optional
import logging

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
    Wrapper for Qwen3-Embedding models using AutoModel with Context7 optimizations.

    This fixes the compatibility issues with SentenceTransformer
    and provides proper Qwen3-Embedding-8B loading with:
    - last_token_pool instead of mean pooling
    - flash_attention_2 optimization
    - Proper instruction formatting
    - MRL (Matryoshka Representation Learning) support
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        max_seq_length: int = 8192,  # Updated from 32768 to 8192 per docs
        embedding_dim: Optional[int] = None  # For MRL support
    ):
        """
        Initialize Qwen3 embedding model with Context7 optimizations.

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

        logger.info(f"Loading Qwen3 model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Max sequence length: {max_seq_length}")

        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")

        # Load tokenizer and model with Context7 best practices
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='left',  # Critical for Qwen3
                trust_remote_code=True
            )

            # Try to load with flash_attention_2 first (recommended in docs)
            try:
                logger.info("Loading model with flash_attention_2 optimization...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"  # Context7 optimization
                )
                logger.info("Flash Attention 2 enabled")
            except Exception as flash_error:
                logger.warning(f"Flash Attention 2 not available: {flash_error}")
                logger.info("Loading model without flash_attention_2...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True
                )

            self.model.eval()
            self.model_embedding_dim = self.model.config.hidden_size

            # Use custom embedding dimension if specified (MRL support)
            self.embedding_dim = embedding_dim if embedding_dim else self.model_embedding_dim

            logger.info(f"Qwen3 model loaded successfully!")
            logger.info(f"Model embedding dimension: {self.model_embedding_dim}")
            if embedding_dim:
                logger.info(f"Custom embedding dimension (MRL): {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            raise

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings using Qwen3 model with Context7 optimizations.

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

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use last_token_pool instead of mean pooling (Context7 recommendation)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

                # Apply MRL if custom embedding dimension is specified
                if self.custom_embedding_dim and self.custom_embedding_dim < self.model_embedding_dim:
                    embeddings = embeddings[:, :self.custom_embedding_dim]

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

            if show_progress and i % (batch_size * 4) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return np.vstack(all_embeddings)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings with Qwen3 Context7 instruction formatting.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Normalize embeddings to unit length
            is_query: Whether texts are queries (affects instruction formatting)

        Returns:
            Numpy array of embeddings
        """
        if is_query:
            # Use proper instruction from Context7 documentation
            # Updated task description for better web search retrieval
            task_description = "Given a web search query, retrieve relevant passages that answer the query"
            formatted_texts = [get_detailed_instruct(task_description, text) for text in texts]
        else:
            # Documents don't need instruction (per Context7 docs)
            formatted_texts = texts

        return self._encode_texts(
            formatted_texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings
        )