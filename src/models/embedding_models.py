"""
Embedding models configuration and factory for RAG competition.

Supported models:
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (baseline)
- intfloat/multilingual-e5-large
- BAAI/bge-m3
- google/embeddinggemma-300m
- Qwen/Qwen3-Embedding-8B (GGUF via llama.cpp)
"""

from typing import Dict, Any, Optional, List
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json


class EmbeddingModelConfig:
    """Configuration for embedding models."""

    MODELS = {
        "mpnet-base": {
            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "params": 278_000_000,
            "dimensions": 768,
            "max_seq_length": 512,
            "description": "Baseline model, good multilingual performance",
            "license": "Apache 2.0",
            "access": "public"
        },
        "e5-large": {
            "name": "intfloat/multilingual-e5-large",
            "params": 560_000_000,
            "dimensions": 1024,
            "max_seq_length": 512,
            "description": "Strong multilingual performance, larger embeddings",
            "license": "MIT",
            "access": "public"
        },
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "params": 568_000_000,
            "dimensions": 1024,
            "max_seq_length": 8192,
            "description": "Excellent for Russian, very long context",
            "license": "MIT",
            "access": "public"
        },
        "gemma-300m": {
            "name": "google/embeddinggemma-300m",
            "params": 300_000_000,
            "dimensions": 768,
            "max_seq_length": 2048,
            "description": "MTEB leader <500M params, performance comparable to 560M models",
            "license": "gemma",  # Custom Google license
            "access": "hf_login_required"
        },
        "qwen3-embedding-8b": {
            "name": "Qwen/Qwen3-Embedding-8B",
            "params": 8_000_000_000,
            "dimensions": 4096,
            "max_seq_length": 32768,
            "description": "MTEB multilingual #1 (70.58), 32k context, optimized for Russian",
            "license": "Apache 2.0",
            "access": "public",
            "needs_instruction": True,
            "task_instruction": "Given a web search query, retrieve relevant passages"
        }
    }

    @classmethod
    def get_model_config(cls, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_id not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_id]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model IDs."""
        return list(cls.MODELS.keys())


class EmbeddingModel:
    """Wrapper for sentence-transformers models with consistent interface."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        """
        Initialize embedding model.

        Args:
            model_id: Model identifier from EmbeddingModelConfig
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_id = model_id
        self.config = EmbeddingModelConfig.get_model_config(model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Special handling for different models
        if model_id == "gemma-300m":
            self._load_gemma_model()
        elif model_id == "qwen3-embedding-8b":
            self._load_qwen3_model()
        else:
            self.model = SentenceTransformer(self.config["name"], device=device)

        self.embedding_dim = self.config["dimensions"]

    def _load_gemma_model(self):
        """Load gemma-300m with specific requirements."""
        # Note: gemma-300m requires float32 or bfloat16, not float16
        # This may need special handling
        try:
            self.model = SentenceTransformer(
                self.config["name"],
                device=self.device,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load gemma-300m: {e}")
            print("Make sure you're logged into Hugging Face: huggingface-cli login")
            raise

    def _load_qwen3_model(self):
        """Load Qwen3-Embedding-8B with flash_attention_2 and proper configuration."""
        try:
            self.model = SentenceTransformer(
                self.config["name"],
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "device_map": "auto"
                },
                tokenizer_kwargs={"padding_side": "left"},
                trust_remote_code=True
            )
            print(f"Qwen3-Embedding-8B loaded with flash_attention_2 on {self.device}")
        except Exception as e:
            print(f"Failed to load Qwen3-Embedding-8B: {e}")
            print("Trying fallback without flash_attention_2...")
            try:
                self.model = SentenceTransformer(
                    self.config["name"],
                    device=self.device,
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise

    def _format_qwen3_input(self, texts: List[str], is_query: bool = False) -> List[str]:
        """Format texts for Qwen3 with instruction prompts."""
        if self.model_id != "qwen3-embedding-8b":
            return texts

        if not is_query:
            # Documents don't need instruction prompts
            return texts

        # Add instruction prompts for queries
        task_instruction = self.config["task_instruction"]
        formatted_texts = []
        for text in texts:
            formatted = f"Instruct: {task_instruction}\nQuery: {text}"
            formatted_texts.append(formatted)
        return formatted_texts

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize_embeddings: bool = True,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Normalize embeddings to unit length
            is_query: Whether texts are queries (affects Qwen3 formatting)

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        # Format inputs for Qwen3 if needed
        formatted_texts = self._format_qwen3_input(texts, is_query)

        # Special handling for different models
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress,
            "normalize_embeddings": normalize_embeddings
        }

        if self.model_id == "gemma-300m":
            # gemma-300m requires specific dtype handling
            kwargs["convert_to_tensor"] = False
        elif self.model_id == "qwen3-embedding-8b":
            # Use smaller batch sizes for Qwen3 due to memory constraints
            kwargs["batch_size"] = min(batch_size, 8)

        embeddings = self.model.encode(formatted_texts, **kwargs)

        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "model_name": self.config["name"],
            "parameters": self.config["params"],
            "dimensions": self.config["dimensions"],
            "max_seq_length": self.config["max_seq_length"],
            "description": self.config["description"],
            "license": self.config["license"],
            "access_requirements": self.config["access"],
            "device": self.device
        }


class EmbeddingModelFactory:
    """Factory for creating embedding models."""

    @staticmethod
    def create_model(model_id: str, device: Optional[str] = None) -> EmbeddingModel:
        """Create an embedding model instance."""
        return EmbeddingModel(model_id, device)

    @staticmethod
    def create_baseline_model(device: Optional[str] = None) -> EmbeddingModel:
        """Create the baseline embedding model."""
        return EmbeddingModel("mpnet-base", device)

    @staticmethod
    def get_recommended_models() -> List[str]:
        """Get recommended models for this competition."""
        return [
            "mpnet-base",              # Baseline, reliable
            "e5-large",                # Strong multilingual
            "bge-m3",                  # Best for Russian
            "qwen3-embedding-8b",      # MTEB #1, 32k context, 4096d
            "gemma-300m"               # Compact, efficient
        ]


def compare_models(model_ids: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare specifications of embedding models.

    Args:
        model_ids: List of model IDs to compare, None for all

    Returns:
        Dictionary with model comparisons
    """
    if model_ids is None:
        model_ids = EmbeddingModelConfig.list_models()

    comparison = {}
    for model_id in model_ids:
        config = EmbeddingModelConfig.get_model_config(model_id)
        comparison[model_id] = {
            "name": config["name"],
            "params_millions": config["params"] // 1_000_000,
            "dimensions": config["dimensions"],
            "max_seq_length": config["max_seq_length"],
            "description": config["description"],
            "license": config["license"],
            "access": config["access"]
        }

    return comparison


if __name__ == "__main__":
    # Example usage and model comparison
    print("Embedding Models Available:")
    print("=" * 50)

    comparison = compare_models()
    for model_id, info in comparison.items():
        print(f"\n{model_id}:")
        print(f"  Name: {info['name']}")
        print(f"  Parameters: {info['params_millions']}M")
        print(f"  Dimensions: {info['dimensions']}")
        print(f"  Max Seq Length: {info['max_seq_length']}")
        print(f"  License: {info['license']}")
        print(f"  Access: {info['access']}")
        print(f"  Description: {info['description']}")

    # Test model loading (commented out to avoid actual downloads)
    # print("\nTesting model loading...")
    # model = EmbeddingModelFactory.create_baseline_model()
    # print(f"Baseline model loaded: {model.get_model_info()}")