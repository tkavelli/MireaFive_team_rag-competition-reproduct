"""
Baseline ретривер на основе эмбеддингов и FAISS

Реализует семантический поиск документов с помощью:
1. Sentence Transformers эмбеддингов
2. FAISS индекса для быстрого поиска
3. Гибридного поиска (семантика + BM25)
"""

import inspect
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BitsAndBytesConfig
from models.qwen3_embedding import Qwen3EmbeddingModel

# Memory optimization utilities
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """Базовый ретривер на эмбеддингах"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        load_in_8bit: bool = False
    ):
        """
        Инициализация ретривера

        Args:
            model_name: Название модели эмбеддингов
            device: Устройство ('cuda', 'cpu' или None для автоопределения)
            batch_size: Размер батча для эмбеддингов
            load_in_8bit: Загрузить модель в INT8 квантизации (экономит VRAM)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.load_in_8bit = load_in_8bit

        # Auto-detect device (CUDA if available, else CPU)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Загрузка модели: {model_name}")
        logger.info(f"Устройство: {self.device}")
        if load_in_8bit:
            logger.info(f"INT8 квантизация: включена (экономия VRAM)")

        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")

        # Load model with proper handling for Qwen3
        if "Qwen3-Embedding" in model_name:
            # Use custom Qwen3 wrapper with AutoModel
            logger.info("Using Qwen3EmbeddingModel wrapper for Qwen3-Embedding")
            self.model = Qwen3EmbeddingModel(
                model_name=model_name,
                device=self.device
            )
        elif load_in_8bit and self.device == 'cuda':
            # Use BitsAndBytesConfig for proper INT8 quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            model_kwargs = {
                'quantization_config': quantization_config,
                'device_map': 'auto'
            }
            self.model = SentenceTransformer(
                model_name,
                model_kwargs=model_kwargs
            )
        else:
            self.model = SentenceTransformer(model_name, device=self.device)

        # Хранилище данных
        self.chunks_df = None
        self.embeddings = None
        self.index = None
        self.is_built = False

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги с оптимизацией памяти

        Args:
            texts: Список текстов
            show_progress: Показывать прогресс

        Returns:
            Матрица эмбеддингов
        """
        logger.info(f"Кодирование {len(texts)} текстов с batch_size={self.batch_size}...")

        # Clear memory before encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # For Qwen3 models, use memory-efficient encoding
        if isinstance(self.model, Qwen3EmbeddingModel):
            logger.info("Using memory-efficient encoding for Qwen3 model")
            return self._encode_qwen3_memory_efficient(texts, show_progress)

        # For SentenceTransformer models, use standard encoding
        encode_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
        }
        signature = inspect.signature(self.model.encode)
        params = signature.parameters

        if "show_progress" in params:
            encode_kwargs["show_progress"] = show_progress
        elif "show_progress_bar" in params:
            encode_kwargs["show_progress_bar"] = show_progress
        elif "progress_bar" in params:
            encode_kwargs["progress_bar"] = show_progress

        embeddings = self.model.encode(texts, **encode_kwargs)

        # Clear cache after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Получены эмбеддинги shape: {embeddings.shape}")
        return embeddings

    def _encode_qwen3_memory_efficient(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Memory-efficient encoding for Qwen3 models with batch processing
        """
        all_embeddings = []
        texts_processed = 0

        # Process in smaller batches to reduce memory usage
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Process this batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress=show_progress,
                normalize_embeddings=True
            )

            all_embeddings.append(batch_embeddings)
            texts_processed += len(batch_texts)

            if show_progress and texts_processed % 1000 == 0:
                logger.info(f"Processed {texts_processed}/{len(texts)} texts")

            # Explicit cleanup after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return np.vstack(all_embeddings)

    def build_index(self, chunks_df: pd.DataFrame, text_column: str = 'chunk_text') -> None:
        """
        Строит FAISS индекс для поиска

        Args:
            chunks_df: Датафрейм с чанками
            text_column: Название колонки с текстом
        """
        logger.info("Построение поискового индекса...")

        self.chunks_df = chunks_df.copy()

        # Кодируем все чанки
        texts = chunks_df[text_column].fillna('').astype(str).tolist()
        self.embeddings = self.encode_texts(texts)

        # Создаем FAISS индекс (IndexFlatIP для cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Добавляем эмбеддинги в индекс
        self.index.add(self.embeddings.astype('float32'))

        self.is_built = True
        logger.info(f"Индекс построен. Размерность: {dimension}, документов: {len(self.embeddings)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_scores: bool = True
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Ищет релевантные чанки по запросу

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            include_scores: Включать ли скоры в результат

        Returns:
            Список ID чанков или кортежей (id, score)
        """
        if not self.is_built:
            raise ValueError("Индекс не построен. Сначала вызовите build_index()")

        # Кодируем запрос (is_query=True для Qwen3 instruction prompts)
        query_embedding = self.model.encode([query], normalize_embeddings=True, is_query=True)
        query_embedding = query_embedding.astype('float32')

        # Ищем в индексе
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks_df)))

        # Формируем результат
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS возвращает -1 для недостающих элементов
                if include_scores:
                    results.append((int(idx), float(score)))
                else:
                    results.append(int(idx))

        return results

    def get_chunk_info(self, chunk_ids: List[int]) -> pd.DataFrame:
        """
        Возвращает информацию о чанках по их ID

        Args:
            chunk_ids: Список ID чанков

        Returns:
            Датафрейм с информацией о чанках
        """
        if not self.is_built:
            raise ValueError("Индекс не построен")

        return self.chunks_df.iloc[chunk_ids].copy()

    def save_index(self, save_dir: str) -> None:
        """
        Сохраняет индекс и модель

        Args:
            save_dir: Директория для сохранения
        """
        if not self.is_built:
            raise ValueError("Индекс не построен")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем FAISS индекс
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))

        # Сохраняем эмбеддинги
        np.save(save_path / "embeddings.npy", self.embeddings)

        # Сохраняем датафрейм
        self.chunks_df.to_csv(save_path / "chunks.csv", index=False)

        # Сохраняем метаданные
        metadata = {
            'model_name': self.model_name,
            'dimension': self.embeddings.shape[1],
            'num_chunks': len(self.chunks_df),
            'device': self.device
        }

        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Индекс сохранен в {save_path}")

    def load_index(self, save_dir: str) -> None:
        """
        Загружает сохраненный индекс

        Args:
            save_dir: Директория с сохраненным индексом
        """
        save_path = Path(save_dir)

        # Загружаем метаданные
        with open(save_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        if metadata['model_name'] != self.model_name:
            logger.warning(f"Модель изменилась: {metadata['model_name']} -> {self.model_name}")

        # Загружаем FAISS индекс
        self.index = faiss.read_index(str(save_path / "faiss_index.bin"))

        # Загружаем эмбеддинги
        self.embeddings = np.load(save_path / "embeddings.npy")

        # Загружаем датафрейм
        self.chunks_df = pd.read_csv(save_path / "chunks.csv")

        self.is_built = True
        logger.info(f"Индекс загружен из {save_path}")


class HybridRetriever(EmbeddingRetriever):
    """Гибридный ретривер: эмбеддинги + BM25"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", **kwargs):
        super().__init__(model_name, **kwargs)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def build_index(self, chunks_df: pd.DataFrame, text_column: str = 'chunk_text') -> None:
        """
        Строит гибридный индекс (эмбеддинги + TF-IDF)

        Args:
            chunks_df: Датафрейм с чанками
            text_column: Название колонки с текстом
        """
        # Строим эмбеддинг индекс
        super().build_index(chunks_df, text_column)

        # Строим TF-IDF для BM25
        logger.info("Построение TF-IDF индекса...")
        texts = chunks_df[text_column].fillna('').astype(str).tolist()

        # Используем русский анализатор
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,  # Можно добавить русские стоп-слова
            lowercase=True
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF матрица построена: {self.tfidf_matrix.shape}")

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Гибридный поиск: комбинирует семантический и BM25

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            semantic_weight: Вес семантического поиска
            bm25_weight: Вес BM25 поиска

        Returns:
            Список кортежей (chunk_id, combined_score)
        """
        if not self.is_built:
            raise ValueError("Индекс не построен")

        # Семантический поиск
        semantic_results = self.search(query, top_k * 2, include_scores=True)

        # BM25 поиск через TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query])
        bm25_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # Комбинируем результаты
        combined_scores = {}

        # Добавляем семантические скоры
        for chunk_id, semantic_score in semantic_results:
            combined_scores[chunk_id] = semantic_weight * semantic_score

        # Добавляем BM25 скоры
        for chunk_id, bm25_score in enumerate(bm25_scores):
            if bm25_score > 0:
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += bm25_weight * bm25_score
                else:
                    combined_scores[chunk_id] = bm25_weight * bm25_score

        # Сортируем и возвращаем топ-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


def main():
    """Тестирование ретривера"""
    import argparse

    parser = argparse.ArgumentParser(description="Построение и тестирование ретривера")
    parser.add_argument("--chunks", required=True, help="Путь к CSV с чанками")
    parser.add_argument("--output", required=True, help="Директория для сохранения индекса")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="Модель эмбеддингов")
    parser.add_argument("--query", help="Тестовый запрос")
    parser.add_argument("--top_k", type=int, default=5, help="Количество результатов")
    parser.add_argument("--limit", type=int, help="Лимит чанков для обработки")

    args = parser.parse_args()

    # Загрузка чанков
    logger.info(f"Загрузка чанков из {args.chunks}")
    chunks_df = pd.read_csv(args.chunks)

    if args.limit:
        chunks_df = chunks_df.head(args.limit)
        logger.info(f"Ограничено до {args.limit} чанков")

    # Создание ретривера
    retriever = HybridRetriever(model_name=args.model)

    # Построение индекса
    retriever.build_index(chunks_df)

    # Сохранение индекса
    retriever.save_index(args.output)

    # Тестовый поиск
    if args.query:
        logger.info(f"\nТестовый запрос: {args.query}")
        results = retriever.search_hybrid(args.query, top_k=args.top_k)

        logger.info(f"\nТоп-{args.top_k} результатов:")
        for i, (chunk_id, score) in enumerate(results, 1):
            chunk_info = retriever.get_chunk_info([chunk_id]).iloc[0]
            logger.info(f"\n{i}. Score: {score:.4f}")
            logger.info(f"   Title: {chunk_info['title']}")
            logger.info(f"   Text: {chunk_info['chunk_text'][:200]}...")


if __name__ == "__main__":
    main()
