#!/usr/bin/env python3
"""
Основной pipeline для RAG соревнования

Полный цикл:
1. Загрузка данных
2. Генерация псевдотегов (опционально)
3. Chunking документов
4. Построение ретривера
5. Поиск и создание сабмишна
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from tqdm import tqdm

# Увеличиваем лимит размера поля для CSV (по умолчанию 128KB, некоторые документы больше)
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunker import SmartChunker
from retriever import HybridRetriever


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ Fusion Functions (arXiv:2210.11934) ============

def normalize_minmax(scores_dict: Dict[int, float]) -> Dict[int, float]:
    """
    Нормализация скоров в диапазон [0, 1] методом min-max

    Args:
        scores_dict: {web_id: score}

    Returns:
        Нормализованный словарь {web_id: normalized_score}
    """
    if not scores_dict:
        return {}

    scores = list(scores_dict.values())
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero
    if max_score == min_score:
        return {web_id: 1.0 for web_id in scores_dict}

    return {
        web_id: (score - min_score) / (max_score - min_score)
        for web_id, score in scores_dict.items()
    }


def fuse_with_cc(
    query_type_scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    all_web_ids: set
) -> Dict[int, float]:
    """
    Convex Combination (CC) fusion для multi-query retrieval

    Из arXiv:2210.11934, Table 1:
    - CC > RRF в большинстве случаев
    - Sample efficient (5% данных достаточно)
    - Лучше обобщается out-of-domain (+3.6% NDCG)

    Formula:
        score_final(web_id) = Σ α_i · normalize(score_i(web_id))
        где Σ α_i = 1.0

    Args:
        query_type_scores: {"hyde": {web_id: score}, "original": {...}, ...}
        weights: {"hyde": 0.6, "original": 1.0, "expansion": 0.4}
        all_web_ids: Множество всех web_id из всех результатов

    Returns:
        Финальные скоры {web_id: fused_score}
    """
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Normalize scores for each query type
    normalized_scores = {}
    for query_type, scores in query_type_scores.items():
        if scores:  # Skip empty
            normalized_scores[query_type] = normalize_minmax(scores)

    # Fuse with Convex Combination
    final_scores = {}
    for web_id in all_web_ids:
        fused_score = 0.0
        for query_type, norm_scores in normalized_scores.items():
            if web_id in norm_scores:
                weight = normalized_weights.get(query_type, 0.0)
                fused_score += weight * norm_scores[web_id]

        final_scores[web_id] = fused_score

    return final_scores


def load_data(questions_path: str, corpus_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает вопросы и корпус документов

    Args:
        questions_path: Путь к CSV с вопросами
        corpus_path: Путь к CSV с документами

    Returns:
        Кортеж (questions_df, corpus_df)
    """
    logger.info(f"Загрузка вопросов из {questions_path}")
    questions_df = pd.read_csv(questions_path)
    logger.info(f"Загружено {len(questions_df)} вопросов")

    logger.info(f"Загрузка корпуса из {corpus_path}")
    corpus_df = pd.read_csv(corpus_path)
    logger.info(f"Загружено {len(corpus_df)} документов")

    # Базовая очистка
    corpus_df = corpus_df.dropna(subset=['text'])
    corpus_df = corpus_df[corpus_df['text'].str.len() > 50]  # Удаляем слишком короткие тексты
    logger.info(f"После очистки: {len(corpus_df)} документов")

    return questions_df, corpus_df


def generate_pseudo_tags(
    corpus_df: pd.DataFrame,
    api_key: Optional[str],
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Генерирует псевдотеги для корпуса (опционально)

    Args:
        corpus_df: Датафрейм с документами
        api_key: API ключ для LLM
        limit: Лимит документов для обработки

    Returns:
        Датафрейм с добавленными псевдотегами
    """
    if not api_key:
        logger.warning("API ключ не указан, пропускаем генерацию псевдотегов")
        return corpus_df

    # TODO: Интегрировать с generate_pseudo_tags.py
    logger.info("Генерация псевдотегов...")
    # Здесь будет вызов скрипта генерации тегов
    return corpus_df


def chunk_documents(
    corpus_df: pd.DataFrame,
    chunk_size: int = 512,
    overlap: int = 50,
    max_chunk_size: Optional[int] = None,
    min_chunk_size: int = 100,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Разбивает документы на чанки

    Args:
        corpus_df: Датафрейм с документами
        chunk_size: Размер чанка в токенах
        overlap: Перекрытие в токенах
        max_chunk_size: Максимальный размер чанка (жесткое ограничение)
        min_chunk_size: Минимальный размер чанка
        limit: Лимит документов для обработки

    Returns:
        Датафрейм с чанками
    """
    if limit:
        corpus_df = corpus_df.head(limit)
        logger.info(f"Ограничение до {limit} документов")

    logger.info(f"Разбиение документов на чанки (size={chunk_size}, overlap={overlap}, max={max_chunk_size or chunk_size})")
    chunker = SmartChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size
    )
    chunks = chunker.process_dataframe(corpus_df)

    chunks_df = chunker.chunks_to_dataframe(chunks)
    logger.info(f"Получено {len(chunks)} чанков из {len(corpus_df)} документов")

    # Статистика
    stats = chunker.get_statistics(chunks)
    logger.info(f"Статистика чанков: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    return chunks_df


def build_retriever(
    chunks_df: pd.DataFrame,
    model_name: str,
    index_path: Optional[str] = None
) -> HybridRetriever:
    """
    Строит ретривер

    Args:
        chunks_df: Датафрейм с чанками
        model_name: Название модели эмбеддингов
        index_path: Путь для сохранения индекса

    Returns:
        Объект ретривера
    """
    logger.info(f"Построение ретривера с моделью {model_name}")
    retriever = HybridRetriever(model_name=model_name)

    retriever.build_index(chunks_df)

    if index_path:
        retriever.save_index(index_path)
        logger.info(f"Индекс сохранен в {index_path}")

    return retriever


def search_and_create_submission(
    retriever: HybridRetriever,
    questions_df: pd.DataFrame,
    top_k: int = 5,
    use_hybrid: bool = True,
    hyde_answers_df: Optional[pd.DataFrame] = None,
    query_expansion_df: Optional[pd.DataFrame] = None,
    semantic_weight: float = 0.7,
    original_weight: float = 1.0,
    hyde_weight: float = 0.8,
    expansion_weight: float = 0.6
) -> pd.DataFrame:
    """
    Выполняет поиск и создает сабмишн

    Args:
        retriever: Объект ретривера
        questions_df: Датафрейм с вопросами
        top_k: Количество результатов для каждого запроса
        use_hybrid: Использовать гибридный поиск
        hyde_answers_df: Датафрейм с HyDE ответами (опционально)
        query_expansion_df: Датафрейм с парафразами запросов (опционально)
        semantic_weight: Вес семантического поиска для гибридного режима

    Returns:
        Датафрейм сабмишна в формате: q_id, web_list
        где web_list = '"[id1, id2, id3, id4, id5]"' (с кавычками!)
    """
    logger.info(f"Поиск ответов для {len(questions_df)} вопросов (топ-{top_k})")

    if hyde_answers_df is not None:
        logger.info("Используется HyDE режим (поиск по гипотетическим ответам)")
    if query_expansion_df is not None:
        logger.info("Используется Query Expansion (поиск по парафразам)")

    results = []

    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Searching"):
        q_id = row['q_id']
        original_query = row['query']

        # Собираем все запросы для поиска (каждый элемент = (тип, текст, вес))
        queries_to_search: list[tuple[str, str, float]] = []

        # 1. HyDE: используем гипотетический ответ ДОПОЛНИТЕЛЬНО к оригинальному запросу
        if hyde_answers_df is not None:
            hyde_row = hyde_answers_df[hyde_answers_df['q_id'] == q_id]
            if not hyde_row.empty and pd.notna(hyde_row.iloc[0]['hypothetical_answer']):
                hyde_answer = hyde_row.iloc[0]['hypothetical_answer']
                if hyde_answer.strip():
                    queries_to_search.append(("hyde", hyde_answer, hyde_weight))

        # 2. Query Expansion: добавляем парафразы
        if query_expansion_df is not None:
            exp_row = query_expansion_df[query_expansion_df['q_id'] == q_id]
            if not exp_row.empty:
                for col in ['paraphrase_1', 'paraphrase_2', 'paraphrase_3']:
                    if col in exp_row.columns and pd.notna(exp_row.iloc[0][col]):
                        paraphrase = exp_row.iloc[0][col]
                        if paraphrase.strip():
                            queries_to_search.append(("expansion", paraphrase, expansion_weight))

        # 3. Всегда добавляем оригинальный запрос (он имеет наивысший вес по умолчанию)
        queries_to_search.append(("original", original_query, original_weight))

        # Поиск по всем запросам и агрегация результатов
        # Собираем скоры отдельно по query_type для CC fusion
        query_type_scores = {}  # {"hyde": {web_id: score}, "original": {...}, "expansion": {...}}
        all_web_ids = set()
        weights_dict = {}

        for query_type, query_text, query_weight in queries_to_search:
            if not query_text or not query_text.strip():
                continue

            # Initialize storage for this query type
            if query_type not in query_type_scores:
                query_type_scores[query_type] = {}
                weights_dict[query_type] = query_weight

            if use_hybrid:
                search_results = retriever.search_hybrid(
                    query_text,
                    top_k=top_k * 3,  # Берём больше для агрегации
                    semantic_weight=semantic_weight
                )
            else:
                search_results = retriever.search(query_text, top_k=top_k * 3, include_scores=True)

            # Собираем скоры по query_type (Convex Combination, arXiv:2210.11934)
            for chunk_id, score in search_results:
                chunk_info = retriever.get_chunk_info([chunk_id]).iloc[0]
                web_id = int(chunk_info['web_id'])
                all_web_ids.add(web_id)

                # Берём максимальный score для каждого web_id внутри query_type
                current_score = query_type_scores[query_type].get(web_id, 0.0)
                query_type_scores[query_type][web_id] = max(current_score, score)

        # Применяем Convex Combination fusion
        fused_scores = fuse_with_cc(query_type_scores, weights_dict, all_web_ids)

        # Сортируем по скору и берём топ-k уникальных web_id
        sorted_web_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_web_ids = [web_id for web_id, score in sorted_web_ids[:top_k]]

        # Паддинг если недостаточно результатов (заполняем случайными из корпуса)
        while len(top_web_ids) < top_k:
            # Берём первый найденный или 1 по дефолту
            top_web_ids.append(top_web_ids[0] if top_web_ids else 1)

        # КРИТИЧЕСКИ ВАЖНО: правильный формат для submission
        # Нужно: '"[1, 2, 3, 4, 5]"' (с кавычками внутри строки!)
        web_list_str = str(top_web_ids[:top_k])  # "[1, 2, 3, 4, 5]"

        results.append({
            'q_id': q_id,
            'web_list': web_list_str  # pandas to_csv автоматически добавит кавычки
        })

    submission_df = pd.DataFrame(results)
    logger.info(f"Создан сабмишн с {len(submission_df)} предсказаниями")

    # Проверка формата
    sample = submission_df.iloc[0]
    logger.info(f"Пример: q_id={sample['q_id']}, web_list={sample['web_list']}")

    return submission_df


def load_experiment_config(config_path: str = "configs/experiment_config.json") -> Dict:
    """Загружает конфигурацию экспериментов"""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_experiment_params(config: Dict, chunk_version: str = None, model_name: str = None):
    """
    Получает параметры эксперимента из конфига

    Returns:
        tuple: (chunk_params, model_params, chunk_name, model_name)
    """
    chunk_params = {}
    model_params = {}
    chunk_name = chunk_version or "unknown"
    model_name_final = model_name or "unknown"

    if not config:
        return chunk_params, model_params, chunk_name, model_name_final

    # Получаем параметры чанкования
    if chunk_version and chunk_version in config.get("chunk_versions", {}):
        chunk_config = config["chunk_versions"][chunk_version]
        chunk_params = {
            "chunk_size": chunk_config["chunk_size"],
            "overlap": chunk_config["overlap"],
            "max_chunk_size": chunk_config["max_chunk_size"],
            "min_chunk_size": chunk_config["min_chunk_size"]
        }
        chunk_name = chunk_config["name"]
        logger.info(f"Loaded chunk config '{chunk_name}': {chunk_config['description']}")

    # Получаем параметры модели
    if model_name and model_name in config.get("embedding_models", {}):
        model_config = config["embedding_models"][model_name]
        model_params = {
            "model_path": model_config["model_path"]
        }
        model_name_final = model_config["name"]
        logger.info(f"Loaded model config '{model_name_final}': {model_config['description']}")

    return chunk_params, model_params, chunk_name, model_name_final


def generate_output_paths(base_output: str, chunk_name: str, model_name: str):
    """
    Генерирует пути для outputs с версионированием

    Args:
        base_output: базовый путь (например, outputs/submission.csv)
        chunk_name: название версии чанкования (ch_v1, ch_v2, etc.)
        model_name: название модели (par_multi_mpnet, bge_m3, etc.)

    Returns:
        tuple: (submission_path, chunks_path, index_path)
    """
    base_path = Path(base_output)
    parent_dir = base_path.parent

    # Генерируем имена с версиями
    version_str = f"{chunk_name}_{model_name}"

    submission_path = parent_dir / f"submission_{version_str}.csv"
    chunks_path = parent_dir / f"chunks_{version_str}.csv"
    index_path = parent_dir / f"faiss_index_{version_str}"

    logger.info(f"Generated paths for version '{version_str}':")
    logger.info(f"  Submission: {submission_path}")
    logger.info(f"  Chunks: {chunks_path}")
    logger.info(f"  Index: {index_path}")

    return str(submission_path), str(chunks_path), str(index_path)


def sanitize_component(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def generate_version_string(chunk_name: str, model_name: str, fusion_method: str, args) -> str:
    if args.experiment_name:
        return sanitize_component(args.experiment_name)

    parts = [
        chunk_name or "unknown_chunk",
        model_name or "unknown_model",
        fusion_method or "cc"
    ]
    if args.use_hybrid:
        parts.append(f"sw{args.semantic_weight:.2f}")
        parts.append(f"bm25{args.original_weight:.2f}")
    if args.hyde_answers:
        parts.append("hyde")
    if args.query_expansion:
        parts.append("qe")

    return "_".join(sanitize_component(part) for part in parts)


def main():
    parser = argparse.ArgumentParser(description="Основной pipeline для RAG соревнования")
    parser.add_argument("--queries", required=True, help="Путь к CSV с вопросами")
    parser.add_argument("--corpus", required=True, help="Путь к CSV с корпусом")
    parser.add_argument("--out", required=True, help="Путь для сохранения сабмишна")
    parser.add_argument("--top_k", type=int, default=5, help="Количество результатов (default: 5)")

    # Версионирование экспериментов (из конфига)
    parser.add_argument("--chunk_version", help="Версия чанкования из конфига (ch_v1, ch_v2, ch_v3)")
    parser.add_argument("--model_name", help="Название модели из конфига (par_multi_mpnet, bge_m3, etc.)")
    parser.add_argument("--fusion_method", choices=["cc", "rrf"], default="cc",
                       help="Метод fusion для именования outputs (Real retrieval still uses CC)")
    parser.add_argument("--experiment-name", help="Переопределить имя версии для outputs")
    parser.add_argument("--chunks_file", help="Если задано, используем этот CSV вместо повторного чанкинга")

    # Параметры чанкинга (используются если не указан --chunk_version)
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка в токенах")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие чанков в токенах")

    # Параметры модели (используются если не указан --model_name)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                       help="Модель эмбеддингов")
    parser.add_argument("--index_path", help="Путь для сохранения/загрузки индекса")

    # Параметры поиска
    parser.add_argument("--use_hybrid", action="store_true", default=True,
                       help="Использовать гибридный поиск (семантика + BM25)")
    parser.add_argument("--semantic_weight", type=float, default=0.7,
                       help="Вес семантического поиска в гибридной схеме")
    parser.add_argument("--original_weight", type=float, default=1.0,
                       help="Вес оригинального запроса при агрегации результатов")
    parser.add_argument("--hyde_weight", type=float, default=0.8,
                       help="Вес HyDE ответа при агрегации результатов")
    parser.add_argument("--expansion_weight", type=float, default=0.6,
                       help="Вес каждого парафраза при агрегации результатов")

    # HyDE и Query Expansion
    parser.add_argument("--hyde_answers", help="Путь к CSV с HyDE ответами (опционально)")
    parser.add_argument("--query_expansion", help="Путь к CSV с парафразами запросов (опционально)")

    # Параметры обработки
    parser.add_argument("--limit", type=int, help="Лимит документов для обработки")
    parser.add_argument("--api_key", help="API ключ для генерации псевдотегов")

    args = parser.parse_args()

    # Загружаем конфиг эксперимента
    config = load_experiment_config()

    # Получаем параметры из конфига (если указаны)
    chunk_params, model_params, chunk_name, model_name = get_experiment_params(
        config, args.chunk_version, args.model_name
    )

    # Применяем параметры из конфига (приоритет у конфига)
    if chunk_params:
        args.chunk_size = chunk_params["chunk_size"]
        args.overlap = chunk_params["overlap"]
        # max_chunk_size и min_chunk_size передадим в chunk_documents
    if model_params:
        args.model = model_params["model_path"]

    if args.chunks_file:
        chunk_name = Path(args.chunks_file).stem

    version_str = generate_version_string(chunk_name, model_name, args.fusion_method, args)
    submission_path, chunks_path, index_path_versioned = generate_output_paths(args.out, version_str)

    args.out = submission_path
    if not args.index_path:
        args.index_path = index_path_versioned

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Experiment Configuration ===")
    logger.info(f"Chunk version: {chunk_name}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Overlap: {args.overlap}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output template: {args.out}")
    logger.info(f"================================\n")

    # 1. Загрузка данных
    questions_df, corpus_df = load_data(args.queries, args.corpus)

    # 2. Загрузка HyDE ответов (если указаны)
    hyde_answers_df = None
    if args.hyde_answers:
        logger.info(f"Загрузка HyDE ответов из {args.hyde_answers}")
        hyde_answers_df = pd.read_csv(args.hyde_answers)
        logger.info(f"Загружено {len(hyde_answers_df)} HyDE ответов")

    # 3. Загрузка Query Expansion (если указаны)
    query_expansion_df = None
    if args.query_expansion:
        logger.info(f"Загрузка парафраз из {args.query_expansion}")
        query_expansion_df = pd.read_csv(args.query_expansion)
        logger.info(f"Загружено {len(query_expansion_df)} расширенных запросов")

    # 4. Генерация псевдотегов (опционально)
    corpus_df = generate_pseudo_tags(corpus_df, args.api_key, args.limit)

    # 5. Чанкинг (с параметрами из конфига или загрузка существующих чанков)
    max_chunk = chunk_params.get("max_chunk_size") if chunk_params else None
    min_chunk = chunk_params.get("min_chunk_size", 100) if chunk_params else 100

    if args.chunks_file:
        logger.info(f"Loading precomputed chunks from {args.chunks_file}")
        chunks_df = pd.read_csv(args.chunks_file)
    else:
        chunks_df = chunk_documents(
            corpus_df,
            args.chunk_size,
            args.overlap,
            max_chunk_size=max_chunk,
            min_chunk_size=min_chunk,
            limit=args.limit
        )

    chunks_df.to_csv(chunks_path, index=False)
    if args.chunks_file:
        logger.info(f"Применён готовый chunk-файл, сохранён копия в {chunks_path}")
    else:
        logger.info(f"Чанки сохранены в {chunks_path}")

    # 6. Построение ретривера
    retriever = build_retriever(
        chunks_df,
        args.model,
        args.index_path,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit
    )

    # 7. Поиск и создание сабмишна
    submission_df = search_and_create_submission(
        retriever,
        questions_df,
        args.top_k,
        args.use_hybrid,
        hyde_answers_df=hyde_answers_df,
        query_expansion_df=query_expansion_df,
        semantic_weight=args.semantic_weight,
        original_weight=args.original_weight,
        hyde_weight=args.hyde_weight,
        expansion_weight=args.expansion_weight
    )

    # 8. Сохранение сабмишна
    submission_df.to_csv(args.out, index=False, quoting=1)  # quoting=1 = QUOTE_ALL для правильного формата
    logger.info(f"Сабмишн сохранен в {args.out}")

    # 9. Базовая статистика
    logger.info(f"\nСтатистика сабмишна:")
    logger.info(f"  Всего предсказаний: {len(submission_df)}")
    logger.info(f"  Уникальных web_id: {len(set(str(submission_df['web_list'])))}")

    # Проверяем формат (сравниваем с sample_submission.csv)
    sample_row = submission_df.iloc[0]
    logger.info(f"  Пример формата: q_id={sample_row['q_id']}, web_list={sample_row['web_list']}")

    # Читаем первые строки сохраненного файла для проверки формата
    logger.info(f"\n Первые 3 строки сохраненного файла:")
    with open(args.out, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                logger.info(f"  {line.strip()}")

    logger.info("\nPipeline завершен успешно!")


if __name__ == "__main__":
    main()
