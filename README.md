# Alfa Bank RAG Competition Solution

🏆 **Команда MireaFive — top 15 в финальном лидерборде**

**Hit@5: 36.40%** — Полностью воспроизводимое решение с Docker контейнером

## О проекте

Это решение для соревнования по поиску релевантных фрагментов документов на сайте Альфа-Банка. Решение использует гибридный подход:

- **Dense Retrieval**: Qwen/Qwen3-Embedding-8B (1024d embeddings)
- **Sparse Retrieval**: BM25 с TF-IDF векторизацией
- **Fusion**: Convex Combination (70% semantic + 30% BM25)

## Финальные артефакты

- Ретривер (chunk_v6, Qwen3-8B, CC 0.7/1.0): `outputs/submission_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00.csv`
- Реранкер (Qwen3 rerank from cache, pool=50, int4): `outputs/qwen3_rerank_runs/qwen3_v6_pool50_int4_fast_resume_20251127_130947/submission_qwen3_v6_pool50_int4_fast_resume_20251127_130947.csv`

## Быстрый старт

### 1. Docker (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/tkavelli/retrieval-competition-reproduct.git
cd retrieval-competition-reproduct

# Сборка контейнера (Ubuntu 24.04 + CUDA 12.8)
docker build -t alfa-rag-solution .

# Запуск. Данные уже упакованы в образ, выносим только результаты и кэш HF (опционально)
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.hf_cache:/app/.hf_cache \
  alfa-rag-solution

# Примечание: сборка занимает 3–10 минут из-за компиляции Python 3.13.7 внутри образа.
# Если нужен подробный лог, включите BuildKit:
# DOCKER_BUILDKIT=1 docker build --progress=plain -t alfa-rag-solution .
```

Если хотите использовать локальные CSV вместо упакованных в образ:

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.hf_cache:/app/.hf_cache \
  alfa-rag-solution \
  bash -c "cd /app && bash run_qwen3_rrf_v2_stable.sh"
```

### 2. Локальный запуск

```bash
# Требования:
# - Python 3.13.7 (точно как в Docker)
# - NVIDIA GPU с 24GB+ VRAM
# - CUDA 12.8+
# - GPU с поддержкой FA 2

# Установка зависимостей
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Установка PyTorch с CUDA
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Запуск pipeline
bash run_qwen3_rrf_v2_stable.sh
```

## Результаты

- **Hit@5**: 36.40%
- **Модель**: Qwen/Qwen3-Embedding-8B
- **Время выполнения**: ~15-30 минут на RTX 4090

## Архитектура

```
questions_clean.csv (6,977 запросов)
         ↓
    [SmartChunker]
         ↓
websites.csv → chunks (2048 токенов, overlap 200) → ~4,242 чанков
         ↓
[Qwen3-Embedding-8B]
         ↓
   FAISS index ← embeddings
         ↓
[Hybrid Retrieval: FAISS (70%) + BM25 (30%)]
         ↓
[Convex Combination Fusion]
         ↓
   Top-5 результатов для каждого запроса
         ↓
submission.csv
```

## Структура репозитория

```
retrieval-competition-reproduct/
├── Dockerfile                    # Docker контейнер
├── requirements.txt              # Python зависимости
├── run_qwen3_rrf_v2_stable.sh   # Основной скрипт запуска
├── REPRODUCE.md                  # Детальная инструкция воспроизведения
├── README.md                     # Этот файл
│
├── data/                         # Данные соревнования
│   ├── questions_clean.csv       # ~6,977 вопросов
│   ├── websites.csv             # ~1,937 документов
│   └── sample_submission.csv    # Пример формата
│
├── src/                          # Основной код
│   ├── chunker.py               # Чанкование документов
│   ├── retriever.py             # Гибридный поиск
│   └── models/
│       ├── embedding_models.py  # Конфигурация моделей
│       └── qwen3_embedding_stable.py  # Qwen3 wrapper
│
├── pipelines/
│   └── retrieve.py              # Основной pipeline
│
└── outputs/                     # Результаты (создаются после запуска)
    ├── submission_ch_v5_qwen3_8b.csv
    ├── chunks_ch_v5_qwen3_8b.csv
    └── faiss_index_ch_v5_qwen3_8b/
        ├── embeddings.npy
        ├── faiss_index.bin
        ├── chunks.csv
        └── metadata.json
```

## Конфигурация

### Ключевые параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--model` | `Qwen/Qwen3-Embedding-8B` | Embedding модель |
| `--chunk_size` | `2048` | Размер чанков в токенах |
| `--overlap` | `200` | Перекрытие между чанками |
| `--batch_size` | `2` | Batch size (для 24GB VRAM) |
| `--use_hybrid` | `true` | Включает BM25 + FAISS гибрид |

### Fusion параметры

- `semantic_weight`: 0.7 (FAISS)
- `bm25_weight`: 0.3 (BM25)
- Fusion method: Convex Combination

## Проверка результатов

```bash
# Проверка формата submission файла
head -5 outputs/submission_ch_v5_qwen3_8b.csv

# Проверка количества строк
wc -l outputs/submission_ch_v5_qwen3_8b.csv
# Ожидается: 6978 (заголовок + 6977 предсказаний)
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Уменьшить batch_size в run_qwen3_rrf_v2_stable.sh
--batch_size 1  # вместо 2
```

### Медленная работа
```bash
# Проверить GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Docker GPU поддержка
```bash
# Установка NVIDIA Container Toolkit
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Примечания

1. **Данные**: Используются официальные данные соревнования
2. **Детерминизм**: Результаты могут незначительно отличаться из-за floating-point операций
3. **VRAM**: Ожидается ~20-22GB VRAM при batch_size=2
4. **Время**: ~15-30 минут на RTX 4090

## Контрибьюшн

Это решение создано в исследовательских целях для демонстрации полного воспроизведения результатов ML соревнования.

## Контакты

GitHub: [@tkavelli](https://github.com/tkavelli)
Telegram [@Nikolay_Bubnov]

## Pending / Not Yet Integrated

- **Qwen3 reranking (отдельный шаг):**
  - Скрипт: `scripts/run_qwen3_rerank_from_cache.py` (поддерживает int4 через bitsandbytes).
  - План: брать готовый FAISS из `outputs/faiss_index_ch_v5_qwen3_8b` (или свой), пул 100–120 кандидатов, собирать контекст из 2–3 топ-чанков + соседей до ~6k токенов, реранкер `Qwen/Qwen3-Reranker-8B` (или 4B) в int4, `batch_size` 4–6, `rerank_to=5`.
  - Запуск (пример):
    ```bash
    python scripts/run_qwen3_rerank_from_cache.py \
      --queries data/questions_clean.csv \
      --index-dir outputs/faiss_index_ch_v5_qwen3_8b \
      --pool-size 120 --rerank-to 5 \
      --reranker-model Qwen/Qwen3-Reranker-8B \
      --reranker-quantization int4 \
      --max-doc-tokens 6000 --chunks-per-doc 3 --max-chunks-per-doc 5
    ```

- **Альтернативные чанки (v6 и др.):**
  - Генерация: `scripts/generate_chunk_variants.py` (можно через JSON-конфиг). Рекомендуемый старт для реранка — chunk_size ~1000–1200, overlap 80–120.
  - Текущий baseline пересчитывает v5 внутри `run_qwen3_rrf_v2_stable.sh`. Чтобы использовать готовые чанки/индекс, передайте `--chunks-file` и `--index_path` в `pipelines/retrieve.py` или соберите FAISS из CSV перед реранком.

---

**Последнее обновление**: 2025-11-25

## Двухэтапный запуск (ветка `feature/two-stage-docker`)
Быстрая репликация готовых артефактов `chunk_v6 + Qwen3-8B` и финального сабмишна через реранк из кеша.

```bash
# 1) Ретривер: копирует готовые чанки и FAISS-индекс
docker build -f Dockerfile.retriever -t retriever-cached .
docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs retriever-cached

# 2) Реранкер из кеша (Qwen3-Embedding-8B + Qwen3-Reranker-8B int4)
docker build -f Dockerfile.reranker -t qwen3-reranker .
docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs qwen3-reranker

# Результаты:
#  - outputs/submission_chunks_chunk_v6_semantic768_qwen3_8b_cc_sw0.70_bm251.00.csv (retriever)
#  - outputs/qwen3_rerank_runs/<run>/submission_*.csv (после реранка)
```

Тюнинг через переменные окружения при `docker run`: `OUTPUT_DIR`, `INDEX_DIR`, `POOL_SIZE`, `RERANK_TO`, `EMBED_BATCH_SIZE`, `RERANKER_BATCH_SIZE`, `RUN_TAG`.
