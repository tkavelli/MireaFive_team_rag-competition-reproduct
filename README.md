# Alfa Bank RAG Competition Solution

**Hit@5: 36.40%** - Полностью воспроизводимое решение с Docker контейнером

##  О проекте

Это решение для соревнования по поиску релевантных фрагментов документов на сайте Альфа-Банка. Решение использует гибридный подход:

- **Dense Retrieval**: Qwen/Qwen3-Embedding-8B (1024d embeddings)
- **Sparse Retrieval**: BM25 с TF-IDF векторизацией
- **Fusion**: Convex Combination (70% semantic + 30% BM25)

##  Быстрый старт

### 1. Docker (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/tkavelli/retrieval-competition-reproduct.git
cd retrieval-competition-reproduct

# Сборка и запуск Docker контейнера
docker build -t alfa-rag-solution .
docker run --gpus all -v $(pwd)/outputs:/app/outputs alfa-rag-solution
```

### 2. Локальный запуск

```bash
# Требования:
# - Python 3.13+
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

## 📊 Результаты

- **Hit@5**: 36.40%
- **Модель**: Qwen/Qwen3-Embedding-8B
- **Время выполнения**: ~15-30 минут на RTX 4090

## 🏗️ Архитектура

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

## 📁 Структура репозитория

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
    └── submission_qwen3_rrf_v2_stable.csv
```

## ⚙️ Конфигурация

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

## 🔍 Проверка результатов

```bash
# Проверка формата submission файла
head -5 outputs/submission_qwen3_rrf_v2_stable.csv

# Проверка количества строк
wc -l outputs/submission_qwen3_rrf_v2_stable.csv
# Ожидается: 6978 (заголовок + 6977 предсказаний)
```

## 🐛 Troubleshooting

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

## 📝 Примечания

1. **Данные**: Используются официальные данные соревнования
2. **Детерминизм**: Результаты могут незначительно отличаться из-за floating-point операций
3. **VRAM**: Ожидается ~20-22GB VRAM при batch_size=2
4. **Время**: ~15-30 минут на RTX 4090

## 🤝 Контрибьюшн

Это решение создано в исследовательских целях для демонстрации полного воспроизведения результатов ML соревнования.

## 📧 Контакты

GitHub: [@tkavelli](https://github.com/tkavelli)

---

**Последнее обновление**: 2025-11-25