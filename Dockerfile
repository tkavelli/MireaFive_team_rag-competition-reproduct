FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Описание контейнера
LABEL maintainer="tkavelli"
LABEL description="Alfa Bank RAG Competition Solution - Hit@5: 36.40% reproducible"
LABEL version="1.0"

# Установка системных зависимостей
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка PyTorch 2.8.0 с CUDA 12.8
RUN pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Копирование и установка зависимостей
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Установка faiss-gpu для ускорения (fallback на cpu)
RUN pip3 install --no-cache-dir faiss-gpu==1.12.0 || \
    pip3 install --no-cache-dir faiss-cpu==1.12.0

# Создание рабочих директорий
WORKDIR /app
RUN mkdir -p /app/data /app/outputs /app/artifacts

# Копирование кода проекта
COPY src/ /app/src/
COPY pipelines/ /app/pipelines/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY run_*.sh /app/

# Права доступа
RUN chmod +x /app/run_*.sh

# Переменные окружения
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache
RUN mkdir -p /app/.hf_cache

# Команда запуска по умолчанию
CMD ["bash", "run_qwen3_rrf_v2_stable.sh"]