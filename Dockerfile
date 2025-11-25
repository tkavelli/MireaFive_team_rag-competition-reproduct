FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Описание контейнера
LABEL maintainer="tkavelli"
LABEL description="Alfa Bank RAG Competition Solution - Hit@5: 36.40% reproducible"
LABEL version="1.0"

# Установка системных зависимостей (сборка Python 3.13.7 из исходников)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Сборка и установка Python 3.13.7 (совпадает с локальным окружением пользователя)
RUN PYTHON_VERSION=3.13.7 && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# Создание виртуального окружения на Python 3.13.7
RUN /usr/local/bin/python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка PyTorch 2.8.0 с CUDA 12.8 (cp313 wheels доступны на официальном индексe)
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

# Копирование кода проекта и данных
COPY src/ /app/src/
COPY pipelines/ /app/pipelines/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY data/*.csv /app/data/
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
