#!/usr/bin/env python3
"""
Генерация нескольких версий чанков с контролем имен файлов.

Примеры:
1) Одна версия:
   python scripts/generate_chunk_variants.py \
       --input websites.csv \
       --version chunk_v2 \
       --chunk-size 512 \
       --overlap 50

2) Несколько версий через JSON:
   python scripts/generate_chunk_variants.py \
       --input websites.csv \
       --config configs/chunk_variants.json

Формат JSON:
{
  "output_dir": "artifacts/chunks",
  "variants": [
    {"version": "chunk_v1", "chunk_size": 512, "overlap": 50},
    {"version": "chunk_v2", "chunk_size": 384, "overlap": 64}
  ]
}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Добавляем src в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from chunker import SmartChunker


def build_chunker(params: Dict[str, Any]) -> SmartChunker:
    return SmartChunker(
        chunk_size=params["chunk_size"],
        overlap=params["overlap"],
        min_chunk_size=params["min_chunk_size"],
        encoding_model=params["encoding_model"],
        max_chunk_size=params["max_chunk_size"],
    )


def run_variant(base_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    version = params["version"]
    output_dir = Path(params["output_dir"]) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    df = base_df
    if params["limit"]:
        df = base_df.head(params["limit"])

    chunker = build_chunker(params)
    print(f"[{version}] Разбиение {len(df)} документов (chunk_size={params['chunk_size']}, overlap={params['overlap']})")
    chunks = chunker.process_dataframe(df, progress=True)
    stats = chunker.get_statistics(chunks)

    chunks_df = chunker.chunks_to_dataframe(chunks)
    chunks_path = output_dir / f"chunks_{version}.csv"
    chunks_df.to_csv(chunks_path, index=False)

    metadata = {
        "version": version,
        "input_file": params["input"],
        "output_file": str(chunks_path),
        "params": {
            "chunk_size": params["chunk_size"],
            "max_chunk_size": params["max_chunk_size"],
            "overlap": params["overlap"],
            "min_chunk_size": params["min_chunk_size"],
            "encoding_model": params["encoding_model"],
            "limit": params["limit"],
            "notes": params["notes"],
        },
        "stats": stats,
    }

    metadata_path = output_dir / f"metadata_{version}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[{version}] Чанки сохранены: {chunks_path}")
    print(f"[{version}] Метаданные: {metadata_path}")
    print(f"[{version}] Статистика: {stats}")

    return metadata


def load_variants_from_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if isinstance(config, list):
        variants = config
        output_dir = "artifacts/chunks"
    else:
        variants = config.get("variants", [])
        output_dir = config.get("output_dir", "artifacts/chunks")

    return {"variants": variants, "output_dir": output_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация версионированных чанков")
    parser.add_argument("--input", default="websites.csv", help="Путь к файлу с документами")
    parser.add_argument("--output-dir", default="artifacts/chunks", help="Директория для сохранения версий")
    parser.add_argument("--version", help="Название версии (например chunk_v2)")
    parser.add_argument("--chunk-size", type=int, default=768, help="Размер чанка в токенах")
    parser.add_argument("--max-chunk-size", type=int, help="Жёсткий лимит токенов (по умолчанию = chunk_size)")
    parser.add_argument("--overlap", type=int, default=96, help="Перекрытие между чанками")
    parser.add_argument("--min-chunk-size", type=int, default=150, help="Минимальный размер чанка")
    parser.add_argument("--encoding-model", default="cl100k_base", help="Модель токенизатора")
    parser.add_argument("--limit", type=int, help="Ограничение числа документов")
    parser.add_argument("--notes", default="", help="Комментарий к версии")
    parser.add_argument("--config", help="JSON с набором вариантов")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    base_df = pd.read_csv(input_path)
    variants: List[Dict[str, Any]] = []

    if args.config:
        config = load_variants_from_config(Path(args.config))
        for variant in config["variants"]:
            if "version" not in variant:
                raise ValueError("Каждый вариант в конфиге должен содержать ключ 'version'")
            variants.append({
                "input": str(input_path),
                "output_dir": config["output_dir"],
                "chunk_size": variant.get("chunk_size", args.chunk_size),
                "max_chunk_size": variant.get("max_chunk_size", variant.get("chunk_size", args.chunk_size)),
                "overlap": variant.get("overlap", args.overlap),
                "min_chunk_size": variant.get("min_chunk_size", args.min_chunk_size),
                "encoding_model": variant.get("encoding_model", args.encoding_model),
                "limit": variant.get("limit", args.limit),
                "version": variant["version"],
                "notes": variant.get("notes", args.notes),
            })
    else:
        if not args.version:
            raise ValueError("Нужно указать --version, если не используется --config")
        variants.append({
            "input": str(input_path),
            "output_dir": args.output_dir,
            "chunk_size": args.chunk_size,
            "max_chunk_size": args.max_chunk_size or args.chunk_size,
            "overlap": args.overlap,
            "min_chunk_size": args.min_chunk_size,
            "encoding_model": args.encoding_model,
            "limit": args.limit,
            "version": args.version,
            "notes": args.notes,
        })

    all_metadata = []
    for params in variants:
        metadata = run_variant(base_df, params)
        all_metadata.append(metadata)

    summary_path = Path(variants[0]["output_dir"]) / "chunk_versions_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Сводный список версий сохранён в {summary_path}")


if __name__ == "__main__":
    main()
