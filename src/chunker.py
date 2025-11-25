"""
Разбиение документов на чанки с учетом псевдотегов

Умный чанкинг:
1. Разделяет текст по семантическим границам
2. Сохраняет контекст через overlap
3. Учитывает псевдотеги для лучшей релевантности
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import pandas as pd
from tqdm import tqdm


@dataclass
class Chunk:
    """Структура для хранения чанка"""
    chunk_id: str
    web_id: int
    chunk_text: str
    chunk_index: int
    start_char: int
    end_char: int
    title: str
    url: str
    tags: List[str] = None
    section_title: str = ""
    token_count: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SmartChunker:
    """Умный чанкер для банковских документов"""

    def __init__(
        self,
        chunk_size: int = 768,
        overlap: int = 96,
        min_chunk_size: int = 150,
        encoding_model: str = "cl100k_base",
        max_chunk_size: Optional[int] = None
    ):
        """
        Инициализация чанкера

        Args:
            chunk_size: Размер чанка в токенах
            overlap: Перекрытие между чанками в токенах
            min_chunk_size: Минимальный размер чанка в токенах
            encoding_model: Модель для подсчета токенов
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or chunk_size
        self.tokenizer = tiktoken.get_encoding(encoding_model)

    def count_tokens(self, text: str) -> int:
        """Считает количество токенов в тексте"""
        return len(self.tokenizer.encode(text))

    def clean_text(self, text: str) -> str:
        """Очищает текст и приводит к читабельному виду, сохраняя структуру"""
        if not text or pd.isna(text):
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Блоковые HTML-теги превращаем в переносы строк
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</?(p|div|section|article|li|ul|ol|h[1-6])[^>]*>", "\n\n", text, flags=re.IGNORECASE)
        # Удаляем скрипты/стили полностью
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        # Остальные теги заменяем на пробел
        text = re.sub(r"<[^>]+>", " ", text)
        # Схлопываем табы/множественные пробелы
        text = re.sub(r"[ \t]+", " ", text)
        # Сохраняем абзацы: более двух переносов -> двойной перенос
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_by_sections(self, text: str, sections: List[Dict]) -> List[Dict]:
        """
        Разделяет текст на секции согласно псевдотегам

        Args:
            text: Исходный текст
            sections: Секции из псевдотегов

        Returns:
            Список секций с текстом
        """
        if not sections:
            return [{"title": "Основной контент", "text": text, "start": 0, "end": len(text)}]

        # Сортируем секции по start_idx
        sections = sorted(sections, key=lambda x: x.get('start_idx', 0))

        result_sections = []
        for i, section in enumerate(sections):
            start_idx = section.get('start_idx', 0)
            end_idx = section.get('end_idx', len(text))

            # Ограничиваем границы
            start_idx = max(0, start_idx)
            end_idx = min(len(text), end_idx)

            if start_idx < end_idx:
                section_text = text[start_idx:end_idx]
                title = section.get('title', f'Раздел {i+1}')

                result_sections.append({
                    'title': title,
                    'text': section_text,
                    'start': start_idx,
                    'end': end_idx
                })

        # Если секции пустые, возвращаем весь текст
        if not result_sections:
            result_sections = [{"title": "Основной контент", "text": text, "start": 0, "end": len(text)}]

        return result_sections

    def _split_text_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Делит текст на части по количеству токенов"""
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []

        if overlap_tokens >= max_tokens:
            overlap_tokens = 0

        step = max_tokens - overlap_tokens if overlap_tokens else max_tokens
        segments = []

        for start in range(0, len(tokens), step):
            segment_tokens = tokens[start:start + max_tokens]
            segments.append(self.tokenizer.decode(segment_tokens))
            if start + max_tokens >= len(tokens):
                break

        return segments

    def _get_overlap_tail(self, text: str) -> str:
        """Возвращает хвост текста для перекрытия"""
        if self.overlap <= 0:
            return ""

        tokens = self.tokenizer.encode(text)
        if not tokens:
            return ""

        tail_tokens = tokens[-self.overlap:]
        return self.tokenizer.decode(tail_tokens)

    def split_into_chunks(self, text: str, section_info: Dict = None) -> List[str]:
        """
        Разделяет текст на чанки с учетом границ предложений

        Args:
            text: Текст для разделения
            section_info: Информация о секции

        Returns:
            Список чанков
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # Разделяем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for raw_sentence in sentences:
            if not raw_sentence:
                continue

            sentence_segments = [raw_sentence]
            sentence_token_count = self.count_tokens(raw_sentence)

            if sentence_token_count > self.max_chunk_size:
                sentence_segments = self._split_text_by_tokens(
                    raw_sentence,
                    self.max_chunk_size
                )

            for sentence in sentence_segments:
                sentence_tokens = self.count_tokens(sentence)

                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = self._get_overlap_tail(current_chunk)

                    if overlap_text:
                        current_chunk = f"{overlap_text} {sentence}".strip()
                        current_tokens = self.count_tokens(current_chunk)
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    if current_chunk:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        current_chunk = sentence.strip()
                    current_tokens = self.count_tokens(current_chunk)

        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # КРИТИЧНО: жесткая валидация всех чанков на max_chunk_size
        # Это исправляет баг с чанками на 14k+ токенов
        validated_chunks = []
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens > self.max_chunk_size:
                # Форсированно разбиваем на части
                sub_chunks = self._split_text_by_tokens(chunk, self.max_chunk_size)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def process_document(self, row: pd.Series) -> List[Chunk]:
        """
        Обрабатывает один документ и возвращает чанки

        Args:
            row: Строка датафрейма с данными документа

        Returns:
            Список чанков
        """
        web_id = row['web_id']
        title = row.get('title', '')
        url = row.get('url', '')
        text = self.clean_text(row.get('text', ''))
        tags = row.get('pseudo_tags', [])
        sections = row.get('sections', [])

        # Если sections строка, парсим JSON
        if isinstance(sections, str):
            try:
                sections = json.loads(sections)
            except:
                sections = []

        # Если tags строка, парсим
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = []

        # Разделяем на секции
        text_sections = self.split_by_sections(text, sections)

        chunks = []
        chunk_index = 0

        for section in text_sections:
            section_title = section['title']
            section_text = section['text']
            section_start = section['start']

            # Разделяем секцию на чанки
            section_chunks = self.split_into_chunks(section_text, section)

            for chunk_text in section_chunks:
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                # Считаем токены
                token_count = self.count_tokens(chunk_text)

                # Создаем объект чанка
                chunk = Chunk(
                    chunk_id=f"{web_id}_{chunk_index}",
                    web_id=web_id,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=section_start,
                    end_char=section_start + len(chunk_text),
                    title=title,
                    url=url,
                    tags=tags.copy(),
                    section_title=section_title,
                    token_count=token_count
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def process_dataframe(self, df: pd.DataFrame, progress: bool = True) -> List[Chunk]:
        """
        Обрабатывает датафрейм документов

        Args:
            df: Датафрейм с документами
            progress: Показывать прогресс бар

        Returns:
            Список всех чанков
        """
        all_chunks = []

        iterator = tqdm(df.iterrows(), total=len(df), desc="Chunking documents") if progress else df.iterrows()

        for _, row in iterator:
            chunks = self.process_document(row)
            all_chunks.extend(chunks)

        return all_chunks

    def chunks_to_dataframe(self, chunks: List[Chunk]) -> pd.DataFrame:
        """
        Преобразует список чанков в датафрейм

        Args:
            chunks: Список чанков

        Returns:
            Датафрейм с чанками
        """
        data = []
        for chunk in chunks:
            data.append({
                'chunk_id': chunk.chunk_id,
                'web_id': chunk.web_id,
                'chunk_text': chunk.chunk_text,
                'chunk_index': chunk.chunk_index,
                'title': chunk.title,
                'url': chunk.url,
                'section_title': chunk.section_title,
                'tags': json.dumps(chunk.tags, ensure_ascii=False),
                'token_count': chunk.token_count,
                'char_count': len(chunk.chunk_text)
            })

        return pd.DataFrame(data)

    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """
        Сохраняет чанки в CSV файл

        Args:
            chunks: Список чанков
            output_path: Путь для сохранения
        """
        df = self.chunks_to_dataframe(chunks)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Сохранено {len(chunks)} чанков в {output_path}")

    def get_statistics(self, chunks: List[Chunk]) -> Dict:
        """
        Возвращает статистику по чанкам

        Args:
            chunks: Список чанков

        Returns:
            Словарь со статистикой
        """
        if not chunks:
            return {}

        token_counts = [c.token_count for c in chunks]
        char_counts = [len(c.chunk_text) for c in chunks]
        unique_docs = len(set(c.web_id for c in chunks))

        stats = {
            'total_chunks': len(chunks),
            'unique_documents': unique_docs,
            'chunks_per_document': len(chunks) / unique_docs,
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens_per_chunk': min(token_counts),
            'max_tokens_per_chunk': max(token_counts),
            'avg_chars_per_chunk': sum(char_counts) / len(char_counts),
            'total_tokens': sum(token_counts),
            'chunks_with_tags': sum(1 for c in chunks if c.tags),
            'unique_tags': len(set(tag for chunk in chunks for tag in chunk.tags))
        }

        return stats


def main():
    """Тестирование чанкера"""
    import argparse

    parser = argparse.ArgumentParser(description="Разбиение документов на чанки")
    parser.add_argument("--input", required=True, help="Путь к CSV с документами")
    parser.add_argument("--output", required=True, help="Путь для сохранения чанков")
    parser.add_argument("--chunk_size", type=int, default=768, help="Размер чанка в токенах")
    parser.add_argument("--overlap", type=int, default=96, help="Перекрытие в токенах")
    parser.add_argument("--min_chunk_size", type=int, default=150, help="Минимальный размер чанка")
    parser.add_argument("--limit", type=int, help="Лимит документов для обработки")

    args = parser.parse_args()

    # Загрузка данных
    print(f"Загрузка данных из {args.input}")
    df = pd.read_csv(args.input)

    if args.limit:
        df = df.head(args.limit)
        print(f"Ограничено до {args.limit} документов")

    # Инициализация чанкера
    chunker = SmartChunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size
    )

    # Обработка
    print("Начинаем разбиение на чанки...")
    chunks = chunker.process_dataframe(df)

    # Сохранение
    chunker.save_chunks(chunks, args.output)

    # Статистика
    stats = chunker.get_statistics(chunks)
    print("\nСтатистика:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
