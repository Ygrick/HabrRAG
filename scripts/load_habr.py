import argparse
from typing import List, Optional

import torch
from datasets import load_dataset
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from src.chunking import chunk_documents
from src.logger import logger
from src.retrievers import _prepare_qdrant_collection
from src.settings import app_settings


def _init_qdrant_client() -> QdrantClient:
    qdrant_settings = app_settings.qdrant
    qdrant_kwargs = {"prefer_grpc": qdrant_settings.prefer_grpc}
    if qdrant_settings.url:
        qdrant_kwargs["url"] = qdrant_settings.url
        if qdrant_settings.api_key and qdrant_settings.api_key.get_secret_value():
            qdrant_kwargs["api_key"] = qdrant_settings.api_key.get_secret_value()
    else:
        qdrant_kwargs["path"] = qdrant_settings.path
    return QdrantClient(**qdrant_kwargs)


def _pick_texts(dataset, field: Optional[str]) -> List[str]:
    # Явно заданное поле
    if field:
        if field in dataset.column_names:
            return dataset[field]
        raise ValueError(f"В датасете нет поля '{field}'. Доступные колонки: {dataset.column_names}")

    # Эвристики для популярных форматов
    candidates = [
        "text_markdown",
        "context",
        "text",
        "content",
        "article",
        "body",
    ]
    for name in candidates:
        if name in dataset.column_names:
            return dataset[name]

    # Комбинации полей (например title + text)
    if "title" in dataset.column_names and "text" in dataset.column_names:
        return [f"{t}\n\n{x}" for t, x in zip(dataset["title"], dataset["text"])]

    raise ValueError(
        f"Не удалось автоматически определить текстовое поле. Доступные колонки: {dataset.column_names}. "
        "Укажите --field явно."
    )


def _load_local_jsonl_zst(path: str):
    try:
        # datasets поддерживает zstd через параметр compression='zstd'
        return load_dataset("json", data_files=path, compression="zstd", split="train")
    except Exception as e:
        # Частый случай — отсутствует пакет 'zstandard'
        if "zstandard" in str(e).lower() or "zstd" in str(e).lower():
            raise RuntimeError(
                "Для чтения *.zst требуется пакет 'zstandard'. Установите его и повторите попытку."
            ) from e
        raise


def run_ingest(
    local: Optional[str],
    hf: Optional[str],
    split: str,
    field: Optional[str],
    limit: Optional[int],
    chunk_size: int,
    chunk_overlap: int,
):
    # 1) Источник данных
    if not local and not hf:
        # По умолчанию — из настроек
        hf = app_settings.dataset
        split = split or app_settings.split_dataset

    if local:
        logger.info(f"Загрузка локального файла: {local}")
        ds = _load_local_jsonl_zst(local)
    else:
        logger.info(f"Загрузка датасета из HF: {hf} (split={split})")
        ds = load_dataset(hf, split=split)

    texts = _pick_texts(ds, field)
    if limit is not None:
        texts = texts[:limit]
        logger.info(f"Ограничение по числу документов: {len(texts)}")
    else:
        logger.info(f"Всего документов: {len(texts)}")

    # 2) Чанкинг
    chunks: List[Document] = chunk_documents(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 3) Qdrant + индексация
    client = _init_qdrant_client()
    try:
        logger.info("Инициализация модели эмбеддингов...")
        embedding = HuggingFaceEmbeddings(
            model_name=app_settings.retrieval.embedding,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        _prepare_qdrant_collection(client, chunks, embedding)
        logger.info(
            f"Коллекция '{app_settings.qdrant.collection_name}' готова к использованию. Чанков: {len(chunks)}"
        )
    finally:
        # Освобождение GPU, закрытие клиента
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Первичная загрузка данных Habr в Qdrant: локальный habr.jsonl.zst или датасет HF. "
            "Настройки Qdrant и моделей берутся из src.settings.app_settings"
        )
    )

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--local", type=str, help="Путь к локальному habr.jsonl.zst")
    src.add_argument("--hf", type=str, help="Имя датасета на Hugging Face (напр. IlyaGusev/habr)")

    parser.add_argument("--split", type=str, default=app_settings.split_dataset, help="Сплит HF датасета")
    parser.add_argument("--field", type=str, default=None, help="Название текстовой колонки (если нужно задать явно)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число документов для индексации")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Размер чанка (символы)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Перекрытие чанков (символы)")

    args = parser.parse_args()

    run_ingest(
        local=args.local,
        hf=args.hf,
        split=args.split,
        field=args.field,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
