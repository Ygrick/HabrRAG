import io
import json
from pathlib import Path

import zstandard as zstd
from datasets import load_dataset

from src.logger import logger
from src.settings import app_settings


def load_local_jsonl_zst(path: str) -> list[dict]:
    """
    Загружает документы из локального файла habr.jsonl.zst.

    Args:
        path (str): Путь к файлу habr.jsonl.zst

    Returns:
        list[dict]: Список словарей с полными данными документов
    """
    logger.info(f"Чтение файла {path} с использованием zstandard...")
    documents = []

    with open(path, 'rb') as f:
        # Создаем декомпрессор
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            # Читаем и декодируем построчно
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            for line_num, line in enumerate(text_reader):
                try:
                    data = json.loads(line.strip())
                    documents.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка JSON в строке {line_num}: {e}")
                    continue

    logger.info(f"Успешно загружено {len(documents)} документов")
    return documents


def load_documents(limit: int = None) -> list[dict]:
    """
    Загружает документы из локального файла или HuggingFace датасета.

    Args:
        limit (int, optional): Максимальное количество документов для загрузки

    Returns:
        list[dict]: Список словарей с полными данными документов
    """
    local_file = Path("habr.jsonl.zst")

    if local_file.exists():
        logger.info(f"Найден локальный файл: {local_file}")
        docs = load_local_jsonl_zst(str(local_file))
        if limit:
            docs = docs[:limit]
        return docs

    logger.info(
        f"Локальный файл не найден, загрузка датасета "
        f"{app_settings.dataset} из HuggingFace..."
    )

    if app_settings.dataset == "IlyaGusev/habr":
        habr_url = (
            "https://huggingface.co/datasets/IlyaGusev/habr/resolve/"
            "main/habr.jsonl.zst"
        )
        rag_dataset = load_dataset(
            "json",
            data_files=habr_url,
            split=app_settings.split_dataset
        )
    else:
        rag_dataset = load_dataset(
            app_settings.dataset, split=app_settings.split_dataset
        )

    documents = [dict(item) for item in rag_dataset]
    if limit:
        documents = documents[:limit]
    logger.info(f"Датасет загружен: {len(documents)} документов (limit={limit})")
    return documents
