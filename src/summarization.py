from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.logger import logger
from src.prompts import SUMMARIZATION_PROMPT
from src.settings import app_settings

# Ограничиваем длину текста для суммаризации, чтобы не переполнить контекст LLM
MAX_SUMMARIZATION_CHARS = 12_000
# Сколько чанков забираем максимум на один документ
MAX_SUMMARY_CHUNKS = 50
# Размер страницы при скролле коллекции
SCROLL_PAGE_SIZE = 64


def _prepare_summary_llm() -> ChatOpenAI:
    """Создание LLM для суммаризации источников."""
    return ChatOpenAI(
        model=app_settings.llm.model,
        temperature=0.3,
        max_tokens=min(app_settings.llm.max_tokens, 512),
        base_url=app_settings.llm.base_url,
        api_key=app_settings.llm.api_key.get_secret_value(),
    )


summary_llm = _prepare_summary_llm()


def _fetch_chunks(
    qdrant_client: QdrantClient,
    document_id: int,
    chunk_ids: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """
    Забирает чанки из Qdrant по ID документа (и опционально списку chunk_id).
    Возвращает отсортированный список (chunk_id, content).
    """
    filter_conditions = [
        qdrant_models.FieldCondition(
            key="metadata.document_id",
            match=qdrant_models.MatchValue(value=document_id),
        ),
        # fallback for payloads without nested metadata
        qdrant_models.FieldCondition(
            key="document_id",
            match=qdrant_models.MatchValue(value=document_id),
        ),
    ]

    scroll_filter = qdrant_models.Filter(should=filter_conditions)
    collected: List[Tuple[int, str]] = []
    next_offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=app_settings.qdrant.collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            limit=SCROLL_PAGE_SIZE,
            offset=next_offset,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            chunk_id = payload.get("chunk_id") or meta.get("chunk_id")
            if chunk_ids and chunk_id not in chunk_ids:
                continue

            content = (
                payload.get("page_content")
                or payload.get("content")
                or payload.get("text")
                or meta.get("page_content")
                or meta.get("content")
            )
            if not content:
                continue

            collected.append((int(chunk_id) if chunk_id is not None else 0, content))
            if len(collected) >= MAX_SUMMARY_CHUNKS:
                break

        if next_offset is None or len(collected) >= MAX_SUMMARY_CHUNKS:
            break

    collected.sort(key=lambda item: item[0])
    return collected


async def summarize_document(
    document_id: int,
    qdrant_client: QdrantClient,
    chunk_ids: Optional[List[int]] = None,
) -> str:
    """
    Строит краткую суммаризацию для документа по его ID.

    Args:
        document_id: ID исходного документа (payload.document_id)
        qdrant_client: Клиент Qdrant
        chunk_ids: Опциональный список chunk_id, если нужно ограничить область

    Returns:
        str: Суммаризация статьи или сообщение об ошибке
    """
    if qdrant_client is None:
        logger.error("Qdrant клиент не инициализирован для суммаризации")
        return "Хранилище источников недоступно: Qdrant клиент не инициализирован."

    chunks = _fetch_chunks(qdrant_client, document_id, chunk_ids)
    if not chunks:
        return f"Не удалось найти текст для источника {document_id}."

    joined_text = "\n\n".join(text for _, text in chunks)
    if len(joined_text) > MAX_SUMMARIZATION_CHARS:
        logger.info(
            "Обрезаем текст для суммаризации документа %s: %d символов -> %d",
            document_id,
            len(joined_text),
            MAX_SUMMARIZATION_CHARS,
        )
        joined_text = joined_text[:MAX_SUMMARIZATION_CHARS]

    try:
        response = await summary_llm.ainvoke(
            [
                SystemMessage(content=SUMMARIZATION_PROMPT),
                HumanMessage(content=joined_text),
            ]
        )
        return response.content.strip()
    except Exception as exc:
        logger.error("Ошибка суммаризации документа %s: %s", document_id, exc)
        return "Не удалось построить суммаризацию источника."
