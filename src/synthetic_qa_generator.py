import asyncio
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.settings import app_settings
from src.logger import logger

PROMPT_QUESTION = (
    "Сгенерируй один информативный вопрос, на который можно ответить, используя только этот текст. "
    "Текст:\n{chunk_content}\n\nВопрос:"
)
PROMPT_ANSWER = (
    "Ответь на вопрос, используя только информацию из текста. "
    "Текст:\n{chunk_content}\n\nВопрос: {question}\n\nОтвет:"
)

async def generate_question(llm, chunk_content):
    prompt = PROMPT_QUESTION.format(chunk_content=chunk_content)
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    return response.content.strip()

async def generate_answer(llm, chunk_content, question):
    prompt = PROMPT_ANSWER.format(chunk_content=chunk_content, question=question)
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    return response.content.strip()

async def process_chunk(llm, chunk):
    chunk_id = chunk.payload.get("global_chunk_id", chunk.id)
    chunk_content = chunk.payload.get("text_markdown", chunk.payload.get("page_content", ""))
    question = await generate_question(llm, chunk_content)
    answer = await generate_answer(llm, chunk_content, question)
    return {"chunk_id": chunk_id, "question": question, "answer": answer}

async def main(output_path="synthetic_qa.jsonl", batch_size=20, limit=None):
    # Настройка LLM
    llm = ChatOpenAI(
        model=app_settings.llm.model,
        temperature=0.6,
        max_tokens=256,
        base_url=app_settings.llm.base_url,
        api_key=app_settings.llm.api_key.get_secret_value(),
    )
    # Настройка Qdrant
    qdrant_settings = app_settings.qdrant
    qdrant_kwargs = {"prefer_grpc": qdrant_settings.prefer_grpc}
    if qdrant_settings.url:
        qdrant_kwargs["url"] = qdrant_settings.url
        api_key = qdrant_settings.api_key
        if api_key and api_key.get_secret_value():
            qdrant_kwargs["api_key"] = api_key.get_secret_value()
    else:
        qdrant_kwargs["path"] = qdrant_settings.path
    client = QdrantClient(**qdrant_kwargs)
    collection_name = qdrant_settings.collection_name

    # Получаем все чанки из Qdrant
    scroll_filter = None
    offset = None
    total = 0
    processed = 0
    with open(output_path, "w", encoding="utf-8") as f:
        while True:
            response = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=batch_size,
                with_payload=True,
            )
            points = response[0]
            if not points:
                break
            # Ограничиваем количество чанков для теста
            if limit:
                points = points[:max(0, limit - processed)]
            tasks = [process_chunk(llm, chunk) for chunk in points]
            results = await asyncio.gather(*tasks)
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Достигнут лимит: {limit}")
                    return
            offset = response[1]
            total += len(points)
            logger.info(f"Обработано чанков: {total}")
            if limit and processed >= limit:
                break
    logger.info(f"Генерация завершена. Всего записей: {processed}")

if __name__ == "__main__":
    asyncio.run(main())
