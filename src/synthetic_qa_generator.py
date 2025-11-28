import asyncio
import json
from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI
from qdrant_client import QdrantClient

from src.settings import app_settings
from src.logger import logger

class ChunkQA(BaseModel):
    question: str = Field(description="Информативный вопрос только из данного текста")
    answer: str = Field(description="Точный ответ, основанный исключительно на тексте. 'Недостаточно информации' если факт недоступен")

PROMPT_QA = (
    "Ты помощник по генерации обучающих QA пар. Дай ровно один информативный вопрос и его точный ответ, используя только текст. "
    "Если нельзя ответить строго по тексту, ответ должен быть 'Недостаточно информации'. Верни только структуру без пояснений."
    "\nТекст:\n{chunk_content}"
)

def generate_qa(mistral_llm: ChatMistralAI, chunk_content: str) -> dict:
    prompt = PROMPT_QA.format(chunk_content=chunk_content)
    structured_llm = mistral_llm.with_structured_output(ChunkQA)
    try:
        qa_obj: ChunkQA = structured_llm.invoke(prompt)
        return {"question": qa_obj.question.strip(), "answer": qa_obj.answer.strip()}
    except Exception as e:
        logger.error(f"Ошибка structured output вызова Mistral: {e}")
        return {"question": "", "answer": ""}

def process_chunk(mistral_llm: ChatMistralAI, chunk):
    chunk_id = chunk.payload.get("metadata").get("global_chunk_id")
    chunk_content = chunk.payload.get("page_content", "")
    qa = generate_qa(mistral_llm, chunk_content)
    return {"chunk_id": chunk_id, "question": qa.get("question", ""), "answer": qa.get("answer", "")}

def main(output_path="synthetic_qa.jsonl", batch_size=20, limit=None):
    # Инициализация LangChain Mistral LLM
    api_key = app_settings.llm.api_key.get_secret_value()
    if not api_key:
        raise ValueError("Не задан API ключ для Mistral (RAG_APP__LLM__API_KEY)")
    mistral_llm = ChatMistralAI(
        model="mistral-small-2501",  # например 'mistral-large-latest'
        api_key=api_key,
        temperature=app_settings.llm.temperature,
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
    qdrant_client = QdrantClient(**qdrant_kwargs)
    collection_name = qdrant_settings.collection_name

    # Получаем все чанки из Qdrant
    scroll_filter = None
    offset = None
    total = 0
    processed = 0
    with open(output_path, "w", encoding="utf-8") as f:
        while True:
            response = qdrant_client.scroll(
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
            tasks = [process_chunk(mistral_llm, chunk) for chunk in points]
            for item in tasks:
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
    main()
