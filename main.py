import re
import uvicorn
from fastapi import FastAPI
from src.schemas import RAGRequest, RAGResponse, SummarizationRequest, SummarizationResponse
from src.settings import app_settings
from src.caching import get_cached_answer, set_cached_answer, get_cache_count, get_cache_count
from src.logger import logger
from src.lifespan import lifespan, app_state


def parse_sources(text: str) -> dict[int, str]:
    # заглушка для парсинга источников из текста ответа
    # [1] ... - http://...
    pattern = r"\[(\d+)\].*?-\s*(https?://[^\s\"']+)"
    
    matches = re.findall(pattern, text)
    
    # return {int(num): url for num, url in matches}
    return {1:"https://habr.com/ru/companies/ru_mts/articles/965622/",
            2: "https://habr.com/ru/companies/avito/articles/966018/"}


app = FastAPI(
    title="RAG API",
    description="API для получения ответов с использованием Retrieval-Augmented Generation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/answer", response_model=RAGResponse)
async def get_rag_answer(request: RAGRequest) -> RAGResponse:
    """Получить ответ от RAG.
    
    Args:
        request (RAGRequest): Запрос с вопросом пользователя
    
    Returns:
        RAGResponse: Ответ и информация о его источнике (кэш или генерация)
    """
    query = request.query.strip()

    logger.info(f"Новый запрос: {query}")

    
    cached_answer = await get_cached_answer(query)
    if cached_answer:
        logger.info("Ответ найден в кэше")
        return RAGResponse(answer=cached_answer, from_cache=True, links=parse_sources(cached_answer))
    
    logger.info("Ответ не в кэше, генерируем новый ответ")
    try:
        answer = await app_state.rag_graph.run(query)
        await set_cached_answer(query, answer)
        return RAGResponse(answer=answer, from_cache=False, links=parse_sources(answer))
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return RAGResponse(answer="Ошибка при обработке запроса", from_cache=False, links={})


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_article(request: SummarizationRequest) -> SummarizationResponse:
    """Получить суммаризацию статьи по её ID.
    
    Args:
        request (SummarizationRequest): URL статьи
    
    Returns:
        SummarizationResponse: Суммаризация статьи
    """
    logger.info(f"Запрос суммаризации статьи: {request.article_url}")
    
    summary = f"Это заглушка для суммаризации статьи с URL {request.article_url}."
    
    return SummarizationResponse(summary=summary)


@app.get("/health")
async def health_check():
    cache_count = await get_cache_count()
    return {
        "status": "ok",
        "message": "RAG запущен",
        "cached_answers": cache_count
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=app_settings.fastapi.host,
        port=app_settings.fastapi.port,
        reload=app_settings.fastapi.reload,
    )
