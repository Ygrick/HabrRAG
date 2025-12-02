import uvicorn
from fastapi import FastAPI, HTTPException
from src.schemas import RAGRequest, RAGResponse, SummarizationRequest, SummarizationResponse
from src.settings import app_settings
from src.caching import get_cached_answer, set_cached_answer, get_cache_count, get_cache_count
from src.logger import logger
from src.lifespan import lifespan, app_state
from src.build_sources import build_sources
from src.rag.schemas import Document
from src.summarization import summarize_document

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
    
    cached_entry = await get_cached_answer(query)
    if cached_entry:
        logger.info("Ответ найден в кэше")
        
        documents = [Document(**doc) for doc in cached_entry.get("documents", [])]
        cached_sources = build_sources(documents)

        print (cached_sources)
        cached_answer = cached_entry.get("answer", "")

        return RAGResponse(answer=cached_answer, from_cache=True, sources=cached_sources)
    
    logger.info("Ответ не в кэше, генерируем новый ответ")
    try:
        rag_state = await app_state.rag_graph.run(query)

        sources = build_sources(rag_state.documents)

        await set_cached_answer(query, rag_state)

        return RAGResponse(answer=rag_state.answer, from_cache=False, sources=sources)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return RAGResponse(
            answer="Ошибка при обработке запроса", from_cache=False, sources=[]
        )


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_source(request: SummarizationRequest) -> SummarizationResponse:
    """Получить суммаризацию статьи/документа по его ID."""
    logger.info(
        "Запрос суммаризации: document_id=%s, chunk_ids=%s",
        request.document_id,
        request.chunk_ids,
    )

    if not app_state.qdrant_client:
        raise HTTPException(
            status_code=503, detail="Qdrant клиент не инициализирован."
        )

    summary = await summarize_document(
        document_id=request.document_id,
        qdrant_client=app_state.qdrant_client,
        chunk_ids=request.chunk_ids,
    )

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
