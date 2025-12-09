import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from src.schemas import RAGRequest, RAGResponse, SummarizationRequest, SummarizationResponse
from src.settings import app_settings
from src.caching import get_cached_answer, set_cached_answer, get_cache_count
from src.logger import logger
from src.lifespan import lifespan, app_state
from src.build_sources import build_sources
from src.rag.schemas import Document
from contextlib import nullcontext
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
    logger.info(f"Новый запрос: {query}, run_id: {request.run_id}")
    
    cached_entry = await get_cached_answer(query)
    # if cached_entry:
    if False:
        logger.info("Ответ найден в кэше")
        documents = [Document(**doc) for doc in cached_entry["documents"]]
        return RAGResponse(
            answer=cached_entry["answer"],
            from_cache=True,
            sources=build_sources(documents)
        )
    logger.info("Ответ не в кэше, генерируем новый ответ")

    # Используем контекстный менеджер mlflow если передан run_id
    use_mlflow = request.run_id and app_settings.mlflow.enabled
    ctx = mlflow.start_run(run_id=request.run_id) if use_mlflow else nullcontext()

    try:
        with ctx:
            rag_state = await app_state.rag_graph.run(query)
            sources = build_sources(rag_state.documents)
            await set_cached_answer(query, rag_state)
            return RAGResponse(answer=rag_state.answer, from_cache=False, sources=sources)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса (run_id={request.run_id}): {e}")
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
