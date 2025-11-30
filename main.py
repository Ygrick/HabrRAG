from contextlib import asynccontextmanager
import gc
import re

import mlflow
import torch
import uvicorn
from fastapi import FastAPI
from qdrant_client import QdrantClient

from src.schemas import AppState, RAGRequest, RAGResponse, SummarizationRequest, SummarizationResponse
from src.settings import app_settings
from src.caching import get_cached_answer, set_cached_answer, get_cache_count, get_cache_count
from src.logger import logger
from src.rag.graph import RAGGraph
from src.retrievers import (
    collection_exists_and_not_empty,
    create_qdrant_only_retriever,
    create_reranked_retriever,
)

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация приложения при старте и освобождение ресурсов при завершении.
    
    Args:
        app (FastAPI): Инстанс FastAPI приложения
    """
    logger.info("Инициализация приложения RAG...")
    
    # Инициализация MLflow
    if app_settings.mlflow.enabled:
        logger.info("Подключение к MLflow...")
        mlflow.set_tracking_uri(app_settings.mlflow.tracking_uri)
        mlflow.set_experiment(app_settings.mlflow.experiment_name)
        mlflow.langchain.autolog()
        logger.info(f"MLflow настроен: {app_settings.mlflow.tracking_uri}")
    
    # Инициализация Qdrant
    logger.info("Инициализация Qdrant клиента...")
    qdrant_settings = app_settings.qdrant
    qdrant_kwargs = {"prefer_grpc": qdrant_settings.prefer_grpc}
    if qdrant_settings.url:
        qdrant_kwargs["url"] = qdrant_settings.url
        api_key = qdrant_settings.api_key
        if api_key and api_key.get_secret_value():
            qdrant_kwargs["api_key"] = api_key.get_secret_value()
    else:
        qdrant_kwargs["path"] = qdrant_settings.path
    app_state.qdrant_client = QdrantClient(**qdrant_kwargs)
    logger.info("Qdrant клиент готов")

    # Проверяем существование коллекции
    collection_name = app_settings.qdrant.collection_name
    collection_ready = collection_exists_and_not_empty(
        app_state.qdrant_client, collection_name
    )

    if collection_ready:
        logger.info(
            f"Коллекция '{collection_name}' уже существует и содержит данные. "
            "Пропускаем загрузку документов и используем существующую "
            "коллекцию."
            "Создание ретривера на основе существующей коллекции..."
        )
        app_state.retriever = create_qdrant_only_retriever(
            app_state.qdrant_client
        )
    else:
        logger.info(
            f"Коллекция '{collection_name}' не существует или пуста. "
            "Загружаем и индексируем документы..."
        )
        app_state.retriever = create_reranked_retriever(app_state.qdrant_client)
    logger.info("Ретривер создан и готов к использованию")
    
    # Инициализация RAG графа
    logger.info("Инициализация RAG графа...")
    rag_graph = RAGGraph(retriever=app_state.retriever)
    app_state.rag_graph = rag_graph
    logger.info("RAG граф инициализирован")
    logger.info("Приложение инициализировано успешно")
    
    yield
    
    logger.info("Завершение работы приложения...")
    logger.info("Освобождение ресурсов...")
    # Удаляем ссылки на большие объекты
    qdrant_client = app_state.qdrant_client
    app_state.rag_graph = None
    app_state.retriever = None
    app_state.qdrant_client = None
    
    # Очищаем сборщик мусора
    gc.collect()
    
    # Очищаем GPU память если есть
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU память очищена")
    
    # Закрываем MLflow если он включен
    if app_settings.mlflow.enabled:
        mlflow.end_run()
        logger.info("MLflow сессия закрыта")
    if qdrant_client:
        qdrant_client.close()
        logger.info("Qdrant клиент закрыт")
    
    logger.info("Приложение завершено")

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
