from contextlib import asynccontextmanager
import gc

import mlflow
import torch
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI

from src.schemas import AppState, RAGRequest, RAGResponse
from src.settings import app_settings
from src.caching import load_answer_cache, save_answer_cache
from src.chunking import chunk_documents
from src.logger import logger
from src.rag.graph import RAGGraph
from src.retrievers import create_ensemble_retriever, create_reranked_retriever

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Инициализация приложения при старте и освобождение ресурсов при завершении.
    
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
    
    # Загрузка данных и создание ретривера
    logger.info(f"Загрузка датасета {app_settings.dataset}...")
    rag_dataset = load_dataset(app_settings.dataset, split=app_settings.split_dataset)
    documents = rag_dataset["context"]
    logger.info(f"Датасет загружен: {len(documents)} документов")
    
    logger.info("Чанкирование документов...")
    chunked_docs = chunk_documents(documents)
    
    logger.info("Создание ретривера...")
    ensemble_retriever = create_ensemble_retriever(chunked_docs)
    app_state.retriever = create_reranked_retriever(ensemble_retriever)
    logger.info("Ретривер создан и готов к использованию")
    
    # Инициализация RAG графа
    logger.info("Инициализация RAG графа...")
    rag_graph = RAGGraph(retriever=app_state.retriever)
    app_state.rag_graph = rag_graph
    logger.info("RAG граф инициализирован")
    
    # Загрузка кэша ответов
    logger.info("Загрузка кэша ответов...")
    app_state.cache = await load_answer_cache()
    logger.info(f"Кэш загружен: {len(app_state.cache)} ответов")
    
    logger.info("Приложение инициализировано успешно")
    
    yield
    
    logger.info("Завершение работы приложения...")
    
    # Сохранение кэша перед завершением
    if app_state.cache:
        await save_answer_cache(app_state.cache)
        logger.info("Кэш сохранён")
    
    # Очистка ресурсов
    logger.info("Освобождение ресурсов...")
    
    # Удаляем ссылки на большие объекты
    app_state.rag_graph = None
    app_state.retriever = None
    app_state.cache = None
    
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
    
    logger.info("Приложение завершено")


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
    logger.info(f"Новый запрос: {request.query}")
    
    # Проверка кэша
    if request.query in app_state.cache:
        logger.info("Ответ найден в кэше")
        return RAGResponse(answer=app_state.cache[request.query], from_cache=True)
    
    logger.info("Ответ не в кэше, генерируем новый ответ")
    try:
        answer = await app_state.rag_graph.run(request.query) # RAG
        app_state.cache[request.query] = answer # Update cache
        await save_answer_cache(app_state.cache) # Save cache
        return RAGResponse(answer=answer, from_cache=False)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return RAGResponse(answer="Ошибка при обработке запроса", from_cache=False)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "RAG запущен",
        "cached_answers": len(app_state.cache) if app_state.cache else 0
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=app_settings.fastapi.host,
        port=app_settings.fastapi.port,
        reload=app_settings.fastapi.reload,
    )
