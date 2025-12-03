from contextlib import asynccontextmanager
import gc

import mlflow
import torch
from fastapi import FastAPI
from qdrant_client import QdrantClient

from src.schemas import AppState
from src.settings import app_settings
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
    if app_settings.qdrant.url:
        qdrant_kwargs = {
            "prefer_grpc": app_settings.qdrant.prefer_grpc,
            "url": app_settings.qdrant.url,
            "api_key": app_settings.qdrant.api_key.get_secret_value() if app_settings.qdrant.api_key else None,
        }
    else:
        qdrant_kwargs = {
            "prefer_grpc": app_settings.qdrant.prefer_grpc,
            "path": app_settings.qdrant.path,
        }
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
            "коллекцию. Создание ретривера на основе существующей коллекции..."
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
    app_state.rag_graph = RAGGraph(retriever=app_state.retriever)
    logger.info("RAG граф инициализирован")
    logger.info("Приложение инициализировано успешно")
    
    yield
    
    logger.info("Завершение работы приложения...")
    logger.info("Освобождение ресурсов...")
    app_state.rag_graph = None
    app_state.retriever = None
    qdrant_client = app_state.qdrant_client
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
