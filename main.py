from contextlib import asynccontextmanager
import gc
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import torch
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from qdrant_client import QdrantClient
from mlflow.tracking import MlflowClient

from src.schemas import AppState, RAGRequest, RAGResponse
from src.settings import app_settings
from src.caching import load_answer_cache, save_answer_cache
from src.chunking import chunk_documents
from src.logger import logger
from src.rag.graph import RAGGraph
from src.retrievers import create_ensemble_retriever, create_reranked_retriever

app_state = AppState()


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Проверяет доступность TCP-порта.

    Args:
        host (str): Адрес хоста для проверки.
        port (int): Порт, доступность которого нужно проверить.
        timeout (float): Максимальное время ожидания соединения в секундах.

    Returns:
        bool: True, если порт доступен, иначе False.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_backend_store_location(uri: str) -> None:
    """
    Создаёт директории для sqlite backend, если требуется.

    Args:
        uri (str): URI backend-хранилища MLflow.

    Returns:
        None
    """
    parsed = urlparse(uri)
    if parsed.scheme != "sqlite":
        return
    if uri.startswith("sqlite:////"):
        db_path = Path(parsed.path)
    else:
        db_path = Path(parsed.path.lstrip("/"))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _find_free_port(host: str) -> int:
    """
    Находит свободный локальный порт для указанного хоста.

    Args:
        host (str): Хост, на котором ищем свободный порт.

    Returns:
        int: Номер свободного порта.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _mlflow_endpoint_ready(tracking_uri: str) -> bool:
    """
    Проверяет, отвечает ли указанный MLflow endpoint на API-запросы.

    Args:
        tracking_uri (str): URI трекинг-сервера MLflow.

    Returns:
        bool: True, если сервер отвечает на запросы, иначе False.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        MlflowClient().list_experiments(max_results=1)
        return True
    except Exception as exc:
        logger.debug("MLflow endpoint %s недоступен: %s", tracking_uri, exc)
        return False


def _maybe_start_local_mlflow_server() -> tuple[subprocess.Popen | None, str]:
    """
    Автоматически запускает локальный MLflow сервер при необходимости.

    Returns:
        tuple[subprocess.Popen | None, str]: Процесс MLflow (если был запущен) и актуальный tracking URI.
    """
    settings = app_settings.mlflow
    if not settings.auto_start:
        return None, settings.tracking_uri
    parsed = urlparse(settings.tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return None, settings.tracking_uri
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host not in {"127.0.0.1", "localhost"}:
        return None, settings.tracking_uri

    tracking_uri = settings.tracking_uri
    if _is_port_open(host, port):
        if _mlflow_endpoint_ready(tracking_uri):
            logger.info("MLflow сервер уже доступен по адресу %s:%s", host, port)
            return None, tracking_uri
        logger.warning(
            "Порт %s:%s занят, но MLflow API недоступен. Будет запущен новый сервер на другом порту.",
            host,
            port,
        )
        port = _find_free_port(host)
        tracking_uri = f"{parsed.scheme}://{host}:{port}"

    logger.info("MLflow сервер не найден на %s:%s. Запускаем локальный экземпляр...", host, port)
    artifact_root = Path(app_settings.mlflow.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    _ensure_backend_store_location(app_settings.mlflow.backend_store_uri)

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        app_settings.mlflow.backend_store_uri,
        "--default-artifact-root",
        str(artifact_root),
    ]
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    deadline = time.time() + app_settings.mlflow.startup_timeout_seconds
    while time.time() < deadline:
        if _is_port_open(host, port):
            logger.info("Локальный MLflow сервер запущен (%s:%s)", host, port)
            return process, tracking_uri
        time.sleep(1)

    process.terminate()
    logger.error(
        "Не удалось запустить MLflow сервер за %s секунд",
        app_settings.mlflow.startup_timeout_seconds,
    )
    return None, settings.tracking_uri

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Инициализация приложения при старте и освобождение ресурсов при завершении.
    
    Args:
        app (FastAPI): Инстанс FastAPI приложения
    """
    logger.info("Инициализация приложения RAG...")
    
    mlflow_active = False
    app_state.mlflow_process = None
    active_tracking_uri = app_settings.mlflow.tracking_uri
    # Инициализация MLflow
    if app_settings.mlflow.enabled:
        try:
            mlflow_process, tracking_uri = _maybe_start_local_mlflow_server()
            app_state.mlflow_process = mlflow_process
            logger.info("Подключение к MLflow...")
            if tracking_uri != app_settings.mlflow.tracking_uri:
                app_settings.mlflow.tracking_uri = tracking_uri
            active_tracking_uri = tracking_uri
            mlflow.set_tracking_uri(active_tracking_uri)
            mlflow.set_experiment(app_settings.mlflow.experiment_name)
            mlflow.langchain.autolog()
            mlflow_active = True
            logger.info(f"MLflow настроен: {active_tracking_uri}")
        except Exception as exc:
            logger.warning("Не удалось инициализировать MLflow (%s). Трекинг отключён.", exc)
    
    # Инициализация Qdrant
    logger.info("Инициализация Qdrant клиента...")
    qdrant_settings = app_settings.qdrant
    qdrant_kwargs = {"prefer_grpc": qdrant_settings.prefer_grpc}
    if qdrant_settings.url:
        qdrant_kwargs["url"] = qdrant_settings.url
        if qdrant_settings.api_key and qdrant_settings.api_key.get_secret_value():
            qdrant_kwargs["api_key"] = qdrant_settings.api_key.get_secret_value()
    else:
        qdrant_kwargs["path"] = qdrant_settings.path
    app_state.qdrant_client = QdrantClient(**qdrant_kwargs)
    logger.info("Qdrant клиент готов")
    
    # Загрузка данных и создание ретривера
    logger.info(f"Загрузка датасета {app_settings.dataset}...")
    rag_dataset = load_dataset(app_settings.dataset, split=app_settings.split_dataset)
    documents = rag_dataset["context"]
    logger.info(f"Датасет загружен: {len(documents)} документов")
    
    logger.info("Чанкирование документов...")
    chunked_docs = chunk_documents(documents)
    
    logger.info("Создание ретривера...")
    ensemble_retriever = create_ensemble_retriever(chunked_docs, app_state.qdrant_client)
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
    qdrant_client = app_state.qdrant_client
    mlflow_process = app_state.mlflow_process
    app_state.rag_graph = None
    app_state.retriever = None
    app_state.cache = None
    app_state.qdrant_client = None
    app_state.mlflow_process = None
    
    # Очищаем сборщик мусора
    gc.collect()
    
    # Очищаем GPU память если есть
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU память очищена")
    
    # Закрываем MLflow если он включен
    if mlflow_active:
        mlflow.end_run()
        logger.info("MLflow сессия закрыта")
    
    if qdrant_client:
        qdrant_client.close()
        logger.info("Qdrant клиент закрыт")
    
    if mlflow_process:
        mlflow_process.terminate()
        try:
            mlflow_process.wait(timeout=10)
        except Exception:
            mlflow_process.kill()
        logger.info("Локальный MLflow сервер остановлен")
    
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
