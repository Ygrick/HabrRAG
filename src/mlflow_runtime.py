from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

from src.logger import logger
from src.settings import MLflowSettings


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Проверяет, доступен ли указанный TCP-порт.

    Args:
        host (str): Адрес хоста.
        port (int): Порт для проверки.
        timeout (float): Таймаут подключения.

    Returns:
        bool: True, если порт открыт.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_backend_store_location(uri: str) -> None:
    """
    Создаёт директорию для SQLite backend, если это необходимо.

    Args:
        uri (str): URI backend-хранилища MLflow.
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
    Находит свободный порт на указанном хосте.

    Args:
        host (str): Хост для поиска порта.

    Returns:
        int: Свободный порт.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _mlflow_endpoint_ready(tracking_uri: str) -> bool:
    """
    Проверяет, отвечает ли MLflow endpoint на API-запросы.

    Args:
        tracking_uri (str): URI трекинг-сервера MLflow.

    Returns:
        bool: True, если сервер доступен.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        MlflowClient().list_experiments(max_results=1)
        return True
    except Exception as exc:
        logger.debug("MLflow endpoint %s недоступен: %s", tracking_uri, exc)
        return False


def ensure_local_mlflow_server(settings: MLflowSettings) -> tuple[subprocess.Popen | None, str]:
    """
    При необходимости запускает локальный MLflow сервер и возвращает его URI.

    Args:
        settings (MLflowSettings): Настройки MLflow.

    Returns:
        tuple[subprocess.Popen | None, str]: Процесс MLflow (если был запущен) и актуальный tracking URI.
    """
    if not settings.auto_start:
        return None, settings.tracking_uri

    parsed = urlparse(settings.tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return None, settings.tracking_uri

    host = parsed.hostname
    if not host or host not in {"127.0.0.1", "localhost"}:
        return None, settings.tracking_uri

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    tracking_uri = settings.tracking_uri

    if _is_port_open(host, port):
        if _mlflow_endpoint_ready(tracking_uri):
            logger.info("MLflow сервер уже доступен по адресу %s:%s", host, port)
            return None, tracking_uri
        logger.warning(
            "Порт %s:%s занят, но MLflow API недоступен. Новый сервер будет запущен на другом порту.",
            host,
            port,
        )
        port = _find_free_port(host)
        tracking_uri = f"{parsed.scheme}://{host}:{port}"

    logger.info("MLflow сервер не найден на %s:%s. Запускаем локальный экземпляр...", host, port)
    artifact_root = Path(settings.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    _ensure_backend_store_location(settings.backend_store_uri)

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
        settings.backend_store_uri,
        "--default-artifact-root",
        str(artifact_root),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )

    deadline = time.time() + settings.startup_timeout_seconds
    while time.time() < deadline:
        if _is_port_open(host, port):
            logger.info("Локальный MLflow сервер запущен (%s:%s)", host, port)
            return process, tracking_uri
        time.sleep(1)

    process.terminate()
    logger.error(
        "Не удалось запустить MLflow сервер за %s секунд",
        settings.startup_timeout_seconds,
    )
    return None, settings.tracking_uri
