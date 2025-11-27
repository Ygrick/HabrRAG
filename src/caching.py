import json
from pathlib import Path

from src.logger import logger
from src.settings import app_settings


async def load_answer_cache() -> dict:
    """
    Загружает кэш ответов из файла.
    
    Returns:
        dict: Словарь кэша, где ключ — запрос, значение — ответ
    """
    cache_file = Path(app_settings.cache_path)
    
    if cache_file.exists():
        try:
            content = cache_file.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша ответов: {e}")
    return {}


async def save_answer_cache(cache: dict) -> None:
    """
    Сохраняет кэш ответов в файл.
    
    Args:
        cache (dict): Словарь кэша
    """
    cache_file = Path(app_settings.cache_path)
    
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(cache, ensure_ascii=False, indent=4)
        cache_file.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша ответов: {e}")
