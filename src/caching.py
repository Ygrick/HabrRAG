import asyncpg

from src.logger import logger
from src.settings import app_settings


async def get_db_connection() -> asyncpg.Connection:
    """
    Создает и возвращает соединение с базой данных.
    
    Returns:
        asyncpg.Connection: Соединение с БД
    """
    return await asyncpg.connect(
        host=app_settings.database.host,
        port=app_settings.database.port,
        user=app_settings.database.user,
        password=app_settings.database.password.get_secret_value(),
        database=app_settings.database.database,
    )


async def load_answer_cache() -> dict:
    """
    Загружает весь кэш ответов из базы данных.
    
    Returns:
        dict: Словарь кэша, где ключ — запрос, значение — ответ
    """
    conn = await get_db_connection()
    try:
        rows = await conn.fetch(f"SELECT query, response FROM {app_settings.database.cache_table}")
        cache = {row['query']: row['response'] for row in rows}
        return cache
    except Exception as e:
        logger.error(f"Ошибка загрузки кэша ответов: {e}")
        return {}
    finally:
        await conn.close()


async def save_answer_cache(cache: dict) -> None:
    """
    Сохраняет весь кэш ответов в базу данных.
    
    Args:
        cache (dict): Словарь кэша
    """
    conn = await get_db_connection()
    try:
        # Очистить таблицу и вставить новые данные
        await conn.execute(f"DELETE FROM {app_settings.database.cache_table}")
        if cache:
            values = [(query, response) for query, response in cache.items()]
            await conn.executemany(f"INSERT INTO {app_settings.database.cache_table} (query, response) VALUES ($1, $2)", values)
        logger.info("Кэш ответов сохранен в базу данных.")
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша ответов: {e}")
    finally:
        await conn.close()


async def get_cached_answer(query: str) -> str | None:
    """
    Получает кэшированный ответ для запроса из базы данных.
    
    Args:
        query (str): Запрос
    
    Returns:
        str | None: Кэшированный ответ или None, если не найден
    """
    conn = await get_db_connection()
    try:
        row = await conn.fetchrow(f"SELECT response FROM {app_settings.database.cache_table} WHERE query = $1", query)
        return row['response'] if row else None
    except Exception as e:
        logger.error(f"Ошибка получения кэшированного ответа: {e}")
        return None
    finally:
        await conn.close()


async def set_cached_answer(query: str, response: str) -> None:
    """
    Сохраняет ответ в кэш в базе данных.
    
    Args:
        query (str): Запрос
        response (str): Ответ
    """
    conn = await get_db_connection()
    try:
        await conn.execute(f"INSERT INTO {app_settings.database.cache_table} (query, response) VALUES ($1, $2) ON CONFLICT (query) DO UPDATE SET response = EXCLUDED.response, created_at = CURRENT_TIMESTAMP", query, response)
        logger.info(f"Ответ для запроса '{query}' сохранен в кэш.")
    except Exception as e:
        logger.error(f"Ошибка сохранения ответа в кэш: {e}")
    finally:
        await conn.close()


async def get_cache_count() -> int:
    """
    Получает количество записей в кэше.
    
    Returns:
        int: Количество записей
    """
    conn = await get_db_connection()
    try:
        row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {app_settings.database.cache_table}")
        return row['count'] if row else 0
    except Exception as e:
        logger.error(f"Ошибка получения количества записей в кэше: {e}")
        return 0
    finally:
        await conn.close()