import logging

from settings import app_settings


def setup_logger() -> logging.Logger:
    """
    Инициализирует логгер с логированием только в консоль.
    
    Returns:
        logging.Logger: Настроенный логгер приложения
    """
    logger = logging.getLogger("rag_app")
    logger.setLevel(app_settings.logger.level)
    
    # Удаляем существующие хендлеры
    logger.handlers.clear()
    
    # Форматтер
    formatter = logging.Formatter(app_settings.logger.format)
    
    # Хендлер для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger()
