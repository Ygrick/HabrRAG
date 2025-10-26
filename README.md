# 🔍 HabrRAG: Retrieval-Augmented Generation API

HabrRAG — это асинхронный REST API для Retrieval-Augmented Generation (RAG), использующий LangGraph для управления пайплайном.

## 🚀 Основные возможности

- ✅ **FastAPI** - асинхронный веб-фреймворк
- ✅ **LangGraph** - граф-ориентированный RAG пайплайн
- ✅ **Pydantic** - типизация и валидация данных
- ✅ **FAISS + BM25** - гибридный поиск документов
- ✅ **Cross-Encoder** - переранжирование результатов
- ✅ **ChatOpenAI** - интеграция с LLM
- ✅ **Асинхронность** - полная поддержка async/await
- ✅ **Кэширование** - сохранение ответов с pathlib

## 📋 Структура проекта

```
HabrRAG/
├── main.py                   # FastAPI приложение
├── src/
│   ├── logger.py             # Логирование в консоль
│   ├── schemas.py            # Pydantic модели (AppState, RAGRequest, RAGResponse)
│   ├── settings.py           # Конфигурация из переменных окружения
│   ├── caching.py            # Кэширование ответов (pathlib)
│   ├── rag/
│   │   ├── graph.py          # LangGraph RAG пайплайн
│   │   └── state.py          # Pydantic модели RAGState и Document
│   ├── chunking.py           # Чанкирование документов
│   ├── retrievers.py         # Создание FAISS + BM25 ретривера
│   └── prompts.py            # Системные промпты для LLM
├── Dockerfile                # Docker конфиг
├── pyproject.toml            # Зависимости проекта
└── README.md                 # Документация
```

## 🚀 Быстрый старт

### 1. Получение API ключа

- Перейти на https://openrouter.ai/
- Зарегистрироваться и получить API ключ
- Создать `.env` файл на основе `.env.example`

### 2. Запуск MLflow сервера

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 3. Установка и запуск

```bash
# Установить зависимости
uv sync

# Запуск приложения
uv run main.py
```

Приложение будет доступно на `http://localhost:8000`

## 📡 API Эндпоинты

### GET `/health`

Проверка статуса приложения

```bash
curl http://localhost:8000/health
```

### POST `/answer`

Получить ответ от RAG системы. Запросы можно отправлять через Swagger UI по адресу `http://localhost:8000/docs`

**Примеры запросов:**

```json
{ "query": "Who was the original owner of the lot of items being sold?" }
```

```json
{ "query": "What are some of the skills taught in the Trail Patrol Training course?" }
```

```json
{ "query": "Who were the two convicted killers that escaped from an upstate New York maximum-security prison?" }
```

## 🛠️ Используемые технологии

| Технология | Назначение |
|-----------|-----------|
| **FastAPI** | REST API фреймворк |
| **LangGraph** | Граф-ориентированный пайплайн |
| **LangChain** | Работа с LLM и документами |
| **Pydantic** | Валидация данных |
| **FAISS** | Векторный поиск |
| **ChatOpenAI** | Интеграция с OpenAI API |
| **PyTorch** | Работа с эмбедингами |

