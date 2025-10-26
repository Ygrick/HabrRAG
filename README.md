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
├── main.py                    # FastAPI приложение
├── schemas.py                 # Pydantic модели (AppState, RAGRequest, RAGResponse)
├── settings.py                # Конфигурация из переменных окружения
├── src/
│   ├── models.py             # Pydantic модель Document с методом __str__
│   ├── logger.py             # Логирование в консоль
│   ├── caching.py            # Кэширование ответов (pathlib)
│   ├── rag_graph.py          # LangGraph RAG пайплайн (3 узла)
│   ├── chunking.py           # Чанкирование документов
│   ├── retrievers.py         # Создание FAISS + BM25 ретривера
│   └── prompts.py            # Системные промпты для LLM
├── Dockerfile                # Docker конфиг
├── pyproject.toml            # Зависимости проекта
├── QUICKSTART.md             # Быстрый старт
└── CHANGES.md                # История изменений
```

## 🔗 LangGraph RAG Граф

Пайплайн состоит из 3 узлов:

```
START
  ↓
retrieve_docs        # Поиск релевантных документов (FAISS + BM25 + Cross-Encoder)
  ↓
identify_docs        # Идентификация ID документов (LLM с низкой температурой)
  ↓
generate_answer      # Генерация финального ответа (LLM с базовой температурой)
  ↓
END
```

**Файл:** `src/rag_graph.py`

### RAGState (Pydantic модель)

```python
class RAGState(BaseModel):
    query: str                      # Вопрос пользователя
    documents: List[Document]       # Список найденных документов (Pydantic)
    documents_text: str            # Текстовое представление документов
    doc_ids: str                   # Идентификаторы релевантных документов
    answer: str                    # Финальный ответ
```

### Document (Pydantic модель)

```python
class Document(BaseModel):
    document_id: int        # ID документа
    chunk_id: int          # ID чанка
    content: str           # Содержимое

    def __str__(self) -> str:
        return f"[DOC:{self.document_id}|CHUNK:{self.chunk_id}]\n{self.content}"
```

**Файл:** `src/models.py`

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
uv sync
# или
pip install -r requirements.txt
```

### 2. Конфигурация

Создайте `.env` файл:

```env
RAG_APP__LLM__API_KEY=sk-or-v1-...
```

### 3. Запуск

```bash
python main.py
```

API будет доступен на `http://localhost:8000`

## 📡 API Эндпоинты

### GET `/health`

Проверка статуса приложения

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "ok",
  "message": "RAG API работает корректно",
  "cached_answers": 42
}
```

### POST `/answer`

Получить ответ от RAG системы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

**Запрос:**
```json
{
  "query": "Your question here"
}
```

**Ответ:**
```json
{
  "answer": "Generated answer...",
  "from_cache": false
}
```

## 📚 API Документация

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🛠️ Используемые технологии

| Технология | Версия | Назначение |
|-----------|--------|-----------|
| **FastAPI** | 0.118.3 | REST API фреймворк |
| **LangGraph** | 0.6.11 | Граф-ориентированный пайплайн |
| **LangChain** | 0.3.27 | Работа с LLM и документами |
| **Pydantic** | 2.10+ | Валидация данных |
| **FAISS** | 1.12.0 | Векторный поиск |
| **ChatOpenAI** | 0.3.35 | Интеграция с OpenAI API |
| **PyTorch** | 2.5.1 | Работа с эмбедингами |

## 🔄 Асинхронная архитектура

- ✅ Все IO операции асинхронные (ainvoke, async/await)
- ✅ Кэширование с pathlib (синхронное, но быстрое)
- ✅ Полная поддержка async в FastAPI эндпоинтах
- ✅ LangGraph граф с async методами

## 📝 Примеры использования

### Python

```python
import asyncio
from src.rag_graph import RAGGraph

async def main():
    rag = RAGGraph(retriever=retriever)
    answer = await rag.run("What is AI?")
    print(answer)

asyncio.run(main())
```

### curl

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning"}'
```

### Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/answer",
    json={"query": "What is deep learning?"}
)
print(response.json())
```

## 🐳 Docker

```bash
# Сборка
docker build -t habrrag .

# Запуск
docker run --gpus all -p 8000:8000 \
  -e RAG_APP__LLM__API_KEY=sk-or-v1-... \
  habrrag
```

## 📊 Конфигурация

Все параметры управляются через переменные окружения с префиксом `RAG_APP__`:

```env
# LLM
RAG_APP__LLM__MODEL=qwen/qwen-2.5-7b-instruct:free
RAG_APP__LLM__TEMPERATURE=0.6
RAG_APP__LLM__MAX_TOKENS=1000
RAG_APP__LLM__API_KEY=sk-or-v1-...

# Retrieval
RAG_APP__RETRIEVAL__TOP_K=3
RAG_APP__RETRIEVAL__ENSEMBLE_WEIGHTS_BM25=0.4
RAG_APP__RETRIEVAL__ENSEMBLE_WEIGHTS_FAISS=0.6

# FastAPI
RAG_APP__FASTAPI__HOST=0.0.0.0
RAG_APP__FASTAPI__PORT=8000

# MLflow (опционально)
RAG_APP__MLFLOW__ENABLED=false
```

## 📖 Документы

- **QUICKSTART.md** - быстрый старт в 5 минут
- **CHANGES.md** - история изменений версий

## 📩 Контакты

Вопросы и предложения: 👨‍💻 **https://t.me/Ygrickkk**

