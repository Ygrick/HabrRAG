# 🚀 Quick Start Guide

Быстрый гайд для запуска RAG API.

## 1️⃣ Получить API ключ

1. Перейти на https://openrouter.ai/
2. Зарегистрироваться
3. Получить API ключ в разделе "API Keys"

## 2️⃣ Создать `.env` файл

Создайте файл `.env` в корне проекта с минимальной конфигурацией:

```env
RAG_APP__LLM__API_KEY=sk-or-v1-... # Ваш API ключ из OpenRouter
```

Остальные параметры возьмутся из defaults. Полный список доступен в `.env.example`.

## 3️⃣ Установить зависимости

### Вариант 1: Через UV (рекомендуется)
```bash
uv sync
```

### Вариант 2: Через PIP
```bash
python -m venv .venv
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4️⃣ Запустить приложение

```bash
python main.py
```

Приложение будет доступно на: http://localhost:8000

## 5️⃣ Использовать API

### Health Check
```bash
curl http://localhost:8000/health
```

### Задать вопрос
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### Использовать пример
```bash
python examples_api.py
```

## 📚 Документация

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🐳 Docker

```bash
# Сборка
docker build -t habrrag .

# Запуск (с GPU поддержкой)
docker run --gpus all -p 8000:8000 \
  -e RAG_APP__LLM__API_KEY=sk-or-v1-... \
  habrrag
```

## ⚙️ Основные параметры конфигурации

```env
# Модель LLM
RAG_APP__LLM__MODEL=qwen/qwen-2.5-7b-instruct:free
RAG_APP__LLM__TEMPERATURE=0.6
RAG_APP__LLM__MAX_TOKENS=1000

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

## 🆘 Решение проблем

### Ошибка: "Import langchain_openai could not be resolved"
- Это просто warning от линтера VS Code
- Пакеты установятся после `uv sync` или `pip install -r requirements.txt`

### Ошибка подключения при запросе
- Убедитесь что сервер запущен: `python main.py`
- Проверьте что используется правильный адрес: `http://localhost:8000`

### Ошибка с API ключом
- Проверьте что `.env` файл создан
- Убедитесь что в переменной `RAG_APP__LLM__API_KEY` правильный ключ
- Проверьте что ключ активен в OpenRouter

## 📋 Структура проекта

```
HabrRAG/
├── main.py                 # FastAPI приложение
├── settings.py             # Конфигурация
├── src/
│   ├── chunking.py         # Чанкирование текста
│   ├── retrievers.py       # Создание ретриверов
│   ├── rag_pipeline.py     # RAG логика
│   ├── caching.py          # Асинхронное кэширование
│   ├── logger.py           # Логирование
│   └── prompts.py          # Промпты LLM
├── examples_api.py         # Примеры использования
├── .env.example            # Пример конфигурации
└── README.md              # Полная документация
```

## 🎯 Что дальше?

1. Изучите документацию API: http://localhost:8000/docs
2. Попробуйте примеры в `examples_api.py`
3. Измените параметры в `.env` для экспериментов
4. Читайте логи в `./logs/app.log` для отладки

---

**Happy RAG-ging! 🎉**
