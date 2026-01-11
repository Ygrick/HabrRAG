# Оценка RAG-системы (Evaluation)

Папка `evaluation` содержит инструменты для комплексной оценки качества Retrieval-Augmented Generation (RAG) системы на основе данных из Habr статей. Процесс оценки включает три основных этапа:

1. **Создание датасета в Langfuse** - подготовка тестовых данных с вопросами и ожидаемыми ответами
2. **Запуск RAG цепочки** - выполнение системы на тестовом датасете с логированием в Langfuse
3. **Оценка результатов** - автоматическая оценка качества retrieval и generation с использованием метрик

## Структура папки

- `create_langfuse_dataset.py` - скрипт для создания датасета в Langfuse
- `run_dataset.py` - скрипт для запуска RAG цепочки на датасете
- `evaluate_dataset_run.py` - скрипт для оценки результатов запуска
- `metrics.py` - модуль с функциями расчета метрик оценки
- `structures.py` - определения структур данных для оценки
- `data/` - директория с тестовыми датасетами (JSONL файлы с вопросами и ответами)
- `logs/` - директория для сохранения отчетов об оценке

## Детальное описание процесса

### 1. Создание датасета в Langfuse

**Файл:** `create_langfuse_dataset.py`

Этот скрипт берет JSONL файл с тестовыми данными (вопросы, ожидаемые ответы, метаданные) и загружает их в Langfuse как датасет для тестирования. Каждый элемент датасета содержит:
- Вопрос пользователя
- Ожидаемый ответ (ground truth)
- Метаданные (источники, контекст и т.д.)

#### Запуск 
```bash
python evaluation/create_langfuse_dataset.py \
--file-path <путь_к_файлу с данными> \
--dataset-name <имя_датасета> \ 
[--description <описание>]
```

#### Пример запуска

```bash
python evaluation/create_langfuse_dataset.py \
--file-path evaluation/data/qa_dataset.jsonl \
--dataset-name habr-rag-dataset
```


### 2. Запуск RAG цепочки

**Файл:** `run_dataset.py`

Скрипт выполняет RAG систему на всем датасете Langfuse, генерируя ответы и логируя все шаги в Langfuse для последующего анализа.

#### Запуск 

```bash
python evaluation/run_dataset.py \
[--dataset-name <имя_датасета>] \
[--run-name <имя_запуска>] \
[--run-description <описание_запуска>] \
[--limit <число>]
```

#### Пример запуска

```bash
python evaluation/run_dataset.py \
--dataset-name habr-rag-dataset \
--run-name dataset-run-01 \
--run-description "Тестовый запуск RAG цепочки на датасете" \
--limit 2
```


### 3. Оценка результатов

**Файл:** `evaluate_dataset_run.py`

После выполнения запуска RAG, этот скрипт анализирует результаты и рассчитывает метрики качества.


#### Запуск 

```bash
python evaluation/evaluate_dataset_run.py \
--dataset-name <имя_датасета> \
--run-name <имя_запуска> \
[--hf-model <модель>] \
[--output <путь_к_выходному_файлу>] \
[--limit <число>]
```

#### Пример запуска

```bash
python evaluation/evaluate_dataset_run.py \
--dataset-name habr-rag-dataset \
--run-name dataset-run-01 \
--hf-model intfloat/multilingual-e5-small \
--output evaluation/logs/my_evaluation.json \
--limit 2
```