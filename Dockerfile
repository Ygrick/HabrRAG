# Базовый образ с CUDA 12.1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Устанавливаем Python3 и pip, и чистим кэш для оптимизации docker-контейнера
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Открываем порт для FastAPI
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
