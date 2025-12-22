import json
import httpx
import mlflow
from pathlib import Path
from typing import Any
from datetime import datetime

from src.settings import app_settings


def run_validation(
    dataset_path: str,
    api_url: str = "http://127.0.0.1:8000",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Запустить валидацию RAG системы на датасете с единым run_id.
    
    Args:
        dataset_path (str): Путь к файлу с датасетом (JSONL формат)
        api_url (str): URL API сервера
        run_name (str | None): Имя для run в MLflow. Если None, генерируется автоматически
        
    Returns:
        dict[str, Any]: Результаты валидации с метриками и ответами
    """
    # --------------------------------------------------
    # - Настройка и подготовка к валидации в MLflow -
    # --------------------------------------------------
    dataset = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]
    run_nm = run_name or f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_name = app_settings.mlflow.experiment_name
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_nm) as run:
        run_id = run.info.run_id
        mlflow.log_param("validation_timestamp", datetime.now().isoformat())
        mlflow.log_param("run_type", "validation")
    
    print(f"Создан MLflow Run: {run_id}")
    print(f"Эксперимент: {experiment_name}")
    print(f"Всего вопросов: {len(dataset)}")
    results = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "run_name": run_nm,
        "total_questions": len(dataset),
        "responses": [],
        "errors": [],
    }


    # --------------------------------------------------
    # - Прогон запросов к API -
    # --------------------------------------------------
    with httpx.Client(timeout=60.0) as client:
        for idx, item in enumerate(dataset, 1):
            question = item.get("question", "")
            expected_answer = item.get("answer", "")
            
            print(f"[{idx}/{len(dataset)}] Обработка вопроса: {question[:50]}...")
            
            try:
                response = client.post(
                    f"{api_url}/answer",
                    json={"query": question, "run_id": run_id},
                )
                response.raise_for_status()
                
                result = response.json()
                results["responses"].append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": result.get("answer", ""),
                    "from_cache": result.get("from_cache", False),
                    "sources": result.get("sources", []),
                })
            except Exception as e:
                error_msg = f"Ошибка при обработке вопроса '{question[:50]}...': {e}"
                print(f"ERROR: {error_msg}")
                results["errors"].append({
                    "question": question,
                    "error": str(e),
                })
    

    # --------------------------------------------------
    # - Завершение -
    # --------------------------------------------------
    mlflow.set_tracking_uri(app_settings.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_metric("successful_requests", len(results["responses"]))
        mlflow.log_metric("failed_requests", len(results["errors"]))
    
    print(f"\nВалидация завершена!")
    print(f"Успешно обработано: {len(results['responses'])}")
    print(f"Ошибок: {len(results['errors'])}")
    print(f"Run ID для анализа в MLflow: {run_id}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск валидации RAG системы на датасете")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="src/validation/qa_dataset.jsonl",
        help="Путь к файлу с датасетом (JSONL формат)",
    )
    parser.add_argument(
        "--api-url",
        "-a",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL API сервера",
    )
    parser.add_argument(
        "--run-name",
        "-r",
        type=str,
        default=None,
        help="Имя для run в MLflow (по умолчанию генерируется автоматически)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь к файлу для сохранения результатов (по умолчанию генерируется автоматически)",
    )
    
    args = parser.parse_args()
    
    results = run_validation(args.dataset, args.api_url, args.run_name)
    
    output_file = args.output or f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_file}")
