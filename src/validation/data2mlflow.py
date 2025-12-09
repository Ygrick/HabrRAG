import json
import mlflow
from pathlib import Path
from mlflow.genai.datasets import create_dataset

from src.settings import app_settings


def load_dataset_to_mlflow(
    dataset_path: str,
    dataset_name: str = "habr_qa_dataset",
) -> str:
    """Загрузить датасет вопросов-ответов в MLflow Evaluation Dataset.
    
    Args:
        dataset_path (str): Путь к файлу с датасетом (JSONL формат)
        dataset_name (str): Имя датасета в MLflow
        
    Returns:
        str: ID созданного датасета
    """
    exp_name = app_settings.mlflow.experiment_name
    
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    experiment = mlflow.get_experiment_by_name(exp_name)
    if not experiment:
        raise ValueError(f"Эксперимент '{exp_name}' не найден")
    
    dataset = create_dataset(
        name=dataset_name,
        experiment_id=[experiment.experiment_id],
    )
    
    records = []
    for line in Path(dataset_path).read_text().splitlines():
        item = json.loads(line)
        records.append({
            "inputs": {"question": item["question"]},
            "expectations": {"expected_answer": item["answer"]},
            "metadata": {
                "document_chunk_id": item["document_chunk_id"],
                "global_chunk_id": item["global_chunk_id"],
                "url": item["url"],
            },
        })
    
    dataset.merge_records(records)
    
    return dataset.dataset_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка датасета в MLflow Evaluation Dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="qa_dataset.jsonl",
        help="Путь к файлу с датасетом (JSONL формат)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="habr_qa_dataset",
        help="Имя датасета в MLflow",
    )
    
    args = parser.parse_args()
    
    dataset_id = load_dataset_to_mlflow(args.dataset, args.name)
    print(f"Датасет загружен: {dataset_id}")

