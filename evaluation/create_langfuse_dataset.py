import json
import argparse
from src.logger import logger
from src.utils import build_langfuse_client, get_langfuse_config


def load_data(file_path):
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError("Unsupported file format. Use .jsonl")
    return data


def create_langfuse_dataset(langfuse_client, data, dataset_name, description="Dataset for RAG evaluation"):
    langfuse_client.create_dataset(name=dataset_name, description=description)

    for item in data:
        input_data = {"question": item["question"]}
        
        expected_output = {
            "answer": item["answer"],
            "id": str(item["id"]),
            "document_chunk_id": str(item["document_chunk_id"]),
            "global_chunk_id": str(item["global_chunk_id"]),
            "url": item["url"],

        }
        metadata = {**{k: v for k, v in item.items() if k not in ["question", "answer", "global_chunk_id", "id", "document_chunk_id", "url"]}}

        langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata
        )

    logger.info(f"Dataset '{dataset_name}' created with {len(data)} items.")


def main():
    parser = argparse.ArgumentParser(description="Create Langfuse dataset from JSON or CSV file")
    parser.add_argument("--file-path", 
                        required=True, 
                        help="Path to the input JSON or CSV file")
    parser.add_argument("--dataset-name", 
                        required=True, 
                        help="Name of the dataset to create in Langfuse")
    parser.add_argument("--description", 
                        default="Dataset for RAG evaluation", 
                        help="Description of the dataset")
    
    args = parser.parse_args()

    data = load_data(args.file_path)

    langfuse_config = get_langfuse_config(local=True)

    langfuse_client = build_langfuse_client(langfuse_config)

    create_langfuse_dataset(langfuse_client, data, args.dataset_name, args.description)

if __name__ == "__main__":
    main()