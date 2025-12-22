import argparse
import asyncio
import httpx
from src.logger import logger
from datetime import datetime
from typing import Optional
from src.settings import app_settings

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG chain on Langfuse dataset"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Langfuse dataset",
    )
    parser.add_argument(
        "--run-name",
        help="Name for the Langfuse dataset run (default adds timestamp)",
    )
    parser.add_argument(
        "--run-description",
        default="Langfuse dataset run for RAG chain",
        help="Description that will be attached to every dataset run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of dataset items to process",
    )
    return parser.parse_args()


def build_run_name(dataset_name: str, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{dataset_name}-run-{timestamp}"



async def run_dataset(
        dataset_name: str,
        run_name: str,
        run_description: str,
        api_url: str,
        limit: Optional[int] = None
    ) -> None:
    async with httpx.AsyncClient(timeout=3*3600.0) as client:
        logger.info(f"Running dataset {dataset_name} with run_name {run_name}")
        try:
            response = await client.post(
                f"{api_url}/run_dataset",
                json={
                    "dataset_name": dataset_name,
                    "run_name": run_name,
                    "run_description": run_description,
                    "limit": limit
                },
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "failed":
                logger.error(f"Dataset run failed: {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"Dataset run completed successfully, processed items: {result.get('processed_items', 'unknown')}")
        except Exception as failure:
            logger.error(f"Failed to run dataset: {failure}")


def main() -> None:
    args = parse_args()

    run_name = build_run_name(args.dataset_name, args.run_name)
    api_url = app_settings.rag_chain.api_url
    asyncio.run(run_dataset(
        dataset_name=args.dataset_name,
        run_name=run_name, 
        run_description=args.run_description, 
        api_url=api_url,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()
