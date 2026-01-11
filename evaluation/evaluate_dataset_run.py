import os
import argparse
import json
import asyncio
import nest_asyncio
import time
from pathlib import Path
from src.logger import logger
from langfuse import Langfuse
from dotenv import load_dotenv
from ragas.dataset_schema import SingleTurnSample
from src.utils import build_langfuse_client, get_langfuse_config
from src.settings import app_settings
from typing import Any, Dict, List, Optional, Tuple
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails

from evaluation.structures import EvaluatedItem
from evaluation.metrics import compute_basic_retriever_metrics, compute_basic_generator_metrics, \
            compute_ragas_generator_metrics, compute_ragas_retriever_metrics
from src.utils import (
    build_langfuse_client,
    create_llm,
    serialize_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Langfuse dataset run using trace data."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Langfuse dataset name."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the dataset run to evaluate (must already exist in Langfuse)."
    )
    parser.add_argument(
        "--hf-model",
        help="HuggingFace model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to dump results as JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of items to evaluate.",
    )
    return parser.parse_args()


def format_question(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return payload.get("question", "")
    return str(payload)


def format_expected_answer(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return payload.get("answer", "")
    return str(payload)


def extract_correct_chunk_ids(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        # chunk_ids = payload.get("id", []) # если мы хотим оценивать по ID статьи
        chunk_ids = payload.get("global_chunk_id", [])

        if isinstance(chunk_ids, str): # строка вида "1,2,3"
            return [int(x.strip()) for x in chunk_ids.split(",") if x.strip().isdigit()]

        elif isinstance(chunk_ids, list): # список строк/чисел
            result = []
            for item in chunk_ids:
                if isinstance(item, int):
                    result.append(item)
                elif isinstance(item, str) and item.strip().isdigit():
                    result.append(int(item.strip()))
            return result
    return []


def extract_answer_from_trace(trace: TraceWithFullDetails) -> str:
    try:
        answer = trace.output['answer']
    except Exception as e:
        answer = ""
        logger.warning(f"Failed to extract answer from trace {trace.id}: {e}")
    return answer


def extract_documents_from_trace(trace: TraceWithFullDetails) -> Tuple[List[str], List[str]]:
    docs_ids = []
    docs = []
    try:
        for observation in trace.observations:
            if observation.name == "retrieve_docs":
                if observation.output is None:
                    logger.warning(f"Trace {trace.id} retrieve_docs observation has no output")
                    return docs_ids, docs
                
                for doc in observation.output["documents"]:
                    # docs_ids.append(doc["id"]) # если мы хотим оценивать по ID статьи
                    docs_ids.append(doc["chunk_id"]-1) 
                    docs.append(doc["content"])
                return docs_ids, docs
        logger.warning(f"Trace {trace.id} does not contain retrieve_docs observation, cannot extract documents")
        return docs_ids, docs
    except Exception as e:
        logger.warning(f"Failed to extract documents from trace {trace.id}: {e}")
        return docs_ids, docs


def extract_duration_from_trace(trace: TraceWithFullDetails) -> Optional[float]:
    try:
        if hasattr(trace, 'latency') and trace.latency is not None:
            logger.debug(f"Trace {trace.id} has latency: {trace.latency}")
            return float(trace.latency)
        else:
            return None
    except Exception as e:
        logger.warning(f"Failed to extract duration from trace {trace.id}: {e}")
        return None


def evaluate_run(
    langfuse_client: Langfuse,
    dataset_name: str,
    run_name: str,
    limit: Optional[int] = None
) -> Tuple[List[EvaluatedItem], Dict[str, Any]]:
    dataset = langfuse_client.get_dataset(dataset_name)
    dataset_items = {item.id: item for item in dataset.items}
    run = langfuse_client.get_dataset_run(dataset_name, run_name)

    examples = []
    if not run.dataset_run_items:
        raise Exception("Dataset run does not include any run items")

    for run_item in (run.dataset_run_items[:limit] if limit else run.dataset_run_items):
        dataset_item = dataset_items.get(run_item.dataset_item_id)

        trace_id = run_item.trace_id

        trace = langfuse_client.client.trace.get(trace_id)

        answer = extract_answer_from_trace(trace)

        if answer:
            retrieved_chunk_ids, retrieved_contexts = extract_documents_from_trace(trace)

            examples.append(
                EvaluatedItem(
                    dataset_item_id=dataset_item.id,
                    run_item_id=run_item.id,
                    trace_id=trace_id,
                    question=format_question(dataset_item.input),

                    expected_answer=format_expected_answer(dataset_item.expected_output),
                    answer=answer,

                    correct_chunk_ids=extract_correct_chunk_ids(dataset_item.expected_output),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_contexts=retrieved_contexts,
                    duration=extract_duration_from_trace(trace)
                )
            )
        else:
            logger.warning(f"Trace {trace_id} has no answer, skipping evaluation for this item")

    summary = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "run_id": run.id,
        "total_items": len(examples),
    }
    return examples, summary




def _build_ragas_sample(item: EvaluatedItem) -> Optional[SingleTurnSample]:
    return SingleTurnSample(
        user_input=item.question,
        response=item.answer,
        reference=item.expected_answer,
        retrieved_context_ids=item.retrieved_chunk_ids,
        reference_context_ids=item.correct_chunk_ids,
        retrieved_contexts=item.retrieved_contexts
    )



async def main() -> None:
    nest_asyncio.apply()
    args = parse_args()
    langfuse_config = get_langfuse_config(local=True)
    eval_config = app_settings.evaluation

    hf_model = args.hf_model or eval_config.hf_model

    langfuse_client = build_langfuse_client(langfuse_config)

    start_time = time.time()

    items, summary = evaluate_run(
        langfuse_client=langfuse_client,
        dataset_name=args.dataset_name,
        run_name=args.run_name,
        limit=args.limit
    )

    summary["params"] = serialize_args(args)

    durations = [item.duration for item in items if item.duration is not None]
    total_run_time = sum(durations) if durations else 0
    average_run_time = total_run_time / len(durations) if durations else 0
    summary["total_run_time_seconds"] = total_run_time
    summary["average_run_time_seconds"] = average_run_time

    basic_retriever_metrics = compute_basic_retriever_metrics(items, [1, 2, 3, 5])
    basic_generator_metrics = compute_basic_generator_metrics(items, hf_model)

    ragas_samples = [_build_ragas_sample(item) for item in items]

    llm = create_llm(eval_config.llm, use_json_response=True)

    ragas_retriever_metrics = await compute_ragas_retriever_metrics(ragas_samples)
    ragas_generator_metrics = await compute_ragas_generator_metrics(ragas_samples, hf_model=hf_model, llm=llm)

    end_time = time.time()
    total_time = end_time - start_time
    average_time_per_item = total_time / len(items) if items else 0

    summary["total_time_minutes_evaluation"] = total_time / 60
    summary["average_time_per_item_minutes_evaluation"] = average_time_per_item / 60

    report = {
        "summary": summary,
        "params_rag": app_settings.retrieval.model_dump(),
        "retriever": basic_retriever_metrics | ragas_retriever_metrics,
        "generator": basic_generator_metrics | ragas_generator_metrics,
        "details": [item.as_dict() for item in items],
    }

    logger.info(f"Evaluation complete: {json.dumps(report['summary'], indent=2)}")
    logger.info(f"Retriever metrics: {report['retriever']}")
    logger.info(f"Generator metrics: {report['generator']}")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.info(f"Average time per item: {average_time_per_item:.2f} seconds ({average_time_per_item / 60:.4f} minutes)")
    logger.info(f"Total run time (from traces): {total_run_time:.2f} seconds ({total_run_time / 60:.2f} minutes)")
    logger.info(f"Average run time per item (from traces): {average_run_time:.2f} seconds ({average_run_time / 60:.4f} minutes)")

    if args.output:
        output_path = args.output
    else:
        output_dir = eval_config.default_output_dir
        output_pattern = eval_config.default_output_pattern
        output_path = output_pattern.format(dataset_name=args.dataset_name, run_name=args.run_name)
        output_path = Path(output_dir) / output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved detailed evaluation to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())