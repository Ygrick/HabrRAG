import math
from src.logger import logger
from langchain_openai import ChatOpenAI
from bert_score import score as bert_score
from evaluation.structures import EvaluatedItem
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import HuggingFaceEmbeddings
from typing import Dict, List, Sequence, Optional, Any
from ragas.metrics import AnswerSimilarity, IDBasedContextPrecision, IDBasedContextRecall, BleuScore, RougeScore, ExactMatch, ChrfScore, NonLLMStringSimilarity, FactualCorrectness, Faithfulness



def compute_hit_at_k(items: Sequence[EvaluatedItem], ks: Sequence[int]) -> Dict[str, float]:
    hit_counts = {k: 0 for k in ks}
    relevant_items = 0
    for example in items:
        correct = set(example.correct_chunk_ids)
        if not correct:
            continue
        relevant_items += 1
        retrieved = example.retrieved_chunk_ids
        for k in ks:
            topk = set(retrieved[:k])
            if any(chunk in correct for chunk in topk):
                hit_counts[k] += 1
    total = relevant_items or 1
    return {f"hit@{k}": round(hit_counts[k] / total, 4) for k in ks}

def compute_recall_at_k(items: Sequence[EvaluatedItem], ks: Sequence[int]) -> Dict[str, float]:
    recall_totals = {k: 0.0 for k in ks}
    recall_counts = {k: 0 for k in ks}
    for example in items:
        correct = set(example.correct_chunk_ids)
        if not correct:
            continue
        retrieved = example.retrieved_chunk_ids
        for k in ks:
            topk = list(set(retrieved[:k]))
            recall_totals[k] += sum(1 for chunk in topk if chunk in correct) / len(correct)
            recall_counts[k] += 1
    return {
        f"recall@{k}": round(recall_totals[k] / (recall_counts[k] or 1), 4)
        for k in ks
    }

def compute_mrr(items: Sequence[EvaluatedItem]) -> float:
    mrr_scores: List[float] = []
    for example in items:
        correct = set(example.correct_chunk_ids)
        if not correct:
            continue
        retrieved = example.retrieved_chunk_ids
        first_rank = next((idx for idx, chunk in enumerate(retrieved, start=1) if chunk in correct), None)
        if first_rank:
            mrr_scores.append(1.0 / first_rank)
        else:
            mrr_scores.append(0.0)
    return round(sum(mrr_scores) / (len(mrr_scores) or 1), 4)

def compute_basic_retriever_metrics(
    items: Sequence[EvaluatedItem],
    ks: Sequence[int],
) -> Dict[str, Any]:
    return {
        "hit_at": compute_hit_at_k(items, ks),
        "recall_at": compute_recall_at_k(items, ks),
        "mrr": compute_mrr(items)
    }


def compute_bertscore_metrics(references: List[str], candidates: List[str], hf_model: str) -> Dict[str, float]:
    if not references or not candidates:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_model)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        precision, recall, f1 = bert_score(
            candidates,
            references,
            model_type=hf_model,
            num_layers=num_layers
        )
    except Exception as exc:
        logger.warning(f"Failed to load model config for {hf_model} ({exc}); falling back to lang-only bert_score")
        precision, recall, f1 = bert_score(candidates, references, lang='ru')
    return {
        "bertscore_precision": round(float(precision.mean().item()), 4),
        "bertscore_recall": round(float(recall.mean().item()), 4),
        "bertscore_f1": round(float(f1.mean().item()), 4),
    }

def compute_basic_generator_metrics(
        items: Sequence[EvaluatedItem],
        hf_model: str,
    ) -> Dict[str, Any]:
    references = []
    candidates = []
    for example in items:
        if not example.expected_answer or not example.answer:
            continue
        references.append(example.expected_answer)
        candidates.append(example.answer)
    
    bert_results = compute_bertscore_metrics(references, candidates, hf_model)
    
    return {
        "bertscore": bert_results,
        "samples": len(references),
    }




def _average_metric_score(
    metric: AnswerSimilarity | IDBasedContextPrecision | IDBasedContextRecall | BleuScore | RougeScore | ExactMatch | ChrfScore | NonLLMStringSimilarity | FactualCorrectness | Faithfulness,
    samples: Sequence[SingleTurnSample],
    callbacks: Optional[Sequence[Any]] = None,
) -> float:
    scores: List[float] = []
    for sample in samples:
        try:
            score = metric.single_turn_score(sample, callbacks=callbacks)
        except Exception as exc:
            logger.warning(f"Failed to score sample for {metric.name}: {exc}")
            continue
        if score is None or math.isnan(score):
            continue
        scores.append(score)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def compute_ragas_retriever_metrics(samples: Optional[SingleTurnSample]) -> Dict[str, float]:
    if not samples:
        return {
            "id_based_context_recall": 0.0,
            "id_based_context_precision": 0.0
        }
    
    results = {}

    results["id_based_context_recall"] = _average_metric_score(IDBasedContextRecall(), samples)
    results["id_based_context_precision"] = _average_metric_score(IDBasedContextPrecision(), samples)

    return results


def compute_ragas_generator_metrics(
    samples: Optional[SingleTurnSample],
    hf_model: str,
    llm: ChatOpenAI,
) -> Dict[str, float]:
    if not samples:
        return {
            "samples": 0,
            "similarity_scores": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
            "exact_match": 0.0,
            "chrf_score": 0.0,
            "non_llm_string_similarity": 0.0,
            "factual_correctness": 0.0,
            "faithfulness": 0.0,
        }
    
    results = {}

    results["bleu_score"] = _average_metric_score(BleuScore(), samples)
    results["rouge_score_1_fmeasure"] = _average_metric_score(RougeScore(rouge_type="rouge1", mode="fmeasure"), samples)
    results["rouge_score_l_fmeasure"] = _average_metric_score(RougeScore(rouge_type="rougeL", mode="fmeasure"), samples)
    results["similarity_scores"] = _average_metric_score(
        AnswerSimilarity(embeddings=HuggingFaceEmbeddings(model=hf_model)), samples)

    results["chrf_score"] = _average_metric_score(ChrfScore(), samples)
    results["non_llm_string_similarity"] = _average_metric_score(NonLLMStringSimilarity(), samples)

    if llm:
        results["factual_correctness"] = _average_metric_score(FactualCorrectness(llm=llm), samples)
        results["faithfulness"] = _average_metric_score(Faithfulness(llm=llm), samples)

    return results