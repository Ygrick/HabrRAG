from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvaluatedItem:
    dataset_item_id: str
    run_item_id: str
    trace_id: str
    question: str
    expected_answer: str
    answer: str
    correct_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    retrieved_contexts: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_item_id": self.dataset_item_id,
            "run_item_id": self.run_item_id,
            "trace_id": self.trace_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "answer": self.answer,
            "correct_chunk_ids": self.correct_chunk_ids,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_contexts": self.retrieved_contexts
        }