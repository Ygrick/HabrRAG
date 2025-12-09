from typing import Optional, Any, List, Union

from langchain.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field

from src.rag.graph import RAGGraph


class AppState(BaseModel):
    """Состояние приложения"""
    retriever: Optional[ContextualCompressionRetriever] = None
    rag_graph: Optional[RAGGraph] = None
    qdrant_client: Optional[QdrantClient] = None
    mlflow_process: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True


class RAGRequest(BaseModel):
    """Модель запроса для получения ответа от RAG"""
    query: str
    run_id: Optional[str] = None


class SourceInfo(BaseModel):
    """Метаданные источников, использованных в ответе."""
    document_id: int
    chunk_ids: List[int]
    url: Optional[str] = None
    preview: Optional[str] = None


class RAGResponse(BaseModel):
    """Модель ответа от RAG"""
    answer: str
    from_cache: bool
    sources: List[SourceInfo] = Field(default_factory=list)


class SummarizationRequest(BaseModel):
    """Запрос на суммаризацию конкретного источника."""
    document_id: int
    chunk_ids: Optional[List[int]] = None


class SummarizationResponse(BaseModel):
    """Ответ с суммаризацией статьи/документа."""
    summary: str
