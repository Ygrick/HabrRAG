from typing import Optional, Any

from langchain.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient
from pydantic import BaseModel

from src.rag.graph import RAGGraph


class AppState(BaseModel):
    """Состояние приложения"""
    retriever: Optional[ContextualCompressionRetriever] = None
    rag_graph: Optional[RAGGraph] = None
    cache: Optional[dict] = None
    qdrant_client: Optional[QdrantClient] = None
    mlflow_process: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True


class RAGRequest(BaseModel):
    """Модель запроса для получения ответа от RAG"""
    query: str


class RAGResponse(BaseModel):
    """Модель ответа от RAG"""
    answer: str
    from_cache: bool
