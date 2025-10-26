from typing import Optional

from pydantic import BaseModel
from langchain.retrievers import ContextualCompressionRetriever

from src.rag_graph import RAGGraph


class AppState(BaseModel):
    """Состояние приложения"""
    retriever: Optional[ContextualCompressionRetriever] = None
    rag_graph: Optional[RAGGraph] = None
    cache: Optional[dict] = None
    
    class Config:
        arbitrary_types_allowed = True


class RAGRequest(BaseModel):
    """Модель запроса для получения ответа от RAG"""
    query: str


class RAGResponse(BaseModel):
    """Модель ответа от RAG"""
    answer: str
    from_cache: bool