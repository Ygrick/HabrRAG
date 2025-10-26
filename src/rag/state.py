from pydantic import BaseModel, Field

from src.schemas import Document


class RAGState(BaseModel):
    """Состояние RAG пайплайна"""
    query: str = Field(default="")
    documents: list[Document] = Field(default_factory=list)
    doc_ids: str = Field(default="")
    answer: str = Field(default="")
    
    class Config:
        arbitrary_types_allowed = True