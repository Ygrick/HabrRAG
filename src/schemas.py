from pydantic import BaseModel, Field


class Document(BaseModel):
    """Модель релевантного документа"""
    document_id: int = Field(description="ID документа")
    chunk_id: int = Field(description="ID чанка")
    content: str = Field(description="Содержимое документа")
    
    def __str__(self) -> str:
        """Красивое представление документа"""
        paths = [
            f"Document ID: {self.document_id}",
            f"Chunk ID: {self.chunk_id}",
            f"Content: {self.content}",
        ]
        return "\n\n".join(paths)