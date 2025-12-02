from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Модель релевантного документа"""
    id: int | str = Field(description="ID документа из датасета")
    author: str = Field(description="Автор статьи")
    url: Optional[str] = Field(default=None, description="URL оригинального источника (если доступен)")
    title: str = Field(description="Заголовок статьи")
    document_id: int = Field(description="ID чанка внутри документа")
    chunk_id: int = Field(description="Глобальный ID чанка")
    content: str = Field(description="Содержимое документа")
    
    def __str__(self) -> str:
        """Красивое представление документа"""
        paths = [
            f"Document ID: {self.id}",
            f"Title: {self.title}",
            f"Author: {self.author}",
            f"URL: {self.url}",
            f"Document ID: {self.document_id}",
            f"Chunk ID: {self.chunk_id}",
            f"Content: {self.content}",
        ]
        return "\n\n".join(paths)
