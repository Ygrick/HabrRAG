from pydantic import BaseModel, Field


class Document(BaseModel):
    """Модель релевантного документа"""
    id: int | str = Field(description="ID документа из датасета")
    author: str = Field(description="Автор статьи")
    url: str = Field(description="Ссылка на статью")
    title: str = Field(description="Заголовок статьи")
    document_chunk_id: int = Field(description="ID чанка внутри документа")
    global_chunk_id: int = Field(description="Глобальный ID чанка")
    text_markdown: str = Field(description="Содержимое документа")
    
    def __str__(self) -> str:
        """Красивое представление документа"""
        paths = [
            f"Document ID: {self.id}",
            f"Title: {self.title}",
            f"Author: {self.author}",
            f"URL: {self.url}",
            f"Document Chunk ID: {self.document_chunk_id}",
            f"Global Chunk ID: {self.global_chunk_id}",
            f"Text: {self.text_markdown}",
        ]
        return "\n\n".join(paths)