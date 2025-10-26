import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.logger import logger


def clean_text(text: str) -> str:
    """Очистка текста перед разбиением на чанки."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Разбивает документы на чанки.

    Args:
        documents (List[str]): Список документов
        chunk_size (int): Размер чанка в символах
        chunk_overlap (int): Перекрытие чанков
    
    Returns:
        List[Document]: Разбитые на чанки документы
    """
    logger.info(f"Разбиваем {len(documents)} документов на чанки (размер: {chunk_size}, перекрытие: {chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents: List[Document] = []

    for i, doc in enumerate(documents):
        clean_doc = clean_text(doc)
        langchain_doc = Document(page_content=clean_doc)
        chunks = text_splitter.split_documents([langchain_doc])

        for j, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = i + 1
            chunk.metadata["chunk_id"] = j + 1
            chunked_documents.append(chunk)

    logger.info(f"Создано {len(chunked_documents)} чанков")
    return chunked_documents
