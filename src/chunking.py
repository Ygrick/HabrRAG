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


def chunk_documents(documents: List[dict], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Разбивает документы на чанки с добавлением метаданных.

    Args:
        documents (List[dict]): Список словарей с данными документов
        chunk_size (int): Размер чанка в символах
        chunk_overlap (int): Перекрытие чанков
    
    Returns:
        List[Document]: Разбитые на чанки документы с метаданными
    """
    logger.info(f"Разбиваем {len(documents)} документов на чанки (размер: {chunk_size}, перекрытие: {chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents: List[Document] = []
    global_chunk_id = 1

    for i, doc_data in enumerate(documents):
        # Извлекаем текст и метаданные из документа
        text = doc_data.get('text_markdown', '')
        clean_doc = clean_text(text)
        langchain_doc = Document(page_content=clean_doc)
        chunks = text_splitter.split_documents([langchain_doc])

        for j, chunk in enumerate(chunks):
            # Добавляем метаданные в каждый чанк
            chunk.metadata["id"] = doc_data.get('id', i + 1)  # ID документа из датасета
            chunk.metadata["author"] = doc_data.get('author', 'Unknown')  # Автор
            chunk.metadata["url"] = doc_data.get('url', '')  # Ссылка на статью
            chunk.metadata["title"] = doc_data.get('title', '')  # Заголовок статьи
            chunk.metadata["document_chunk_id"] = j + 1  # ID чанка внутри документа
            chunk.metadata["global_chunk_id"] = global_chunk_id  # Глобальный ID чанка
            global_chunk_id += 1
            chunked_documents.append(chunk)

    logger.info(f"Создано {len(chunked_documents)} чанков")
    return chunked_documents
