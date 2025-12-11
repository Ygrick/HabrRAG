from src.rag.schemas import Document
from src.schemas import SourceInfo


def build_sources(documents: list[Document]) -> list[SourceInfo]:
    """Группирует чанки по document_id и готовит метаданные источников.
    
    Args:
        documents (list[Document]): Список документов для группировки
    
    Returns:
        list[SourceInfo]: Список источников с метаданными
    """
    grouped: dict[int, SourceInfo] = {}

    for doc in documents:
        doc_id = int(doc.id)
        if doc_id == -1:
            continue

        source = grouped.setdefault(
            doc_id,
            SourceInfo(
                document_id=doc_id,
                chunk_ids=[],
                url=doc.url,
                preview=None,
            )
        )

        if doc.chunk_id not in source.chunk_ids:
            source.chunk_ids.append(doc.chunk_id)

        if doc.url and not source.url:
            source.url = doc.url

        if not source.preview:
            source.preview = doc.content[:200].strip()

    return list(grouped.values())
