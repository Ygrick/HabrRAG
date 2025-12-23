import torch
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.chunking import chunk_documents
from src.settings import app_settings
from src.logger import logger


def collection_exists_and_not_empty(
    client: QdrantClient, collection_name: str
) -> bool:
    """
    Проверяет, существует ли коллекция Qdrant и не пуста ли она.

    Args:
        client (QdrantClient): Клиент Qdrant.
        collection_name (str): Имя коллекции.

    Returns:
        bool: True, если коллекция существует и содержит данные, иначе False.
    """
    try:
        exists = client.collection_exists(collection_name=collection_name)
        if not exists:
            return False

        stats = client.count(collection_name=collection_name, exact=True)
        return stats.count > 0
    except Exception as exc:
        logger.warning(
            "Не удалось проверить коллекцию Qdrant: %s", exc
        )
        return False


def _should_reindex(client: QdrantClient, collection_name: str) -> bool:
    """
    Определяет, требуется ли переиндексация коллекции Qdrant.

    Args:
        client (QdrantClient): Клиент Qdrant.
        collection_name (str): Имя коллекции.

    Returns:
        bool: True, если коллекцию нужно пересоздать, иначе False.
    """
    settings = app_settings.qdrant
    if settings.recreate_collection:
        return True

    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as exc:
        logger.warning("Не удалось проверить существование коллекции Qdrant: %s", exc)
        return True

    if not exists:
        return True

    if settings.reindex_on_start:
        return True

    try:
        stats = client.count(collection_name=collection_name, exact=True)
        return stats.count == 0
    except Exception as exc:
        logger.warning("Не удалось получить статистику Qdrant: %s", exc)
        return True


def _prepare_qdrant_collection(
    client: QdrantClient,
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> None:
    """
    Пересоздаёт и заполняет коллекцию Qdrant документами при необходимости.

    Args:
        client (QdrantClient): Клиент Qdrant.
        documents (list[Document]): Чанки документов для индексации.
        embedding_model (HuggingFaceEmbeddings): Модель для вычисления эмбеддингов.

    Returns:
        None
    """
    collection_name = app_settings.qdrant.collection_name
    if not _should_reindex(client, collection_name):
        logger.info("Qdrant коллекция %s уже содержит данные, переиндексация не требуется", collection_name)
        return

    logger.info("Переиндексация коллекции Qdrant (%s)...", collection_name)
    vector_size = len(embedding_model.embed_query("dimension probe"))
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    ingestion_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    ingestion_store.add_documents(documents)
    logger.info("Qdrant коллекция %s заполнена %d документами", collection_name, len(documents))


def create_ensemble_retriever(
    documents: list[Document],
    qdrant_client: QdrantClient,
) -> EnsembleRetriever:
    """
    Создаёт EnsembleRetriever на базе Qdrant + BM25.

    Args:
        documents (list[Document]): Список документов для индексирования.
        qdrant_client (QdrantClient): Клиент Qdrant (embedded или удалённый).

    Returns:
        EnsembleRetriever: Комбинированный ретривер для гибридного поиска.
    """
    logger.info("Инициализация модели эмбеддингов для индексирования...")
    indexing_embedding_model = HuggingFaceEmbeddings(
        model_name=app_settings.retrieval.embedding,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    _prepare_qdrant_collection(qdrant_client, documents, indexing_embedding_model)

    # Освобождаем GPU после индексации
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Создание Qdrant ретривера...")
    query_embedding_model = HuggingFaceEmbeddings(
        model_name=app_settings.retrieval.embedding,
        model_kwargs={"device": "cpu"}
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=app_settings.qdrant.collection_name,
        embedding=query_embedding_model,
    )
    qdrant_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': app_settings.retrieval.qdrant_k}
    )

    logger.info("Создание BM25-ретривера...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = app_settings.retrieval.bm25_k

    logger.info("Объединение ретриверов в EnsembleRetriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[app_settings.retrieval.ensemble_weights_bm25, app_settings.retrieval.ensemble_weights_qdrant]
    )

    logger.info("EnsembleRetriever успешно создан.")
    return ensemble_retriever


def create_qdrant_only_retriever(
    qdrant_client: QdrantClient,
) -> ContextualCompressionRetriever:
    """
    Создаёт ретривер только на основе Qdrant (без BM25).
    Используется когда коллекция уже существует и документы не нужно загружать.

    Args:
        qdrant_client (QdrantClient): Клиент Qdrant.

    Returns:
        ContextualCompressionRetriever: Ретривер с функцией переоценки релевантности.
    """
    logger.info("Создание Qdrant ретривера (коллекция уже существует)...")
    logger.info("Загрузка модели эмбеддингов...")
    query_embedding_model = HuggingFaceEmbeddings(
        model_name=app_settings.retrieval.embedding,
        model_kwargs={"device": "cpu"}
    )
    logger.info("Модель эмбеддингов загружена.")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=app_settings.qdrant.collection_name,
        embedding=query_embedding_model,
    )
    qdrant_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': app_settings.retrieval.qdrant_k}
    )

    logger.info("Создание ретривера с Cross-Encoder Reranking...")
    top_n = app_settings.retrieval.top_k
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=app_settings.retrieval.cross_encoder
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=qdrant_retriever
    )

    logger.info("Ретривер с Cross-Encoder Reranking успешно создан.")
    return compression_retriever


def create_reranked_retriever(
    qdrant_client: QdrantClient, top_n: int = None
) -> ContextualCompressionRetriever:
    """
    Оборачивает ретривер с использованием Cross-Encoder Reranker для переоценки релевантности документов.

    Args:
        qdrant_client (QdrantClient): Клиент Qdrant
        top_n (int, optional): Количество документов, сохраняемых после переоценки. По умолчанию берётся из settings
    
    Returns:
        ContextualCompressionRetriever: Ретривер с функцией переоценки релевантности документов
    """
    documents = chunk_documents()
    retriever = create_ensemble_retriever(documents, qdrant_client)
    top_n = top_n or app_settings.retrieval.top_k
    
    logger.info("Инициализация модели кросс-энкодера для reranking...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=app_settings.retrieval.cross_encoder)
    
    logger.info(f"Создаём CrossEncoderReranker с top_n={top_n}...")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    
    logger.info("Оборачиваем базовый ретривер с помощью ContextualCompressionRetriever...")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )
    
    logger.info("Ретривер с Cross-Encoder Reranking успешно создан.")
    return compression_retriever
