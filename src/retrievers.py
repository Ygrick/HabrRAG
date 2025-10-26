import torch
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.settings import app_settings
from src.logger import logger


def create_ensemble_retriever(documents: list[Document]) -> EnsembleRetriever:
    """
    Создаёт EnsembleRetriever с использованием GPU для индексирования и CPU для обработки запросов.
    
    Этапы:
      1. Инициализируется модель эмбеддингов на GPU для вычисления эмбеддингов документов.
      2. Создаётся FAISS-индекс с использованием этой модели.
      3. Модель для индексирования удаляется, и вызывается torch.cuda.empty_cache() для освобождения GPU-памяти.
      4. Для вычисления эмбеддингов запроса и поиска создаётся модель на CPU, и векторное хранилище перенастраивается.
      5. Создаются FAISS-retriever и BM25-retriever, которые объединяются в EnsembleRetriever.
    
    Args:
        documents (list[Document]): Список документов для индексирования
    
    Returns:
        EnsembleRetriever: Комбинированный ретривер для поиска
    """
    logger.info("Инициализация модели эмбеддингов для индексирования на GPU...")
    indexing_embedding_model = HuggingFaceEmbeddings(
        model_name=app_settings.retrieval.embedding,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    vector_store = FAISS.from_documents(documents, indexing_embedding_model)
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': app_settings.retrieval.faiss_k}
    )
    
    logger.info("Создание BM25-ретривера...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = app_settings.retrieval.bm25_k
    
    logger.info("Объединение ретриверов в EnsembleRetriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[app_settings.retrieval.ensemble_weights_bm25, app_settings.retrieval.ensemble_weights_faiss]
    )
    
    logger.info("EnsembleRetriever успешно создан.")
    return ensemble_retriever


def create_reranked_retriever(
    retriever: EnsembleRetriever, top_n: int = None
) -> ContextualCompressionRetriever:
    """
    Оборачивает ретривер с использованием Cross-Encoder Reranker для переоценки релевантности документов.

    Args:
        retriever (EnsembleRetriever): Базовый ретривер для первоначального поиска
        top_n (int, optional): Количество документов, сохраняемых после переоценки. По умолчанию берётся из settings
    
    Returns:
        ContextualCompressionRetriever: Ретривер с функцией переоценки релевантности документов
    """
    if top_n is None:
        top_n = app_settings.retrieval.top_k
    
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
