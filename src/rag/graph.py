from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils import create_llm, get_callbacks, get_langfuse_config
from langgraph.graph import END, START, StateGraph

from src.settings import app_settings
from src.logger import logger
from src.rag.prompts import ANSWER_GENERATION_PROMPT, DOC_RETRIEVAL_PROMPT, PARAPHRASE_PROMPT
from src.rag.schemas import Document, RelevantDocumentsResponse
from src.rag.state import RAGState

class RAGGraph:
    def __init__(self, retriever: ContextualCompressionRetriever, callback_config: dict = None):
        self.retriever = retriever
        # Используем with_structured_output для гарантии правильного формата ответа
        base_filter_llm = create_llm(app_settings.llm)
        self.filter_docs_llm = base_filter_llm.with_structured_output(RelevantDocumentsResponse)
        self.paraphrase_llm = create_llm(app_settings.llm)
        self.generate_answer_llm = create_llm(app_settings.llm)
        self.graph = self._build_graph()
        self.callbacks = get_callbacks(callback_config or app_settings.callback.langfuse.model_dump(mode='json') | get_langfuse_config(local=False))

    def _build_graph(self):
        """Строит граф RAG пайплайна"""
        graph_builder = StateGraph(RAGState)
        
        # Добавляем узлы
        graph_builder.add_node("paraphrase_query", self._paraphrase_query)
        graph_builder.add_node("retrieve_docs", self._retrieve_docs)
        graph_builder.add_node("identify_docs", self._identify_relevant_docs)
        graph_builder.add_node("generate_answer", self._generate_answer)
        
        # Добавляем рёбра
        graph_builder.add_edge(START, "paraphrase_query")
        graph_builder.add_edge("paraphrase_query", "retrieve_docs")
        graph_builder.add_edge("retrieve_docs", "identify_docs")
        graph_builder.add_edge("identify_docs", "generate_answer")
        graph_builder.add_edge("generate_answer", END)
        
        return graph_builder.compile()

    async def _paraphrase_query(self, state: RAGState) -> RAGState:
        """Переформулировка запроса пользователя."""
        logger.info(f"Запрос пользователя: {state.query}")
        
        response = await self.paraphrase_llm.ainvoke([
            SystemMessage(content=PARAPHRASE_PROMPT),
            HumanMessage(content=state.query)
        ])
        state.query = response.content.strip()
        
        logger.info(f"Переформулированный запрос: {state.query}")
        return state

    async def _retrieve_docs(self, state: RAGState) -> RAGState:
        """Поиск релевантных документов."""
        logger.info(f"Поиск документов для запроса: {state.query}")
        
        relevant_docs = await self.retriever.ainvoke(state.query)
        state.documents.extend(
            Document(
                id=doc.metadata.get("id", -1),
                author=doc.metadata.get("author", "Unknown"),
                title=doc.metadata.get("title", ""),
                document_id=doc.metadata.get("document_id", -1),
                chunk_id=doc.metadata.get("chunk_id", -1),
                content=doc.page_content,
                url=doc.metadata.get("url"),
            )
            for doc in relevant_docs
        )
        logger.info(f"Найдено {len(state.documents)} релевантных документов")
        return state

    def _format_docs_data(self, documents: list[Document]) -> str:
        """Форматирует документы для передачи в LLM."""
        return "\n\n".join(str(doc) for doc in documents)

    def _filter_documents_by_ids(self, documents: list[Document], relevant_docs_response: RelevantDocumentsResponse) -> list[Document]:
        """Фильтрует документы по релевантным ID из структурированного ответа."""
        # Создаем множество пар (document_id, chunk_id) для быстрого поиска
        relevant_pairs = {
            (item.document_id, item.chunk_id)
            for item in relevant_docs_response.relevant_documents
        }
        
        # Фильтруем документы
        filtered_docs = [
            doc for doc in documents
            if (doc.document_id, doc.chunk_id) in relevant_pairs
        ]
        
        logger.info(f"Отфильтровано документов: {len(filtered_docs)} из {len(documents)}")
        return filtered_docs

    async def _identify_relevant_docs(self, state: RAGState) -> RAGState:
        """Идентификация релевантных ID документов и фильтрация."""
        logger.info("Идентификация релевантных документов")
        docs_data = self._format_docs_data(state.documents)
        
        # Используем структурированный вывод - получаем Pydantic объект напрямую
        relevant_docs_response: RelevantDocumentsResponse = await self.filter_docs_llm.ainvoke([
            SystemMessage(content=DOC_RETRIEVAL_PROMPT),
            HumanMessage(content=f"Документы:\n\n{docs_data}\n\nВопрос: {state.query}")
        ])
        
        # Сохраняем JSON строку для использования в промпте генерации ответа
        state.doc_ids = relevant_docs_response.model_dump_json(indent=2)
        
        # Фильтруем документы по релевантным ID только если включен флаг
        if app_settings.filter_documents:
            state.documents = self._filter_documents_by_ids(state.documents, relevant_docs_response)
            logger.info(f"Идентификация завершена, осталось {len(state.documents)} релевантных документов")
        else:
            logger.info(f"Идентификация завершена, фильтрация отключена, осталось {len(state.documents)} документов")
        return state

    async def _generate_answer(self, state: RAGState) -> RAGState:
        """Генерация ответа."""
        logger.info("Генерация ответа")
        docs_data = self._format_docs_data(state.documents)
        response = await self.generate_answer_llm.ainvoke([
            SystemMessage(content=ANSWER_GENERATION_PROMPT.format(retrieved_data=state.doc_ids)),
            HumanMessage(content=f"Документы:\n\n{docs_data}\n\nВопрос: {state.query}")
        ])
        state.answer = response.content
        logger.info("Ответ сгенерирован успешно")
        return state

    async def run(self, query: str, callbacks=None) -> RAGState:
        """Запускает RAG пайплайн и возвращает итоговое состояние."""
        logger.info(f"Запуск RAG пайплайна: {query}")
        
        try:
            result = await self.graph.ainvoke(RAGState(query=query), config={"callbacks": callbacks or self.callbacks})
            result_state = RAGState(**result)
            logger.info(f"Результат: {result_state.answer}")
            return result_state
        except Exception as e:
            logger.error(f"Ошибка в RAG пайплайне: {e}", exc_info=True)
            return RAGState(query=query, answer="Произошла ошибка при генерации ответа.")
