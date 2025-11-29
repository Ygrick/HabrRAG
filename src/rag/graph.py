from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.settings import app_settings
from src.logger import logger
from src.prompts import ANSWER_GENERATION_PROMPT, DOC_RETRIEVAL_PROMPT, PARAPHRASE_PROMPT
from src.rag.schemas import Document
from src.rag.state import RAGState


class RAGGraph:
    def __init__(self, retriever: ContextualCompressionRetriever):
        self.retriever = retriever
        self.filter_docs_llm = ChatOpenAI(
            model=app_settings.llm.model,
            temperature=app_settings.llm.temperature,
            max_tokens=app_settings.llm.max_tokens,
            base_url=app_settings.llm.base_url,
            api_key=app_settings.llm.api_key.get_secret_value(),
        )
        self.paraphrase_llm = ChatOpenAI(
            model=app_settings.llm.model,
            temperature=app_settings.llm.temperature,
            max_tokens=app_settings.llm.max_tokens,
            base_url=app_settings.llm.base_url,
            api_key=app_settings.llm.api_key.get_secret_value(),
        )
        self.generate_answer_llm = ChatOpenAI(
            model=app_settings.llm.model,
            temperature=app_settings.llm.temperature,
            max_tokens=app_settings.llm.max_tokens,
            base_url=app_settings.llm.base_url,
            api_key=app_settings.llm.api_key.get_secret_value(),
        )
        self.graph = self._build_graph()

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
        
        # Вызов ретривера
        relevant_docs = await self.retriever.ainvoke(state.query)

        # Обновление состояния
        for doc in relevant_docs:
            state.documents.append(
                Document(
                    document_id=doc.metadata.get("document_id", -1),
                    chunk_id=doc.metadata.get("chunk_id", -1),
                    content=doc.page_content
                )
            )
        logger.info(f"Найдено {len(state.documents)} релевантных документов")
        return state

    async def _identify_relevant_docs(self, state: RAGState) -> RAGState:
        """Идентификация релевантных ID документов."""
        logger.info("Идентификация релевантных документов")

        docs_data = "\n\n".join([str(doc) for doc in state.documents])
        response = await self.filter_docs_llm.ainvoke([
            SystemMessage(content=DOC_RETRIEVAL_PROMPT),
            HumanMessage(content=f"Документы:\n\n{docs_data}\n\nВопрос: {state.query}")
        ])
        state.doc_ids = response.content
        
        logger.info(f"Идентификация завершена")
        return state

    async def _generate_answer(self, state: RAGState) -> RAGState:
        """Генерация ответа."""
        logger.info("Генерация ответа")

        docs_data = "\n\n".join([str(doc) for doc in state.documents])
        response = await self.generate_answer_llm.ainvoke([
            SystemMessage(content=ANSWER_GENERATION_PROMPT.format(retrieved_data=state.doc_ids)),
            HumanMessage(content=f"Документы:\n\n{docs_data}\n\nВопрос: {state.query}")
        ])
        state.answer = response.content

        logger.info(f"Ответ сгенерирован успешно")
        return state

    async def run(self, query: str) -> str:
        """Запускает RAG пайплайн."""
        logger.info(f"Запуск RAG пайплайна: {query}")
        
        try:
            # Инициализация состояния
            initial_state = RAGState(query=query)
            
            # Запуск графа
            result = await self.graph.ainvoke(initial_state)
            result = RAGState(**result)
            logger.info(f"Результат: {result.answer}")
            return result.answer
        
        except Exception as e:
            logger.error(f"Ошибка в RAG пайплайне: {e}", exc_info=True)
            return "Произошла ошибка при генерации ответа."
