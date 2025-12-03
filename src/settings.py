from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    model: str = "qwen/qwen-2.5-7b-instruct"
    temperature: float = 0.6
    max_tokens: int = 1000
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = SecretStr("")


class RetrievalSettings(BaseModel):
    embedding: str = "intfloat/multilingual-e5-large"
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    k: int = 2
    top_k: int = 3
    bm25_k: int = 2
    faiss_k: int = 2
    ensemble_weights_bm25: float = 0.4
    ensemble_weights_faiss: float = 0.6


class MLflowSettings(BaseModel):
    tracking_uri: str = "http://127.0.0.1:5001"
    experiment_name: str = "LangChain. RAG-Агент: Поиск и Генерация Ответов"
    enabled: bool = True
    backend_store_uri: str = "sqlite:///./mlflow/mlflow.db"
    artifact_root: str = "./mlflow/artifacts"
    auto_start: bool = True
    startup_timeout_seconds: int = 30


class QdrantSettings(BaseModel):
    """
    Настройки подключения к Qdrant.

    Attributes:
        collection_name (str): Имя коллекции для хранения документов.
        path (str): Путь к embedded-хранилищу Qdrant.
        url (str | None): URL удалённого Qdrant сервера.
        api_key (SecretStr | None): API ключ для удалённого доступа.
        prefer_grpc (bool): Использовать ли gRPC транспорт.
        recreate_collection (bool): Принудительно пересоздавать коллекцию при старте.
        reindex_on_start (bool): Переиндексировать коллекцию, если она пуста.
    """
    collection_name: str = "habr_rag"
    path: str = "./qdrant_data"
    url: str | None = None
    api_key: SecretStr | None = None
    prefer_grpc: bool = False
    recreate_collection: bool = False
    reindex_on_start: bool = False


class LoggerSettings(BaseModel):
    level: str = "INFO"
    log_file: str = "./logs/app.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class FastAPISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

class CacheDatabaseSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5432
    user: str = "chainlit_user"
    password: SecretStr = SecretStr("chainlit_pass")
    database: str = "postgres_db"
    cache_table: str = "answer_cache"

class ChainlitDatabaseSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5432
    user: str = "chainlit_user"
    password: SecretStr = SecretStr("chainlit_pass")
    database: str = "chainlit_db"
    fastapi_service_url: str = "http://app:8000"

class AppSettings(BaseSettings):
    dataset: str = "IlyaGusev/habr"
    split_dataset: str = "train[:1000]"
    dataset_column: str = "text_markdown"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    logger: LoggerSettings = Field(default_factory=LoggerSettings)
    fastapi: FastAPISettings = Field(default_factory=FastAPISettings)
    chainlit: ChainlitDatabaseSettings = Field(default_factory=ChainlitDatabaseSettings)
    database: CacheDatabaseSettings = Field(default_factory=CacheDatabaseSettings)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="RAG_APP__",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

app_settings = AppSettings()