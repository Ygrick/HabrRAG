from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    model: str = "qwen/qwen-2.5-7b-instruct:free"
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
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "LangChain. RAG-Агент: Поиск и Генерация Ответов"
    enabled: bool = True


class LoggerSettings(BaseModel):
    level: str = "INFO"
    log_file: str = "./logs/app.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class FastAPISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class AppSettings(BaseSettings):
    cache_path: str = "./cache/answer_cache.json"
    dataset: str = "neural-bridge/rag-dataset-1200"
    split_dataset: str = "test"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    logger: LoggerSettings = Field(default_factory=LoggerSettings)
    fastapi: FastAPISettings = Field(default_factory=FastAPISettings)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="RAG_APP__",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

app_settings = AppSettings()
