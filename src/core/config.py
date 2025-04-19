from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # --- 필수 설정 ---
    OPENAI_API_KEY: str
    REDIS_URL: str

    # --- 모델 설정 ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"

    FAQ_DATA_PATH: str = "data/final_result.pkl"

    VECTOR_DB_PATH: str = "chroma_db"


@lru_cache() # 한번 로드된 설정을 캐시하여 사용
def get_settings() -> Settings:
    return Settings()

settings = get_settings()