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



    EMBEDDING_DIMENSIONS: int = 512 # 예시: 512 차원으로 임베딩 요청 (주석 처리 시 또는 값 없으면 기본 차원)


    # --- Retriever Settings ---
    # RETRIEVER_TOP_K_INITIAL: int = 10
    # RETRIEVER_TOP_N_FINAL: int = 3
    # USE_KEYWORD_FILTER: bool = False
    # MIN_KEYWORD_MATCH: int = 1
    # QUESTION_SIMILARITY_THRESHOLD: float = 0.85 # 질문 유사도 임계값 (조정 필요)
    # VECTOR_DB_COLLECTION_NAME: str = "faq_collection" # 컬렉션 이름

    # --- Reranker Settings ---
    USE_RERANKER: bool = False # 기본값 False
    RERANKER_MODEL_NAME: str = "sentence-transformers/bge-reranker-large" # 예시 모델

    # --- Generator Settings ---
    MAX_GENERATION_TOKENS_MAIN: int = 500 # 메인 답변 최대 토큰
    MAX_GENERATION_TOKENS_FOLLOWUP: int = 100 # 후속 질문 최대 토큰
    GENERATOR_TEMPERATURE: float = 0.7 # 답변 생성 온도
    QUESTION_COLLECTION_NAME: str = "faq_questions"
    QNA_PAIR_COLLECTION_NAME: str = "faq_qna_pairs"

    # --- Vector Service Settings ---
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    BM25_CORPUS_PATH: str = "data/bm25_corpus.pkl"

        # --- Retriever Settings ---
    RETRIEVER_TOP_K_VECTOR: int = 20 # 각 벡터 검색 시 가져올 후보 수
    RETRIEVER_TOP_K_BM25: int = 50   # BM25 검색 시 가져올 후보 수
    RETRIEVER_TOP_N_FINAL_FAQS: int = 5 # RRF 융합 후 최종 선택할 FAQ 수
    RRF_K_VALUE: int = 60            # RRF 융합 파라미터


@lru_cache() # 한번 로드된 설정을 캐시하여 사용
def get_settings() -> Settings:
    return Settings()

settings = get_settings()