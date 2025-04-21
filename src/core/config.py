# src/chatbot_api/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional, List

class Settings(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_file_encoding='utf-8')

    # --- 필수 설정 ---
    OPENAI_API_KEY: str
    REDIS_URL: str = "redis://localhost:6379/0" # 기본값 설정 가능

    # --- 모델 설정 ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: Optional[int] = 512 # text-embedding-3용 차원 (None이면 기본값)
    LLM_MODEL: str 

    # --- 데이터 경로 ---
    FAQ_DATA_PATH: str = "data/final_result.pkl" # 원본 데이터 경로

    # --- Vector DB & BM25 설정 ---
    VECTOR_DB_PATH: str = "./chroma_db" # 로컬 Chroma DB 저장 경로
    QUESTION_COLLECTION_NAME: str = "faq_questions" # 질문 벡터 컬렉션 이름
    QNA_PAIR_COLLECTION_NAME: str = "faq_qna_pairs" # Q&A 쌍 벡터 컬렉션 이름
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"   # BM25 인덱스 파일 경로
    BM25_CORPUS_PATH: str = "data/bm25_corpus.pkl" # BM25 코퍼스 맵 파일 경로

    # --- Retriever 설정 ---
    RETRIEVER_TOP_K_VECTOR: int = 20 # 각 벡터 검색 시 가져올 후보 수
    RETRIEVER_TOP_K_BM25: int = 50   # BM25 검색 시 가져올 후보 수
    RETRIEVER_TOP_N_FINAL_FAQS: int = 5 # RRF 융합 후 최종 선택할 FAQ 수
    RRF_K_VALUE: int = 60            # RRF 융합 파라미터
    USE_KEYWORD_FILTER: bool = False # Retriever 내 키워드 필터 (현재 로직에선 우선순위 낮음)
    MIN_KEYWORD_MATCH: int = 1
    # QUESTION_SIMILARITY_THRESHOLD: float = 0.85 # Retriever 내 질문 유사도 (Tiered 로직으로 이동)

    # --- Reranker 설정 ---
    USE_RERANKER: bool = False # Re-ranker 사용 여부
    RERANKER_MODEL_NAME: str = "sentence-transformers/bge-reranker-large"

    # --- Generator 설정 ---
    MAX_GENERATION_TOKENS_MAIN: int = 500 # 메인 답변 최대 토큰
    MAX_GENERATION_TOKENS_FOLLOWUP: int = 100 # 후속 질문 최대 토큰
    GENERATOR_TEMPERATURE: float = 0.7 # 답변 생성 온도

    # --- Guard 설정 ---
    GUARD_CLASSIFIER_MODEL: str 

    # --- Tiered Logic Thresholds ---
    HIGH_THRESHOLD: float = 0.7   # 높은 유사도 기준 (직접 답변 유도)
    MEDIUM_THRESHOLD: float = 0.5 # 중간 유사도 기준 (표준 RAG)
    LARGE_GAP_THRESHOLD: float = 0.04 # Top1 vs Top2 유사도 차이 기준 (신뢰도)

    # --- Memory 설정 ---
    CONTEXT_TURNS: int = 3 # 프롬프트에 포함할 최근 대화 턴 수
    REDIS_TTL_SECONDS: int = 12 * 60 * 60 # Redis TTL (48시간)
    REDIS_URL: str = "redis://localhost:6379/0" # Redis URL

    # --- 앱 정보 ---
    APP_TITLE: str = "SmartStore FAQ Chatbot API"
    APP_VERSION: str = "0.2.0" # 버전 업데이트

@lru_cache()
def get_settings() -> Settings:
    # 설정 파일 로딩 시 유효성 검사 등 추가 가능
    return Settings()

settings = get_settings() # 전역 설정 객체 (필요시 사용)