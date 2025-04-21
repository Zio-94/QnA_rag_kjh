# src/chatbot_api/dependencies/common.py
import logging
from functools import lru_cache
from typing import Annotated, Optional
import redis.asyncio as redis
from fastapi import Depends

# 내부 모듈 및 서비스 임포트 (경로 확인 필수)
from src.core.config import Settings, get_settings
from src.modules.embedder import Embedder
from src.modules.vector_store import VectorStore
# from ..modules.reranker import ReRanker
# from ..modules.category_filter import CategoryFilter # 현재 사용 안 함
# from ..modules.memory import ConversationMemory
from src.modules.guard import Guard
from src.modules.generator import ChatGenerator
from src.modules.retriever import Retriever
from src.services.chat_service import ChatService
from src.modules.query_analyzer import QueryAnalyzer
from src.services.data_ingestion_service import DataIngestionService
# from ..services.vector_service import DataIngestionService # lifespan에서만 직접 사용 가정

logger = logging.getLogger(__name__)
settings = get_settings()
# --- 기본 클라이언트 및 설정 ---

@lru_cache()
def get_redis_client() -> redis.Redis:
    """Redis 클라이언트 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting Redis client instance...")
    try:
        pool = redis.ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)
        client = redis.Redis.from_pool(pool)
        # 앱 시작 시 ping은 lifespan에서 처리
        return client
    except Exception as e:
        logger.critical(f"Failed to create Redis client: {e}")
        raise # 필수 의존성 오류 시 앱 중단

# --- 핵심 모듈 의존성 ---

@lru_cache(maxsize=None)
def get_embedder() -> Embedder:
    """Embedder 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting Embedder instance...")
    try:
        return Embedder(api_key=settings.OPENAI_API_KEY, model_name=settings.EMBEDDING_MODEL, dimensions=settings.EMBEDDING_DIMENSIONS)
    except Exception as e:
        logger.critical(f"Failed to initialize Embedder: {e}")
        raise

@lru_cache(maxsize=None)
def get_vector_store() -> VectorStore:
    """VectorStore 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting VectorStore instance...")
    try:
        return VectorStore(db_path=settings.VECTOR_DB_PATH)
    except Exception as e:
        logger.critical(f"Failed to initialize VectorStore: {e}")
        raise

# @lru_cache(maxsize=None)
# def get_reranker(
#     settings: Annotated[Settings, Depends(get_settings)] = Depends(get_settings)
# ) -> Optional[ReRanker]:
#     """설정에 따라 ReRanker 인스턴스 또는 None을 반환합니다 (캐싱됨)."""
#     logger.debug("Getting ReRanker instance (if enabled)...")
#     if not settings.USE_RERANKER:
#         return None
#     try:
#         # ReRanker 초기화 시 모델 로딩 등 시간 소요 가능
#         return ReRanker(model_name=settings.RERANKER_MODEL_NAME)
#     except Exception as e:
#         logger.error(f"Failed to initialize ReRanker, disabling: {e}")
#         return None # 실패 시 None 반환

# @lru_cache(maxsize=None)
# def get_memory(
#     settings: Annotated[Settings, Depends(get_settings)] = Depends(get_settings),
#     redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = Depends(get_redis_client)
# ) -> ConversationMemory:
#     """ConversationMemory 인스턴스를 반환합니다 (캐싱됨)."""
#     logger.debug("Getting ConversationMemory instance...")
#     try:
#         return ConversationMemory(redis_client=redis_client, ttl_seconds=settings.REDIS_TTL_SECONDS)
#     except Exception as e:
#         logger.critical(f"Failed to initialize ConversationMemory: {e}")
#         raise

@lru_cache(maxsize=None)
def get_guard() -> Guard:
    """Guard 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting Guard instance...")
    try:
        # Guard는 현재 OpenAI API 키만 필요
        return Guard(api_key=settings.OPENAI_API_KEY)
    except Exception as e:
        logger.critical(f"Failed to initialize Guard: {e}")
        raise

@lru_cache(maxsize=None)
def get_generator() -> ChatGenerator:
    """ChatGenerator 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting ChatGenerator instance...")
    try:
        return ChatGenerator(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL
            # 설정값(max_tokens 등)은 Generator 내부에서 settings 참조
        )
    except Exception as e:
        logger.critical(f"Failed to initialize ChatGenerator: {e}")
        raise

@lru_cache(maxsize=None)
def get_retriever(
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    # reranker: Annotated[Optional[ReRanker], Depends(get_reranker)] = Depends(get_reranker)
) -> Retriever:
    """Retriever 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting Retriever instance...")
    try:
        # 설정값은 Retriever 내부에서 settings 참조
        return Retriever(
            embedder=embedder,
            vector_store=vector_store,
            # reranker=reranker
        )
    except Exception as e:
        logger.critical(f"Failed to initialize Retriever: {e}")
        raise

@lru_cache(maxsize=None)
def get_query_analyzer(
    guard: Guard = Depends(get_guard),
    retriever: Retriever = Depends(get_retriever)
) -> QueryAnalyzer:
    """QueryAnalyzer 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting QueryAnalyzer instance...")
    return QueryAnalyzer(
        guard=guard,
        retriever=retriever
    )

# --- 서비스 계층 의존성 ---

# ChatService는 일반적으로 요청마다 생성될 수도 있지만,
# 상태를 가지지 않는다면 캐싱해도 무방합니다. 여기서는 캐싱 예시.
@lru_cache(maxsize=None)
def get_chat_service(
    retriever: Retriever = Depends(get_retriever),
    generator: ChatGenerator = Depends(get_generator),
    guard: Guard = Depends(get_guard),
    query_analyzer: QueryAnalyzer = Depends(get_query_analyzer)
) -> ChatService:
    """ChatService 인스턴스를 반환합니다 (캐싱됨)."""
    logger.debug("Getting ChatService instance...")
    try:
        return ChatService(
            retriever=retriever,
            generator=generator,
            guard=guard,
            query_analyzer=query_analyzer
            # memory=memory
            # 필요시 settings=settings 전달
        )
    except Exception as e:
        logger.critical(f"Failed to initialize ChatService: {e}")
        raise

@lru_cache(maxsize=None)
def get_data_ingestion_service(
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store)
) -> DataIngestionService:
    """DataIngestionService 인스턴스를 반환합니다 (캐싱됨)."""
    try:
        # Depends 값을 직접 사용하지 않고 함수를 호출
        return DataIngestionService(
            embedder=embedder,
            vector_store=vector_store 
        )
    except Exception as e:
        logger.critical(f"Failed to initialize DataIngestionService: {e}")
        raise
