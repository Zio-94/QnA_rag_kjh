# src/chatbot_api/modules/embedder.py
import logging
from typing import List, Optional # Optional 추가
import asyncio

from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError, AuthenticationError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from src.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, OpenAIError)
retry_decorator = retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI API call due to {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
)

class Embedder:
    """OpenAI 임베딩 API를 비동기적으로 호출하는 래퍼 클래스 (dimensions 지원)."""
    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        self.dimensions = settings.EMBEDDING_DIMENSIONS
   
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(f"AsyncOpenAI client initialized for model: {self.model_name}")
         
            if self.dimensions:
                
                 if not self.model_name.startswith("text-embedding-3"):
                      logger.warning(f"Dimensions parameter ({self.dimensions}) is set, but model '{self.model_name}' might not support it. Parameter will be ignored by the API if unsupported.")
                 else:
                      logger.info(f"Target embedding dimensions set to: {self.dimensions}")
        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client: {e}")
            raise AuthenticationError(f"Failed to initialize OpenAI client: {e}")

    @retry_decorator
    async def _create_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """OpenAI 임베딩 생성 API 호출 (dimensions 파라미터 포함, 재시도 로직 포함)"""
        api_params = {
            "input": texts,
            "model": self.model_name
        }
        if self.model_name.startswith("text-embedding-3") and self.dimensions is not None:
            api_params["dimensions"] = self.dimensions
            logger.debug(f"Requesting embeddings with dimensions: {self.dimensions}")

        try:
            response = await self.client.embeddings.create(**api_params)
            embeddings = [item.embedding for item in response.data]

            if len(embeddings) != len(texts):
                 logger.error(f"Embedding count mismatch: Input {len(texts)}, Output {len(embeddings)}")
                 raise ValueError("Embedding count mismatch")

            if embeddings:
                 actual_dim = len(embeddings[0])
                 logger.debug(f"Successfully created {len(embeddings)} embeddings with dimension {actual_dim}.")
                 if self.dimensions and actual_dim != self.dimensions:
                      logger.warning(f"Requested dimension was {self.dimensions}, but received {actual_dim}. Model might not support requested dimension.")
            else:
                 logger.debug("Successfully created 0 embeddings.")

            return embeddings
        except AuthenticationError as e:
             logger.critical(f"OpenAI Authentication Error: {e}. Check your API key.")
             raise
        except OpenAIError as e:
            logger.error(f"OpenAI API Error during embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding creation: {e}")
            raise

    async def get_embeddings(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """텍스트 목록을 받아 배치 단위로 임베딩을 생성합니다."""
        if not texts: return []
        all_embeddings = []
        logger.info(f"Requesting embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch_texts)} texts)")
            try:
                batch_embeddings = await self._create_embeddings_with_retry(batch_texts)
                all_embeddings.extend(batch_embeddings)
                await asyncio.sleep(0.1) 
            except Exception as e:
                logger.error(f"Failed to get embeddings for batch starting at index {i}: {e}")
                return []
        if len(all_embeddings) != len(texts):
            logger.error("Final embedding count does not match input text count.")
            return []
        logger.info(f"Successfully retrieved {len(all_embeddings)} embeddings.")
        return all_embeddings


    async def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩을 생성합니다."""
        if not text: raise ValueError("Cannot embed empty text")
        try:
            embeddings = await self.get_embeddings([text], batch_size=1)
            if embeddings: return embeddings[0]
            else: raise RuntimeError("Failed to get embedding for single text")
        except Exception as e:
            logger.error(f"Failed to get single embedding for text '{text[:50]}...': {e}")
            raise