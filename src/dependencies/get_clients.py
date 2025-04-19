# src/chatbot_api/dependencies/get_clients.py

import logging
from functools import lru_cache
from typing import Annotated, Optional 
from fastapi import Depends

from ..core.config import Settings, get_settings
from ..modules.embedder import Embedder
from ..modules.vector_store import VectorStore 




logger = logging.getLogger(__name__)

# --- Base Clients & Settings ---

def get_embedder(
    settings: Annotated[Settings, Depends(get_settings)]
) -> Embedder:
    """Embedder 인스턴스를 생성하고 반환합니다."""
    logger.info(f"Initializing Embedder with model: {settings.EMBEDDING_MODEL}")
    return Embedder(api_key=settings.OPENAI_API_KEY, model_name=settings.EMBEDDING_MODEL)


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    """VectorStore (ChromaDB Wrapper) 인스턴스를 생성하고 반환합니다."""
    logger.info(f"Initializing VectorStore at path: {settings.VECTOR_DB_PATH}")
    return VectorStore(db_path=settings.VECTOR_DB_PATH)

