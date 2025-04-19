import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd

from .config import settings
from ..dependencies.get_clients import get_vector_store, get_embedder # 수정: 클라이언트 가져오기
from src.core.config import get_settings

logger = logging.getLogger(__name__)


app_state = {}

async def _load_and_index_data():
    """
    앱 시작 시 데이터를 로드하고 Vector DB에 인덱싱합니다.
    이미 인덱싱된 경우 건너뛸 수 있도록 간단한 확인 로직 추가 가능.
    """
    settings = get_settings()
    vector_store = get_vector_store(settings) 
    # embedder = get_embedder()

    
    try:
        collection = await vector_store.get_collection()
        if collection.count() > 0:
            logger.info(f"Vector store '{vector_store.collection_name}' already contains {collection.count()} documents. Skipping ingestion.")
            return
    except Exception as e:
        logger.warning(f"Could not check collection status or collection does not exist: {e}. Proceeding with ingestion.")


    logger.info("Loading FAQ data from pickle file...")
    try:
        df = pd.read_pickle(settings.FAQ_DATA_PATH) # 설정에서 경로 가져오기
        logger.info(f"Loaded {len(df)} FAQ entries.")
    except FileNotFoundError:
        logger.error(f"FAQ data file not found at {settings.FAQ_DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return




    logger.info("Data ingestion process completed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    logger.info("Application startup...")
    # Initialize Redis client (using app_state or DI)
    # Initialize ChromaDB client (using app_state or DI)
    # Load Re-ranker model if enabled (using app_state or DI)
    logger.info("Initializing dependencies...")
    # Ensure vector store client is ready before ingestion
    # get_vector_store()
    # get_embedder()

    # Load and index data if needed
    await _load_and_index_data()

    yield

    # --- config.py 에 추가 ---
    # settings = Settings() 에 아래 내용 추가
    # FAQ_DATA_PATH: str = "data/final_result.pkl"