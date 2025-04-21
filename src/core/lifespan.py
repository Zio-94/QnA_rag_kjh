import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd

from .config import settings
from src.dependencies.common import get_vector_store, get_embedder, get_data_ingestion_service, get_retriever # 수정: 클라이언트 가져오기
from src.core.config import get_settings
from src.services.data_ingestion_service import DataIngestionService
from src.modules.retriever import Retriever
logger = logging.getLogger(__name__)


app_state = {}
settings = get_settings()

async def _load_and_index_data():
    """
    앱 시작 시 데이터를 로드하고 Vector DB에 인덱싱합니다.
    이미 인덱싱된 경우 건너뛸 수 있도록 간단한 확인 로직 추가 가능.
    """


 


    logger.info("Data ingestion process completed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    logger.info("Application startup...")

    logger.info("Initializing dependencies...")
    vector_store = get_vector_store()
    embedder = get_embedder()
    retriever = get_retriever()
    data_ingestion_service = DataIngestionService(embedder, vector_store)
    await data_ingestion_service.run_ingestion_pipeline(overwrite=True) 


    # result = await retriever.retrieve("주문이 취소/반품되면 상품의 재고가 있잖아요. 그 재고가 복구되어서 수가 늘어나나요?")
    # logger.info(result)

    yield

    # --- config.py 에 추가 ---
    # settings = Settings() 에 아래 내용 추가
    # FAQ_DATA_PATH: str = "data/final_result.pkl"