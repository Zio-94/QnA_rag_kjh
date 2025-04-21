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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    logger.info("Application startup...")

    logger.info("Initializing dependencies...")
    vector_store = get_vector_store()
    embedder = get_embedder()
    retriever = get_retriever()
    data_ingestion_service = DataIngestionService(embedder, vector_store)
    await data_ingestion_service.run_ingestion_pipeline(overwrite=False) 


    yield

