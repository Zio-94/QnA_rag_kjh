# src/services/vector_service.py (또는 src/modules/indexing.py)
import sys
import os
import asyncio
import logging
from pathlib import Path
import pandas as pd
import tiktoken
from tqdm.asyncio import tqdm
import re
import hashlib
import pickle # BM25 인덱스 저장용

# pip install rank_bm25 # BM25 라이브러리 설치 필요

# --- 프로젝트 루트 설정 및 경로 추가 ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- 설정 및 모듈 임포트 ---
from src.core.config import settings 
from src.modules.embedder import Embedder
from src.modules.vector_store import VectorStore
# BM25 라이브러리 임포트
from rank_bm25 import BM25Okapi

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_TOKENIZER_MODEL = "cl100k_base"
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 50
INGESTION_BATCH_SIZE = 64

QUESTION_COLLECTION_NAME = settings.QUESTION_COLLECTION_NAME 
QNA_PAIR_COLLECTION_NAME = settings.QNA_PAIR_COLLECTION_NAME 
BM25_INDEX_PATH = settings.BM25_INDEX_PATH 
BM25_CORPUS_PATH = settings.BM25_CORPUS_PATH 

class DataIngestionService: 
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """DataIngestionService 초기화"""
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("DataIngestionService initialized.")

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        feedback_patterns = [
            r"위 도움말이 도움이 되었나요\?.*", r"별점\d점.*", r"소중한 의견을 남겨주시면.*",
            r"관련 도움말/키워드.*", r"도움말 닫기.*"
        ]
        cleaned_text = text
        for pattern in feedback_patterns:
             cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text

    @staticmethod
    def chunk_text_with_token_limit(
        text: str,
        max_tokens: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    ) -> list[str]:
        """Tiktoken 기반 청킹 함수 (이전과 동일)"""
        if not text: return []
        try: encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if not tokens: return []
        chunks = []
        current_pos = 0
        while current_pos < len(tokens):
            end_pos = min(current_pos + max_tokens, len(tokens))
            chunk_tokens = tokens[current_pos:end_pos]
            try: chunks.append(encoding.decode(chunk_tokens))
            except Exception as e: logger.error(f"Error decoding tokens: {e}")
            current_pos += (max_tokens - overlap)
            if current_pos >= end_pos: current_pos = end_pos
        return chunks


    @staticmethod
    def generate_faq_id(question: str) -> str:
        """질문 텍스트 기반으로 고유 ID 생성"""
        return f"faq_{hashlib.sha256(question.encode()).hexdigest()[:10]}"

    async def _index_vector_data(
        self,
        collection_name: str,
        docs_to_index: list[dict]
    ):
        """주어진 데이터를 배치 처리하여 Vector DB에 인덱싱"""
        logger.info(f"Starting vector indexing for collection '{collection_name}'...")
        indexed_count = 0
        for i in tqdm(range(0, len(docs_to_index), INGESTION_BATCH_SIZE), desc=f"Indexing {collection_name}"):
            batch = docs_to_index[i : i + INGESTION_BATCH_SIZE]
            batch_ids = [doc["id"] for doc in batch]
            batch_texts = [doc["text"] for doc in batch] # 임베딩 대상 텍스트
            batch_metadatas = [doc["metadata"] for doc in batch]
            try:
                batch_embeddings = await self.embedder.get_embeddings(batch_texts)
                if len(batch_embeddings) != len(batch_texts):
                    logger.error(f"Embedding count mismatch in batch {i // INGESTION_BATCH_SIZE} for {collection_name}.")
                    continue
                await self.vector_store.add_documents(
                    collection_name=collection_name,
                    ids=batch_ids,
                    documents=batch_texts, 
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                indexed_count += len(batch_ids)
            except Exception as e:
                logger.error(f"Error indexing batch {i // INGESTION_BATCH_SIZE} for {collection_name}: {e}")
        logger.info(f"Finished vector indexing for {collection_name}. Indexed {indexed_count} documents.")

    def _build_bm25_index(self, corpus: list[str]):
        """주어진 텍스트 코퍼스로 BM25 인덱스 구축"""
        logger.info(f"Building BM25 index for {len(corpus)} documents...")
        try:
            # 한국어 처리를 위해 간단한 공백 기반 토크나이저 사용 (개선 가능)
            # 또는 mecab, konlpy 등 형태소 분석기 사용 권장
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully.")
            return bm25
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            return None

    def _save_bm25(self, bm25_index, index_path: str, corpus_map: dict, corpus_path: str):
        """BM25 인덱스와 코퍼스 매핑 정보 저장"""
        try:
            # 인덱스 저장 경로 생성
            index_file_path = Path(index_path)
            index_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_file_path, 'wb') as f:
                pickle.dump(bm25_index, f)
            logger.info(f"BM25 index saved to {index_path}")

            # 코퍼스 매핑 정보 저장 (BM25 결과 인덱스를 원래 ID로 변환하기 위함)
            corpus_file_path = Path(corpus_path)
            corpus_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(corpus_file_path, 'wb') as f:
                pickle.dump(corpus_map, f)
            logger.info(f"BM25 corpus map saved to {corpus_path}")

        except Exception as e:
            logger.error(f"Error saving BM25 data: {e}")


    async def run_ingestion_pipeline(self, overwrite: bool = False):
        """
        전체 데이터 인덱싱 파이프라인 실행:
        1. 데이터 로드 및 전처리
        2. 질문 벡터 인덱싱
        3. Q&A 쌍 벡터 인덱싱
        4. 답변 청크 BM25 인덱싱
        """
        logger.info(f"--- Running Full Data Ingestion Pipeline (Overwrite={overwrite}) ---")

        # --- 컬렉션 준비 ---
        vector_collections = [QUESTION_COLLECTION_NAME, QNA_PAIR_COLLECTION_NAME]
        try:
            for col_name in vector_collections:
                exists = await self.vector_store.collection_exists(col_name)
                if exists:
                    if overwrite:
                        logger.warning(f"Collection '{col_name}' exists. Deleting for overwrite.")
                        await self.vector_store.delete_collection(col_name)
                        await self.vector_store.get_or_create_collection(
                            col_name, 
                            metadata={"hnsw:space": "cosine"}
                        )
                    else:
                        logger.info(f"Collection '{col_name}' exists and overwrite is False. Skipping vector ingestion for this collection.")
                        # 이미 존재하는 컬렉션은 건너뛰도록 플래그 설정 등 가능
                else:
                    await self.vector_store.get_or_create_collection(col_name)
            logger.info("Vector collections prepared.")
        except Exception as e:
            logger.critical(f"Failed to prepare vector collections: {e}")
            return

        # --- 데이터 로드 ---
        logger.info(f"Loading data from: {settings.FAQ_DATA_PATH}")
        try:
            data_dict = pd.read_pickle(settings.FAQ_DATA_PATH)
            if not isinstance(data_dict, dict): raise TypeError("Data is not a dictionary")
            logger.info(f"Loaded {len(data_dict)} Q&A entries.")
        except Exception as e:
            logger.critical(f"Failed to load data: {e}")
            return

        # --- 데이터 처리 및 인덱싱 목록 생성 ---
        logger.info("Processing data for all indices...")
        question_docs_to_index = []
        qna_pair_docs_to_index = []
        answer_chunks_for_bm25 = [] # BM25용 코퍼스 (텍스트만)
        bm25_corpus_map = {} # BM25 인덱스 -> (청크 ID, faq_id) 매핑용

        for raw_question, raw_answer in tqdm(data_dict.items(), total=len(data_dict), desc="Processing FAQs"):
            original_question = str(raw_question).strip()
            if not original_question: continue
            faq_id = self.generate_faq_id(original_question)

            # 1. 질문 데이터 준비
            question_docs_to_index.append({
                "id": faq_id, # 질문은 FAQ ID를 사용
                "text": original_question, # 임베딩 대상
                "metadata": {"source_faq_id": faq_id} # 메타데이터에 faq_id 저장
            })

            # 2. 답변 전처리 및 Q&A 쌍 데이터 준비
            cleaned_answer = self.preprocess_text(str(raw_answer))
            if not cleaned_answer: continue

            qna_pair_text = f"질문: {original_question}\n\n답변: {cleaned_answer}"
            qna_pair_docs_to_index.append({
                "id": faq_id, # Q&A 쌍도 FAQ ID 사용
                "text": qna_pair_text, # 임베딩 대상
                "metadata": {
                    "source_faq_id": faq_id,
                    "original_question": original_question,
                    "answered_text": cleaned_answer
                }
            })

            # 3. 답변 청킹 및 BM25 데이터 준비
            answer_chunks = self.chunk_text_with_token_limit(cleaned_answer)
            if not answer_chunks: continue

            for i, chunk_text in enumerate(answer_chunks):
                if not chunk_text.strip(): continue
                chunk_id = f"{faq_id}_chunk_{i}"
                # BM25 코퍼스 리스트에는 텍스트만 추가
                answer_chunks_for_bm25.append(chunk_text)
                # BM25 결과 인덱스를 원래 정보로 매핑하기 위한 딕셔너리
                # bm25_corpus_map의 key는 0부터 시작하는 정수 인덱스
                bm25_corpus_map[len(answer_chunks_for_bm25) - 1] = {
                    "chunk_id": chunk_id,
                    "source_faq_id": faq_id
                }

        logger.info(f"Prepared {len(question_docs_to_index)} questions for indexing.")
        logger.info(f"Prepared {len(qna_pair_docs_to_index)} Q&A pairs for indexing.")
        logger.info(f"Prepared {len(answer_chunks_for_bm25)} answer chunks for BM25.")

        # --- 벡터 인덱싱 실행 ---
        if question_docs_to_index:
            await self._index_vector_data(QUESTION_COLLECTION_NAME, question_docs_to_index)
        if qna_pair_docs_to_index:
            await self._index_vector_data(QNA_PAIR_COLLECTION_NAME, qna_pair_docs_to_index)

        # --- BM25 인덱스 구축 및 저장 ---
        if answer_chunks_for_bm25:
            bm25_index = self._build_bm25_index(answer_chunks_for_bm25)
            if bm25_index:
                self._save_bm25(bm25_index, BM25_INDEX_PATH, bm25_corpus_map, BM25_CORPUS_PATH)

        logger.info(f"--- Full Data Ingestion Pipeline Finished ---")


