# src/modules/retriever.py
import logging
import asyncio
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# BM25 라이브러리
from rank_bm25 import BM25Okapi

# 내부 모듈 임포트
from .vector_store import VectorStore
from .embedder import Embedder
# from .reranker import ReRanker # 최종 결과에 Re-ranker 추가 가능 (선택적)
from ..core.config import settings

logger = logging.getLogger(__name__)

# --- RRF 구현 함수 ---
def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]], # [(id, score), ...] 리스트의 리스트
    k: int = 60 # RRF 랭크 가중치 파라미터 (조정 가능)
) -> Dict[str, float]:
    """
    여러 검색 결과 목록을 RRF로 융합합니다.
    :param results_list: 각 검색 시스템의 결과. 각 결과는 (문서 ID, 점수) 튜플 리스트.
                         점수는 높을수록 좋다고 가정합니다. (낮을수록 좋으면 변환 필요)
    :param k: RRF의 랭크 보정 파라미터.
    :return: {문서 ID: 최종 RRF 점수} 딕셔너리. 점수가 높을수록 좋음.
    """
    if not results_list:
        return {}

    # 각 문서 ID의 RRF 점수를 저장할 딕셔너리
    rrf_scores: Dict[str, float] = {}

    # 각 검색 결과 리스트 순회
    for results in results_list:
        if not results: continue
        # 각 검색 결과 내 순위 (rank) 계산 (1부터 시작)
        for rank, (doc_id, score) in enumerate(results, 1):
            # RRF 점수 계산: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            # 기존 점수에 합산 (동일 ID가 여러 리스트에 나타날 수 있음)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

    return rrf_scores

# --- Retriever 클래스 ---
class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        # reranker: Optional[ReRanker] = None, # 필요시 추가
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        # self.reranker = reranker if settings.USE_RERANKER else None

        # 컬렉션 이름 로드
        self.question_collection = settings.QUESTION_COLLECTION_NAME
        self.qna_pair_collection = settings.QNA_PAIR_COLLECTION_NAME

        # BM25 인덱스 및 코퍼스 맵 로드
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus_map: Optional[Dict[int, Dict[str, str]]] = None
        self._load_bm25_data()

        # 검색 파라미터 로드
        self.top_k_vector_search = settings.RETRIEVER_TOP_K_VECTOR # 설정 추가 (예: 20)
        self.top_k_bm25_search = settings.RETRIEVER_TOP_K_BM25  # 설정 추가 (예: 50)
        self.top_n_final_faqs = settings.RETRIEVER_TOP_N_FINAL_FAQS # 설정 추가 (예: 5)
        self.rrf_k = settings.RRF_K_VALUE # 설정 추가 (예: 60)

        # BM25 토크나이저 (Ingestion과 동일하게 사용)
        # TODO: 한국어 형태소 분석기 적용 권장
        self.bm25_tokenizer = lambda doc: doc.split()

    def _load_bm25_data(self):
        """BM25 인덱스와 코퍼스 맵을 파일에서 로드합니다."""
        try:
            bm25_index_path = Path(settings.BM25_INDEX_PATH)
            bm25_corpus_path = Path(settings.BM25_CORPUS_PATH)

            if bm25_index_path.exists() and bm25_corpus_path.exists():
                with open(bm25_index_path, 'rb') as f_idx:
                    self.bm25_index = pickle.load(f_idx)
                with open(bm25_corpus_path, 'rb') as f_map:
                    self.bm25_corpus_map = pickle.load(f_map)
                logger.info(f"BM25 index and corpus map loaded successfully from {settings.BM25_INDEX_PATH} and {settings.BM25_CORPUS_PATH}")
            else:
                logger.warning("BM25 index or corpus map file not found. BM25 search will be disabled.")
        except Exception as e:
            logger.error(f"Error loading BM25 data: {e}")
            self.bm25_index = None
            self.bm25_corpus_map = None

    async def _search_questions(self, query_embedding: List[float]) -> List[Tuple[str, float]]:
        """질문 벡터 컬렉션 검색"""
        logger.debug("Searching Question collection...")
        results = await self.vector_store.search_with_scores(
            collection_name=self.question_collection,
            query_embedding=query_embedding,
            k=self.top_k_vector_search
        )
        # 결과 형식: (faq_id, score) - score는 유사도 (1 - distance) 로 변환 가정
        # ChromaDB는 distance 반환 -> 유사도로 변환 필요
        question_results = []
        for res in results:
            faq_id = res.get('metadata', {}).get('source_faq_id')
            distance = res.get('distance')
            if faq_id and distance is not None:
                 # 간단한 유사도 변환 (0~1 범위 가정, 필요시 조정)
                 similarity = max(0.0, 1.0 - distance)
                 question_results.append((faq_id, similarity))
        logger.debug(f"Question search returned {len(question_results)} results.")
        return question_results

    async def _search_qna_pairs(self, query_embedding: List[float]) -> List[Tuple[str, float]]:
        """Q&A 쌍 벡터 컬렉션 검색"""
        logger.debug("Searching QnA Pair collection...")
        results = await self.vector_store.search_with_scores(
            collection_name=self.qna_pair_collection,
            query_embedding=query_embedding,
            k=self.top_k_vector_search
        )
        qna_results = []
        for res in results:
            faq_id = res.get('metadata', {}).get('source_faq_id')
            distance = res.get('distance')
            if faq_id and distance is not None:
                 similarity = max(0.0, 1.0 - distance)
                 qna_results.append((faq_id, similarity))
        logger.debug(f"QnA Pair search returned {len(qna_results)} results.")
        return qna_results

    def _search_bm25(self, query: str) -> List[Tuple[str, float]]:
        """BM25 인덱스 검색 (답변 청크 기반)"""
        if not self.bm25_index or not self.bm25_corpus_map:
            logger.warning("BM25 index not available. Skipping BM25 search.")
            return []

        logger.debug("Searching BM25 index...")
        tokenized_query = self.bm25_tokenizer(query)
        # BM25 점수 계산 (상위 K개)
        # get_top_n은 내부적으로 모든 문서 점수 계산 후 정렬하므로 비효율적일 수 있음
        # 대안: get_scores 사용 후 상위 K개 선택
        try:
            # get_scores는 모든 코퍼스에 대한 점수 반환
            scores = self.bm25_index.get_scores(tokenized_query)
            # 점수와 인덱스 결합 후 상위 K개 선택
            indexed_scores = list(enumerate(scores)) # [(corpus_idx, score), ...]
            # 점수 내림차순 정렬
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            # 상위 K개 선택
            top_k_bm25_results = indexed_scores[:self.top_k_bm25_search]

            bm25_results = []
            # 결과를 (faq_id, score) 로 변환하고, faq_id별 최고 점수만 유지
            faq_scores: Dict[str, float] = {}
            for corpus_idx, score in top_k_bm25_results:
                 if score <= 0: continue # 관련 없는 결과 제외 (BM25 점수는 음수 가능)
                 map_info = self.bm25_corpus_map.get(corpus_idx)
                 if map_info:
                      faq_id = map_info.get("source_faq_id")
                      if faq_id:
                           # 해당 faq_id의 현재 최고 점수보다 높으면 업데이트
                           if score > faq_scores.get(faq_id, -1.0):
                                faq_scores[faq_id] = score

            # 최종 (faq_id, score) 리스트 생성 (점수 높은 순)
            bm25_results = sorted(faq_scores.items(), key=lambda item: item[1], reverse=True)

            logger.debug(f"BM25 search returned {len(bm25_results)} unique FAQ IDs.")
            return bm25_results
        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []

    async def _get_context_for_faqs(self, faq_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        선택된 FAQ ID 목록을 사용하여 Q&A 쌍 컬렉션에서
        원본 질문과 답변 텍스트(메타데이터에 저장됨)를 조회합니다. 
        """
        if not faq_ids:
            return {}
        logger.info(f"Fetching context (question & answered_text) for {len(faq_ids)} FAQs from '{self.qna_pair_collection}'...")
        context_map: Dict[str, Dict[str, Any]] = {}
        try:
            # VectorStore에서 ID 목록으로 메타데이터 조회 (include 사용)
            # 반환 타입이 List[Dict[str, Any]] 또는 None 이라고 가정
            retrieved_docs: Optional[List[Dict[str, Any]]] = await self.vector_store.get_documents_by_ids(
                collection_name=self.qna_pair_collection,
                ids=faq_ids,
                include=["metadatas"] # 메타데이터만 필요함을 명시
            )

            # 결과가 None이거나 비어있는 리스트일 경우 처리
            if not retrieved_docs:
                 logger.warning(f"Could not retrieve any documents for the provided faq_ids: {faq_ids}")
                 return {} # 빈 딕셔너리 반환

            # 이제 retrieved_docs는 List[Dict] 형태임
            found_ids = set()
            for doc_data in retrieved_docs:
                 # 각 딕셔너리에서 'id'와 'metadata' 추출 시도
                 faq_id = doc_data.get('id')
                 metadata = doc_data.get('metadata')

                 if not faq_id:
                     logger.warning(f"Retrieved document data is missing 'id': {doc_data}")
                     continue # ID 없으면 처리 불가

                 found_ids.add(faq_id) # 찾은 ID 기록

                 # 메타데이터 유효성 검사 및 필요한 정보 추출
                 if metadata and isinstance(metadata, dict):
                     original_question = metadata.get('original_question')
                     answered_text = metadata.get('answered_text') # 메타데이터에서 답변 추출

                     if original_question and answered_text:
                          context_map[faq_id] = {
                               "original_question": original_question,
                               "answered_text": answered_text # 전체 답변 텍스트
                          }
                     else:
                          # 메타데이터는 있지만 필요한 키가 없는 경우
                          missing_keys = []
                          if not original_question: missing_keys.append("'original_question'")
                          if not answered_text: missing_keys.append("'answered_text'")
                          logger.warning(f"Metadata for faq_id '{faq_id}' is missing required keys: {', '.join(missing_keys)}")
                 else:
                     # 메타데이터 자체가 없거나 형식이 잘못된 경우
                     logger.warning(f"Missing or invalid metadata for faq_id: '{faq_id}'. Found: {metadata}")

            # 요청한 ID 중 일부를 찾지 못했을 경우 로그 (선택 사항)
            missing_ids = set(faq_ids) - found_ids
            if missing_ids:
                logger.warning(f"Could not find metadata for some requested FAQ IDs: {missing_ids}")

        except Exception as e:
            # Vector Store 통신 오류 등 예외 처리
            logger.exception(f"Error fetching or processing context for FAQs: {e}") # exception으로 스택 트레이스 포함
            context_map = {} # 오류 발생 시 안전하게 빈 딕셔너리 반환

        logger.info(f"Context fetched for {len(context_map)} out of {len(faq_ids)} requested FAQs.")
        return context_map

    async def retrieve_and_fuse(self, query: str) -> Tuple[List[Tuple[str, float]], float, float]:
        """
        3가지 검색을 병렬 수행하고 RRF로 융합하여 최종 FAQ ID 목록/점수와 함께,
        가장 신뢰도 높은 검색(예: QnA 쌍 검색)의 Top-1 및 Top-2 유사도(또는 거리)를 반환합니다.

        :return: ([(faq_id, rrf_score), ...], top1_similarity, top2_similarity_or_distance)
                 유사도 값은 0~1 사이, 높을수록 좋음. (거리 값 그대로 반환할 수도 있음)
        """
        logger.info(f"Starting retrieve_and_fuse for query: {query[:50]}...")
        query_embedding = await self.embedder.get_embedding(query)

        # 1. 병렬 검색 수행
        search_tasks = [
            self._search_questions(query_embedding),
            self._search_qna_pairs(query_embedding), # 이 결과를 Top-1/2 기준으로 삼는다고 가정
            asyncio.to_thread(self._search_bm25, query)
        ]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 결과 처리 및 Top-1/2 유사도 추출
        question_results = results[0] if not isinstance(results[0], Exception) else []
        qna_pair_results = results[1] if not isinstance(results[1], Exception) else []
        bm25_results = results[2] if not isinstance(results[2], Exception) else []

        # Top-1/Top-2 유사도 추출 (QnA 쌍 검색 결과 기준)
        top1_similarity = 0.0
        top2_similarity = 0.0
        if qna_pair_results:
            # qna_pair_results는 (faq_id, similarity) 튜플 리스트 (점수 높은 순)
            if len(qna_pair_results) > 0:
                top1_similarity = qna_pair_results[0][1]
            if len(qna_pair_results) > 1:
                top2_similarity = qna_pair_results[1][1]
        logger.info(f"Top QnA Pair Similarities: Top1={top1_similarity:.4f}, Top2={top2_similarity:.4f}")


        # 오류 로깅 (이전과 동일)
        if isinstance(results[0], Exception): logger.error(f"Question search failed: {results[0]}")
        if isinstance(results[1], Exception): logger.error(f"QnA Pair search failed: {results[1]}")
        if isinstance(results[2], Exception): logger.error(f"BM25 search failed: {results[2]}")

        # 2. RRF 융합
        all_results = [question_results, qna_pair_results, bm25_results]
        valid_results = [res for res in all_results if res]
        if not valid_results:
            logger.warning("All search methods failed or returned no results.")
            # RRF 결과는 비어있고, 유사도는 0으로 반환
            return [], 0.0, 0.0

        logger.info(f"Applying RRF fusion to {len(valid_results)} result lists...")
        fused_scores = reciprocal_rank_fusion(valid_results, k=self.rrf_k)
        final_ranked_faqs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        logger.info(f"Fusion resulted in {len(final_ranked_faqs)} ranked FAQs.")

        # 최종 순위 목록 (상위 N개)과 추출된 Top-1/Top-2 유사도 반환
        final_list = final_ranked_faqs[:self.top_n_final_faqs]


        return final_list, top1_similarity, top2_similarity
    
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        최종 사용자 대상 메서드: 쿼리를 받아 검색/융합 후
        선택된 FAQ에 대한 컨텍스트(원본 질문, 원본 답변 텍스트)를 반환합니다.
        """
        # 1. 검색 및 융합하여 최종 FAQ ID 목록과 점수, 유사도 정보 얻기
        ranked_faq_ids_with_scores, top1_sim, top2_sim = await self.retrieve_and_fuse(query)
        if not ranked_faq_ids_with_scores:
             logger.warning(f"Retrieve_and_fuse returned no ranked FAQs for query: {query[:50]}...")
             return []

        final_faq_ids = [faq_id for faq_id, score in ranked_faq_ids_with_scores]

        # 2. 선택된 FAQ ID에 대한 컨텍스트 조회 
        context_map = await self._get_context_for_faqs(final_faq_ids)


        final_context_list = []
        for faq_id, rrf_score in ranked_faq_ids_with_scores:
             if faq_id in context_map:
                  # context_map에서 해당 FAQ의 정보 가져오기
                  faq_context = context_map[faq_id]
                  # 최종 반환 리스트에 필요한 정보만 담아서 추가
                  context_entry = {
                      "source_faq_id": faq_id,
                      "original_question": faq_context.get("original_question", "N/A"),
                      "answered_text": faq_context.get("answered_text", "N/A"), # 전체 답변 텍스트
                      "rrf_score": rrf_score,
                      # 디버그 정보는 첫 번째 항목에만 추가 (선택적)
                      "debug_info": {}
                  }
                  if not final_context_list: # 첫 번째 항목일 경우
                       context_entry["debug_info"] = {
                           'top1_sim_qna': top1_sim,
                           'top2_sim_qna': top2_sim,
                           'gap_qna': top1_sim - top2_sim
                       }
                  final_context_list.append(context_entry)
             else:
                  # 컨텍스트 조회 실패 시 로그만 남기고 해당 FAQ는 결과에서 제외
                  logger.warning(f"Context could not be fetched for ranked FAQ ID: {faq_id}. Skipping.")


        logger.info(f"Returning {len(final_context_list)} final contexts for LLM.")
        return final_context_list
        # 1. 검색 및 융합 (이제 유사도 정보도 함께 받음)
        ranked_faq_ids_with_scores, top1_sim, top2_sim = await self.retrieve_and_fuse(query) # 반환값 받기
        if not ranked_faq_ids_with_scores:
             return []

        final_faq_ids = [faq_id for faq_id, score in ranked_faq_ids_with_scores]

        # 2. 선택된 FAQ ID에 대한 컨텍스트 조회
        context_map = await self._get_context_for_faqs(final_faq_ids)

        # 3. 최종 결과 조합 (Top-1/Top-2 유사도 정보 추가)
        final_context_list = []
        for faq_id, score in ranked_faq_ids_with_scores:
             if faq_id in context_map:
                  context = context_map[faq_id]
                  context['rrf_score'] = score
                  context['source_faq_id'] = faq_id
                  if not final_context_list:
                       context['debug_similarity'] = {'top1': top1_sim, 'top2': top2_sim, 'gap': top1_sim - top2_sim}
                  # --- <<< 추가 완료 <<< ---
                  final_context_list.append(context)
             else:
                  logger.warning(f"Context not found for ranked FAQ ID: {faq_id}")

        logger.info(f"Returning {len(final_context_list)} contexts for LLM.")
        return final_context_list