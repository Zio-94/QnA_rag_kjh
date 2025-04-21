from typing import Tuple, List, Dict, Any
from src.core.config import settings
from src.modules.guard import Guard
from src.modules.retriever import Retriever
from src.core.config import settings
from src.schemas.chat import ChatFlowType
import logging

logger = logging.getLogger(__name__)
class QueryAnalyzer:
    def __init__(self, guard: Guard, retriever: Retriever):
        self.guard = guard
        self.retriever = retriever

    async def analyze_query(
        self, conversation_id: str, user_query: str
    ) -> Tuple[ChatFlowType, List[Dict[str, Any]], float, float]:
        """
        쿼리 분석 및 검색을 통해 다음 처리 흐름과 필요한 정보를 결정합니다.
        반환값: (처리 유형, 검색된 컨텍스트, top1 유사도, 유사도 gap)
        """
        logger.debug(f"[{conversation_id}] Determining chat flow for query: {user_query[:50]}...")

        # 1. Guard: LLM 도메인 분류
        # is_relevant_llm = await self.guard.is_query_domain_relevant(user_query)
        # if not is_relevant_llm:
        #     logger.info(f"[{conversation_id}] Query classified as OFF_TOPIC by LLM Guard.")
        #     return "OFF_TOPIC", [], 0.0, 0.0 # 컨텍스트 없음, 유사도 0

        # 2. Retriever: 검색 수행
        # retrieve 메서드가 컨텍스트 목록을 반환한다고 가정
        retrieved_contexts = await self.retriever.retrieve(user_query)

        # 3. 유사도 정보 추출
        top1_sim, sim_gap = 0.0, 0.0
        if retrieved_contexts and 'debug_info' in retrieved_contexts[0]:
            debug_info = retrieved_contexts[0]['debug_info']
            top1_sim = debug_info.get('top1_sim_qna', 0.0)
            sim_gap = debug_info.get('gap_qna', 0.0)
        logger.info(f"[{conversation_id}] Retrieval results - Top1 Sim: {top1_sim:.4f}, Sim Gap: {sim_gap:.4f}")

        # 4. Tiered Logic으로 처리 유형 결정
        if not retrieved_contexts:
            logger.warning(f"[{conversation_id}] No relevant contexts found. Treating as marginal (STANDARD).")
            return "GENERATE_STANDARD", [], top1_sim, sim_gap # 컨텍스트 없지만 표준 RAG 시도
        elif top1_sim >= settings.HIGH_THRESHOLD and sim_gap >= settings.LARGE_GAP_THRESHOLD:
            logger.info(f"[{conversation_id}] High confidence match. Flow: GENERATE_DIRECT.")
            return "GENERATE_DIRECT", retrieved_contexts, top1_sim, sim_gap
        elif top1_sim >= settings.MEDIUM_THRESHOLD:
            logger.info(f"[{conversation_id}] Sufficient match. Flow: GENERATE_STANDARD.")
            return "GENERATE_STANDARD", retrieved_contexts, top1_sim, sim_gap
        else:
            logger.warning(f"[{conversation_id}] Low similarity score. Flow: OFF_TOPIC.")
            return "OFF_TOPIC", [], top1_sim, sim_gap # 유사도 낮으면 Off-topic
