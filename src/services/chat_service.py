# src/services/chat_service.py (리팩토링된 구조)
import logging
from typing import AsyncGenerator, List, Dict, Any, Tuple, Literal
from src.modules.retriever import Retriever
from src.modules.generator import ChatGenerator
from src.modules.guard import Guard
from src.modules.query_analyzer import QueryAnalyzer
# from src.core.memory import ConversationMemory
from src.core.config import settings

logger = logging.getLogger(__name__)

# 처리 경로 타입을 명시적으로 정의 (Literal 사용)
ChatFlowType = Literal["OFF_TOPIC", "GENERATE_DIRECT", "GENERATE_STANDARD", "HANDLE_ERROR"]

class ChatService:
    def __init__(self, retriever: Retriever, generator: ChatGenerator, guard: Guard, query_analyzer: QueryAnalyzer):
        self.retriever = retriever
        self.generator = generator
        self.guard = guard
        self.query_analyzer = query_analyzer
        # self.memory = memory
        logger.info("ChatService initialized.")

    async def _load_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """대화 기록 로드 및 포맷팅"""
        logger.debug(f"[{conversation_id}] Loading conversation history...")
        # history_messages = await self.memory.get_history(conversation_id, k=settings.CONTEXT_TURNS)
        # return [msg.model_dump() for msg in history_messages]
        return []

    async def _generate_response_tokens( # 이름 변경: 스트리밍 책임 명확화
        self,
        conversation_id: str,
        flow_type: ChatFlowType,
        user_query: str,
        history: List[Dict[str, str]],
        retrieved_contexts: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]: # 이제 순수 토큰만 yield
        """결정된 처리 유형에 따라 응답 토큰 스트림 생성"""
        logger.debug(f"[{conversation_id}] Generating response tokens for flow type: {flow_type}")

        if flow_type == "OFF_TOPIC":
            async for token in self.generator.stream_off_topic_response(user_query):
                yield token
        elif flow_type in ["GENERATE_DIRECT", "GENERATE_STANDARD"]:
            prompt_type = "MAIN_ANSWER_DIRECT" if flow_type == "GENERATE_DIRECT" else "MAIN_ANSWER_STANDARD"
            main_answer_tokens = [] # 메인 답변 토큰 임시 저장

            # 1. 메인 답변 스트리밍 및 토큰 수집
            async for token, _ in self.generator.stream_main_answer( # 누적 답변은 사용 안 함
                user_query=user_query,
                history=history,
                retrieved_contexts=retrieved_contexts,
                prompt_type=prompt_type
            ):
                main_answer_tokens.append(token)
                yield token # 토큰 즉시 전달

            # 2. 후속 질문 생성 (메인 답변 완료 후)
            if main_answer_tokens:
                main_answer_full = "".join(main_answer_tokens)
                follow_ups = await self.generator.generate_follow_up_questions(
                    user_query=user_query,
                    retrieved_contexts=retrieved_contexts,
                    main_answer=main_answer_full
                )
                if follow_ups:
                    follow_up_header = "\n\n--- 다음 질문 제안 ---\n"
                    yield follow_up_header
                    for fu in follow_ups:
                        follow_up_line = f"- {fu}\n"
                        yield follow_up_line
        else:
            logger.error(f"[{conversation_id}] Unhandled flow type: {flow_type}")
            yield "[오류: 처리 중 문제가 발생했습니다.]"

    async def save_message(self, conversation_id: str, role: str, content: str):
         """메시지 저장 메서드 분리"""
        #  await self.memory.add_message(conversation_id, Message(role=role, content=content))

    # --- Public Method ---
    async def process_chat_stream(
        self, user_query: str, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        """
        채팅 파이프라인 실행 및 응답 토큰 스트리밍.
        대화 저장은 이 메서드 호출 이후 별도 수행 필요 가능성.
        """
        try:
            history = await self._load_history(conversation_id)
            flow_type, contexts, top1_sim, sim_gap = await self.query_analyzer.analyze_query(conversation_id, user_query)

            async for token in self._generate_response_tokens(
                conversation_id, flow_type, user_query, history, contexts
            ):
                yield token

        except Exception as e:
            logger.exception(f"[{conversation_id}] Error during chat processing stream: {e}")
            yield "[오류: 죄송합니다. 예상치 못한 문제가 발생했습니다.]"

