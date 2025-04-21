# src/chatbot_api/modules/generator.py
import logging
import json
from typing import List, Dict, AsyncGenerator, Tuple, Optional, Any

from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError 
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from ..prompts.templates import build_prompt_messages 
from ..core.config import settings

logger = logging.getLogger(__name__)

# 재시도 설정
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, OpenAIError)
retry_decorator = retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI ChatCompletion call due to {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
)

class ChatGenerator:
    def __init__(self, api_key: str, model: str):
        # 클라이언트 초기화는 의존성 주입 시 처리되므로 여기서 AuthenticationError 처리는 불필요
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens_main = settings.MAX_GENERATION_TOKENS_MAIN
        self.max_tokens_followup = settings.MAX_GENERATION_TOKENS_FOLLOWUP
        self.temperature = settings.GENERATOR_TEMPERATURE # 생성 온도 설정 추가

    @retry_decorator
    async def _stream_llm_response(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float # 온도 파라미터 추가
    ) -> AsyncGenerator[str, None]:
        """LLM 스트리밍 호출 내부 로직 (재시도 포함)"""
        logger.info(f"Calling OpenAI ChatCompletion with model: {self.model}, temp: {temperature}")
        # logger.debug(f"Messages: {messages}")
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature, # 온도 적용
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            logger.info("OpenAI stream finished.")
        except OpenAIError as e: # 구체적인 에러 타입 로깅
             logger.error(f"OpenAI API Error during stream call (type: {type(e).__name__}): {e}")
             yield "[오류: 답변 생성 중 API 문제가 발생했습니다.]"
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI stream call: {e}")
            yield "[오류: 답변 생성 중 예상치 못한 문제가 발생했습니다.]"

    async def stream_main_answer(
        self,
        user_query: str,
        history: List[Dict[str, str]],
        retrieved_contexts: List[Dict[str, Any]],
        prompt_type: str 
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """메인 답변을 스트리밍으로 생성하고, 전체 답변을 반환 (prompt_type 사용)"""
        logger.info(f"Streaming main answer using prompt type: {prompt_type}")
        try:

            messages = build_prompt_messages(
                prompt_type=prompt_type, # 전달받은 값 사용
                user_query=user_query,
                history=history,
                retrieved_contexts=retrieved_contexts
            )

        except ValueError as e: # build_prompt_messages 에서 잘못된 타입 에러 처리
             logger.error(f"Error building prompt for type '{prompt_type}': {e}")
             yield "[오류: 답변 생성 준비 중 문제가 발생했습니다.]", "[오류: 답변 생성 준비 중 문제가 발생했습니다.]"
             return

        full_answer = ""
        # 메인 답변 생성 시 설정된 온도 사용
        async for token in self._stream_llm_response(messages, self.max_tokens_main, self.temperature):
            full_answer += token
            yield token, full_answer # 토큰과 누적 답변 동시 반환

    async def generate_follow_up_questions(
        self,
        user_query: str,
        retrieved_contexts: List[Dict[str, Any]],
        main_answer: str
    ) -> List[str]:
        """컨텍스트와 메인 답변 기반으로 후속 질문 생성 (JSON 파싱)"""
        logger.info("Generating follow-up questions...")
        try:
            messages = build_prompt_messages(
                prompt_type="FOLLOW_UP",
                user_query=user_query,
                history=[],
                retrieved_contexts=retrieved_contexts,
                main_answer=main_answer
            )
        except ValueError as e:
             logger.error(f"Error building prompt for follow-up: {e}")
             return []

        full_response = ""
        try:
            # 후속 질문 생성 시에는 온도를 약간 낮춰 더 예측 가능하게
            response = await self.client.chat.completions.create(
                 model=self.model, # 필요시 더 저렴한 모델 (e.g., settings.FOLLOWUP_MODEL) 사용
                 messages=messages,
                 max_tokens=self.max_tokens_followup,
                 temperature=max(0.0, self.temperature - 0.2), # 메인 답변 온도보다 약간 낮게
                 response_format={"type": "json_object"}
            )
            full_response = response.choices[0].message.content or ""
            logger.debug(f"Raw follow-up response: {full_response}")

            # JSON 파싱
            try:
                parsed_json = json.loads(full_response)
                follow_ups = parsed_json.get("follow_ups", [])
                if isinstance(follow_ups, list) and all(isinstance(q, str) for q in follow_ups):
                    logger.info(f"Successfully generated {len(follow_ups)} follow-up questions.")
                    return follow_ups[:2] # 최대 2개
                else:
                    logger.warning(f"Follow-up response JSON format is invalid: {follow_ups}")
                    return []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse follow-up response as JSON: {full_response}")
                return []
        except OpenAIError as e:
             logger.error(f"OpenAI API Error generating follow-up questions (type: {type(e).__name__}): {e}")
             return []
        except Exception as e:
            logger.error(f"Unexpected error generating follow-up questions: {e}")
            return []

    async def stream_off_topic_response(
        self, user_query: str
    ) -> AsyncGenerator[str, None]:
        """Off-topic 안내 메시지를 스트리밍으로 생성"""
        logger.info("Streaming off-topic response...")
        try:
            messages = build_prompt_messages(
                prompt_type="OFF_TOPIC",
                user_query=user_query,
                history=[],
                retrieved_contexts=[]
            )
        except ValueError as e:
             logger.error(f"Error building prompt for off-topic: {e}")
             yield "[오류: 응답 생성 준비 중 문제가 발생했습니다.]"
             return

        # Off-topic 응답은 낮은 온도로, 적은 토큰으로
        async for token in self._stream_llm_response(messages, max_tokens=150, temperature=0.1):
            yield token