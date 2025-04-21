# src/modules/guard.py
import logging
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from ..core.config import settings
# 프롬프트 템플릿 필요 시 임포트 (여기서는 간단히 처리)

logger = logging.getLogger(__name__)

# LLM 호출 재시도 설정 (Generator와 유사하게)
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, OpenAIError)
retry_decorator = retry(
    wait=wait_random_exponential(min=1, max=10), # 분류기는 더 짧게 재시도
    stop=stop_after_attempt(2), # 최대 2번 시도
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logger.warning(f"Retrying LLM domain classification due to {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
)

class Guard:
    """
    사용자 쿼리의 도메인 관련성 및 잠재적 위험성을 판단하는 모듈.
    LLM 기반 Zero-shot 분류기를 포함할 수 있음.
    """
    def __init__(self, api_key: str):
        # LLM 분류기용 클라이언트 (별도 모델 사용 가능)
        # 필요시 settings에서 분류기 모델명 가져오기
        self.classifier_model = settings.GUARD_CLASSIFIER_MODEL # 설정 추가 필요 (예: "gpt-3.5-turbo")
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(f"Guard initialized with classifier model: {self.classifier_model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for Guard: {e}")
            # 클라이언트 초기화 실패 시 LLM 분류 기능 비활성화 또는 에러 발생
            self.client = None

        self.domain_name = "네이버 스마트스토어 FAQ" # 도메인 이름 설정

    @retry_decorator
    async def _call_llm_classifier(self, query: str) -> Optional[bool]:
        """LLM을 호출하여 쿼리의 도메인 관련성 판단 (Yes/No)"""
        if not self.client:
            logger.warning("Guard LLM client not initialized. Skipping LLM classification.")
            return None # LLM 분류 불가 시 None 반환

        # Zero-shot 프롬프트
        system_prompt = f"당신은 사용자 질문이 '{self.domain_name}'와(과) 관련된 내용인지 판단하는 분류기입니다."
        user_prompt = f"""다음 사용자 질문이 '네이버 스마트스토어 판매자 지원 FAQ'와 관련된 질문인지 판단해주세요.
        '네이버 스마트스토어 판매자 지원 FAQ'는 스마트스토어 개설, 상품 등록, 판매 관리, 정산, 배송, 광고, 고객 문의 처리 등 판매 활동 전반에 대한 내용을 다룹니다.

        <판단 기준 예시>
        질문: "스마트스토어 입점 절차가 궁금해요." -> Yes
        질문: "상품 상세페이지는 어떻게 꾸미나요?" -> Yes
        질문: "오늘 날씨 어때?" -> No
        질문: "강남역 맛집 추천해줘." -> No
        질문: "네이버 아이디 비밀번호를 잊어버렸어요." -> No (스마트스토어 직접 관련 문의 아님)

        <사용자 질문>
        "{query}"

        판단 (Yes 또는 No):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.classifier_model,
                messages=messages,
                max_tokens=5, # "Yes" or "No" 만 필요
                temperature=0.0, # 확정적 결과 유도
                stream=False # 스트리밍 불필요
            )
            result_text = response.choices[0].message.content.strip().lower()
            logger.debug(f"LLM domain classification for '{query[:30]}...': Raw='{result_text}'")

            if "yes" in result_text:
                return True
            elif "no" in result_text:
                return False
            else:
                logger.warning(f"LLM classifier returned ambiguous result: {result_text}")
                return None # 애매한 경우 None 반환

        except Exception as e:
            logger.error(f"Error calling LLM classifier: {e}")
            return None # 오류 시 None 반환

    async def is_query_domain_relevant(self, query: str) -> bool:
        """
        LLM 분류기를 호출하여 쿼리가 도메인 관련성이 있는지 최종 판단.
        LLM 호출 실패 시 기본값(예: True) 반환 또는 에러 발생 선택 가능.
        """
        llm_result = await self._call_llm_classifier(query)

        if llm_result is None:
            # LLM 판단 실패 시 처리: 보수적으로 관련 없다고 보거나(False),
            # 일단 관련 있다고 보고(True) 검색 결과에 의존할 수 있음.
            logger.warning("LLM domain classification failed or ambiguous. Assuming relevant for now.")
            return True # 여기서는 일단 True 반환 (검색 결과에 판단 위임)
        else:
            logger.info(f"LLM classified query '{query[:30]}...' as domain relevant: {llm_result}")
            return llm_result

    # TODO: 향후 OpenAI Moderation API 등을 이용한 유해성 검사 기능 추가 가능
    # async def is_query_safe(self, query: str) -> bool: ...