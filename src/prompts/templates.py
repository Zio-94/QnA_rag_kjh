# src/chatbot_api/prompts/templates.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- System Prompts ---

BASE_SYSTEM_PROMPT = """당신은 네이버 스마트스토어의 FAQ 내용을 기반으로 답변하는 친절하고 정확한 챗봇 도우미입니다.
사용자의 질문에 대해 아래 제공된 <FAQ_CONTEXT>를 최우선으로 참고하여 답변해야 합니다.
만약 컨텍스트에 질문과 관련된 내용이 없다면, "제공된 정보만으로는 정확한 답변을 드리기 어렵습니다." 라고 솔직하게 답변하세요.
답변은 명확하고 간결하게 한국어로 작성해주세요."""


MAIN_ANSWER_STANDARD_SYSTEM_ADDENDUM = """
<FAQ_CONTEXT>
{faq_context}
</FAQ_CONTEXT>

위 컨텍스트를 바탕으로 사용자의 질문 "{user_query}"에 대해 답변해주세요. 여러 문서 내용이 있다면 종합적으로 고려하여 자연스럽게 설명해주세요.
답변은 최대 {max_answer_length}자 내외로 해주세요.
"""



MAIN_ANSWER_DIRECT_SYSTEM_ADDENDUM = """
<FAQ_CONTEXT>
{faq_context}
</FAQ_CONTEXT>

사용자의 질문 "{user_query}"과 매우 관련성이 높은 정보가 위 컨텍스트에 포함되어 있습니다.
컨텍스트 내용을 바탕으로 사용자의 질문에 **직접적이고 명확하게** 답변해주세요. 추측하거나 불확실한 표현은 삼가세요.
답변은 최대 {max_answer_length}자 내외로 해주세요.
"""

HISTORY_ANSWER_DIRECT_SYSTEM_ADDENDUM = """
<FAQ_CONTEXT>
{faq_context}
</FAQ_CONTEXT>

<HISTORY>
{history}
</HISTORY>

사용자의 마지막 질문 "{user_query}"에 대해 답변해야 합니다.
**중요:** 만약 사용자의 마지막 질문이 <HISTORY>의 내용과 직접적으로 이어진다면, <FAQ_CONTEXT>보다 <HISTORY>을 우선적으로 고려하여 답변하세요. 예를 들어, 이전 답변에 대한 설명을 요구하거나 관련된 추가 질문을 하는 경우가 해당됩니다.
만약 사용자의 마지막 질문이 새로운 주제이거나 <HISTORY>과 관련성이 낮다면, <FAQ_CONTEXT>를 최우선으로 참고하여 답변하세요.
컨텍스트에 관련 내용이 없다면 "제공된 정보만으로는 정확한 답변을 드리기 어렵습니다."라고 답변하세요.
답변은 최대 {max_answer_length}자 내외로 해주세요.
"""

FOLLOW_UP_SYSTEM_ADDENDUM = """
<FAQ_CONTEXT>
{faq_context}
</FAQ_CONTEXT>
<USER_QUERY>{user_query}</USER_QUERY>
<MAIN_ANSWER>{main_answer}</MAIN_ANSWER>

위 대화 내용을 바탕으로, 사용자가 다음에 궁금해할 만한 **후속 질문 2개**를 생성해주세요.

반드시 다음 조건을 지켜주세요:

1. 후속 질문은 FAQ_CONTEXT 안에 실제 존재하는 질문들만 조합하거나 의미적으로 변형한 형태여야 합니다.
2. 질문은 사용자의 원래 질문 의도(관심 주제 흐름)를 유지하도록 자연스럽게 연결되어야 하며, 단순 키워드 일치는 피해주세요.
3. 질문이 완전히 동떨어진 느낌을 주지 않도록, **의도 기반 유사도와 FAQ 내 존재 여부를 동시에 고려**해야 합니다.

정답은 반드시 다음 JSON 형식 배열로 반환하세요:  
{{"follow_ups": ["관련 질문 1?", "관련 질문 2?"]}}

JSON 외의 텍스트는 절대 포함하지 마세요."""

OFF_TOPIC_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """
현재 대화는 '네이버 스마트스토어'와 관련된 내용만 다루어야 합니다.

사용자의 질문이 스마트스토어와 직접 관련이 없더라도, 
질문의 주제를 분석하여 스마트스토어와 연관된 주제로 연결해주는 방식으로 응답해야 합니다.

아래 형식을 참고하여 답변해주세요:

"저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.
다만, 질문 주제를 참고하여 아래와 같은 관련 질문을 드려볼 수 있어요:
- (사용자 질문을 스마트스토어 관점에서 리프레이즈한 제안 질문 1)
- (리프레이즈한 제안 질문 2)"
"""

# --- >>> 추가: 도메인 분류용 프롬프트 <<< ---
DOMAIN_CLASSIFICATION_SYSTEM_PROMPT = """당신은 사용자 질문이 '네이버 스마트스토어 판매자 지원 FAQ'와 관련된 내용인지 판단하는 분류기입니다. '네이버 스마트스토어 판매자 지원 FAQ'는 스마트스토어 개설, 상품 등록, 판매 관리, 정산, 배송, 광고, 고객 문의 처리 등 판매 활동 전반에 대한 내용을 다룹니다."""
DOMAIN_CLASSIFICATION_USER_PROMPT = """다음 사용자 질문이 '네이버 스마트스토어 판매자 지원 FAQ'에 대한 질문입니까?
오직 'Yes' 또는 'No'로만 대답해주세요. 다른 설명은 절대 추가하지 마세요.

<판단 기준 예시>
질문: "스마트스토어 입점 절차가 궁금해요." -> Yes
질문: "상품 상세페이지는 어떻게 꾸미나요?" -> Yes
질문: "오늘 날씨 어때?" -> No
질문: "강남역 맛집 추천해줘." -> No
질문: "네이버 아이디 비밀번호를 잊어버렸어요." -> No

<사용자 질문>
"{query}"

판단 (Yes 또는 No):"""
# --- <<< 추가 완료 <<< ---


def format_faq_context(
    retrieved_contexts: List[Dict[str, Any]], # 각 요소는 FAQ 단위 {'source_faq_id': ..., 'original_question': ..., 'answered_text': ..., 'rrf_score': ...}
    max_len: int = 1500, # 길이 제한 (토큰 기반 권장)
    # max_faqs: int = 3 # 포함할 최대 FAQ 수 (선택적)
) -> str:
    """
    검색된 FAQ 컨텍스트 목록(FAQ 단위)을 LLM 프롬프트 형식으로 변환합니다.
    메타데이터에 저장된 전체 답변 텍스트를 사용합니다.
    """
    if not retrieved_contexts:
        return "관련 FAQ 정보를 찾지 못했습니다."

    context_str = ""
    current_len = 0
    faqs_added = 0


    for i, faq_data in enumerate(retrieved_contexts):
        faq_id = faq_data.get('source_faq_id', f'unknown_faq_{i}')
        original_question = faq_data.get('original_question', 'N/A')
        answered_text = faq_data.get('answered_text', 'N/A') # 메타데이터에서 답변 전체 가져오기
        rrf_score = faq_data.get('rrf_score', 'N/A')

        # 유효한 데이터인지 확인
        if not original_question or original_question == 'N/A' or not answered_text or answered_text == 'N/A':
            logger.warning(f"Skipping FAQ {faq_id} due to missing question or answer text.")
            continue

        # FAQ 항목 생성
        entry = f"<FAQ Document {i+1} (Source FAQ ID: {faq_id}, RRF Score: {rrf_score:.4f})>\n"
        entry += f"Question: {original_question}\n"
        entry += f"Answer:\n{answered_text.strip()}\n" # 답변 텍스트 포함
        entry += f"</FAQ Document {i+1}>\n\n"

        entry_len = len(entry) # TODO: tiktoken으로 토큰 수 계산 권장

        # 길이 제한 확인
        if current_len + entry_len > max_len:
            logger.warning(f"Context length limit ({max_len}) reached. Stopped adding more FAQs.")
            break

        context_str += entry
        current_len += entry_len
        faqs_added += 1

        # 포함할 최대 FAQ 수 제한 (선택적)
        # if faqs_added >= max_faqs:
        #     logger.info(f"Maximum number of FAQs ({max_faqs}) reached.")
        #     break


    final_context = context_str.strip()
    if not final_context: # 유효한 컨텍스트가 하나도 없는 경우
         logger.warning("No valid context could be formatted.")
         return "관련 FAQ 정보를 찾지 못했습니다."

    logger.info(f"Formatted context created with {faqs_added} FAQs (approx. length: {current_len}).")
    return final_context


# --- Prompt Building Function ---
def build_prompt_messages(
    prompt_type: str,
    user_query: str,
    history: List[Dict[str, str]],
    retrieved_contexts: List[Dict[str, Any]] | None = None,
    main_answer: str | None = None,
    max_answer_len: int = 500
) -> List[Dict[str, str]]:
    """주어진 정보로 LLM API 호출을 위한 메시지 목록 생성"""

    messages = []
    system_content = ""
    user_content = user_query

    context_formatted = format_faq_context(retrieved_contexts or [])


    if prompt_type == "MAIN_ANSWER_STANDARD":
        system_content = BASE_SYSTEM_PROMPT + MAIN_ANSWER_STANDARD_SYSTEM_ADDENDUM.format(
            faq_context=context_formatted,
            user_query=user_query,
            max_answer_length=max_answer_len
        )
    elif prompt_type == "MAIN_ANSWER_DIRECT":
        system_content = BASE_SYSTEM_PROMPT + MAIN_ANSWER_DIRECT_SYSTEM_ADDENDUM.format(
            faq_context=context_formatted,
            user_query=user_query,
            max_answer_length=max_answer_len
        )

    elif prompt_type == "FOLLOW_UP":
        if not main_answer: raise ValueError("Main answer required for FOLLOW_UP")
        # FOLLOW_UP은 시스템 프롬프트가 길어질 수 있으므로 BASE 생략 고려 가능
        system_content = FOLLOW_UP_SYSTEM_ADDENDUM.format(
            faq_context=context_formatted, # 컨텍스트 포함 유지
            user_query=user_query,
            main_answer=main_answer
        )
        user_content = None # System 프롬프트만 사용
    elif prompt_type == "OFF_TOPIC":
        system_content = OFF_TOPIC_SYSTEM_PROMPT
        user_content = f"다음 사용자 질문은 스마트스토어와 관련이 없습니다. 규칙에 따라 응답해주세요: \"{user_query}\""

    elif prompt_type == "DOMAIN_CLASSIFICATION":
         system_content = DOMAIN_CLASSIFICATION_SYSTEM_PROMPT
         user_content = DOMAIN_CLASSIFICATION_USER_PROMPT.format(query=user_query)
         history = [] # 분류 시에는 히스토리 불필요

    elif prompt_type == "HISTORY_ANSWER_DIRECT":
        system_content = BASE_SYSTEM_PROMPT + HISTORY_ANSWER_DIRECT_SYSTEM_ADDENDUM.format(
            faq_context=context_formatted,
            user_query=user_query,
            max_answer_length=max_answer_len,
            history=history
        )
        
    # --- <<< 추가 완료 <<< ---
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    # 최종 메시지 목록 구성
    if system_content:
        messages.append({"role": "system", "content": system_content})

    # 히스토리 추가 (필요한 경우만)
    if prompt_type in ["MAIN_ANSWER_STANDARD", "MAIN_ANSWER_DIRECT"]:
        messages.extend(history)

    if user_content:
        messages.append({"role": "user", "content": user_content})

    # TODO: 전체 메시지 토큰 수 계산 및 제한 로직 추가 (tiktoken 사용)
    #       제한 초과 시 history나 context를 줄이는 전략 필요

    return messages