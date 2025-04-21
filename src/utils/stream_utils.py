# src/chatbot_api/utils/stream_utils.py
import logging
import json
from typing import AsyncGenerator, Callable, Awaitable, Any

from ..schemas.chat import StreamData # 스키마 경로 확인
from ..services.chat_service import ChatService # ChatService 타입 힌트 및 저장 메서드 호출용

logger = logging.getLogger(__name__)

async def sse_format_stream(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """순수 토큰 스트림을 SSE 형식 문자열로 변환합니다."""
    try:
        async for token in token_stream:
            stream_data = StreamData(token=token)
            yield f"data: {stream_data.model_dump_json()}\n\n"
    except Exception as e:
        # 스트림 생성 중 발생한 에러는 상위에서 처리 가정
        logger.error(f"Error during SSE formatting: {e}")
        error_data = StreamData(token=f"[STREAM_FORMAT_ERROR: {e}]", is_final=True, error=True)
        yield f"data: {error_data.model_dump_json()}\n\n"
    finally:
        # 스트림 종료 시그널
        done_data = StreamData(token="[DONE]", is_final=True)
        yield f"data: {done_data.model_dump_json()}\n\n"


async def handle_chat_stream_and_save(
    token_stream: AsyncGenerator[str, None], # ChatService 에서 생성된 토큰 스트림
    conversation_id: str,
    user_query: str,
    chat_service: ChatService # 저장 메서드 호출 위해 필요
) -> AsyncGenerator[str, None]:
    """
    토큰 스트림을 처리하여 SSE 형식으로 yield하고,
    완료 후 전체 응답을 이용해 대화를 저장합니다.
    """
    assistant_response_buffer = []
    formatted_stream = sse_format_stream(token_stream) # SSE 포맷터 적용

    try:
        async for sse_event in formatted_stream:
            yield sse_event # 포맷된 SSE 이벤트를 그대로 전달

            # SSE 데이터에서 실제 토큰 추출 (저장용) - 비효율적일 수 있음
            # 더 나은 방법: token_stream을 직접 순회하며 저장하고, 별도로 sse_format_stream 호출
            # 여기서는 일단 SSE 파싱으로 구현
            if sse_event.startswith("data:"):
                try:
                    data_str = sse_event[len("data:"):].strip()
                    if data_str:
                        stream_data = json.loads(data_str)
                        token = stream_data.get("token")
                        is_final = stream_data.get("is_final", False)
                        error = stream_data.get("error", False)
                        if not is_final and not error and token:
                            assistant_response_buffer.append(token)
                except json.JSONDecodeError:
                    pass # 파싱 오류 무시

    except Exception as e:
        logger.error(f"[{conversation_id}] Error while handling chat stream: {e}")
        # 스트리밍 중 에러 발생 시에도 저장 시도
        final_response = "".join(assistant_response_buffer)
        if not final_response: final_response = "[STREAM_HANDLER_ERROR]"
    else:
        # 스트림 정상 종료
        final_response = "".join(assistant_response_buffer)
    finally:
        # 대화 저장 로직
        logger.info(f"[{conversation_id}] Stream finished. Saving conversation via handler.")
        try:
            # 사용자 메시지 저장
            await chat_service.save_message(conversation_id, "user", user_query)
            # 최종 Assistant 응답 저장
            if final_response:
                await chat_service.save_message(conversation_id, "assistant", final_response.strip())
        except Exception as e:
            logger.error(f"[{conversation_id}] Failed to save conversation in stream handler: {e}")

# --- 개선된 핸들러 (토큰 직접 처리) ---
async def handle_chat_stream_and_save_v2(
    token_stream: AsyncGenerator[str, None],
    conversation_id: str,
    user_query: str,
    chat_service: ChatService
) -> AsyncGenerator[str, None]:
    """
    토큰 스트림을 직접 처리하여 SSE 형식으로 yield하고, 완료 후 저장 (개선된 버전)
    """
    assistant_response_buffer = []
    try:
        async for token in token_stream:
            assistant_response_buffer.append(token) # 토큰 바로 수집
            # SSE 형식으로 변환하여 yield
            stream_data = StreamData(token=token)
            yield f"data: {stream_data.model_dump_json()}\n\n"
    except Exception as e:
        logger.error(f"[{conversation_id}] Error during chat stream processing: {e}")
        # 에러 발생 시에도 수집된 내용 기반으로 최종 응답 구성 시도
        final_response = "".join(assistant_response_buffer)
        if not final_response: final_response = "[STREAM_PROCESSING_ERROR]"
        # 에러 SSE 이벤트 생성
        error_data = StreamData(token=f"[ERROR: {e}]", is_final=True, error=True)
        yield f"data: {error_data.model_dump_json()}\n\n"
    else:
        # 스트림 정상 종료
        final_response = "".join(assistant_response_buffer)
        # 종료 시그널 SSE 이벤트 생성
        done_data = StreamData(token="[DONE]", is_final=True)
        yield f"data: {done_data.model_dump_json()}\n\n"
    finally:
        # 대화 저장
        logger.info(f"[{conversation_id}] Stream finished. Saving conversation via handler v2.")
        try:
            await chat_service.save_message(conversation_id, "user", user_query)
            if final_response:
                await chat_service.save_message(conversation_id, "assistant", final_response.strip())
        except Exception as e:
            logger.error(f"[{conversation_id}] Failed to save conversation in stream handler v2: {e}")