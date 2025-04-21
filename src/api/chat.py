
# src/chatbot_api/api/chat.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncGenerator

# 내부 모듈 및 유틸리티 임포트
from src.schemas.chat import ChatRequest
from src.services.chat_service import ChatService
from src.dependencies.common import get_chat_service
from src.utils.stream_utils import handle_chat_stream_and_save_v2
from src.core.lifespan import lifespan
from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_class=StreamingResponse)
async def handle_chat_request(
    chat_request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    """
    사용자 채팅 요청을 받아 ChatService를 호출하고,
    스트림 핸들러를 통해 SSE 응답을 반환하며 대화를 저장합니다.
    """
    user_query = chat_request.message
    conversation_id = chat_request.conversation_id or f"session_{hash(user_query)}"

    logger.info(f"[{conversation_id}] Received chat request: {user_query[:50]}...")

    try:
        token_stream = chat_service.process_chat_stream(user_query, conversation_id)

        # 2. 스트림 핸들러(SSE 포맷팅 + 저장 포함)를 사용하여 StreamingResponse 생성
        return StreamingResponse(
            handle_chat_stream_and_save_v2( 
                token_stream=token_stream,
                conversation_id=conversation_id,
                user_query=user_query,
                chat_service=chat_service
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"[{conversation_id}] Failed to initiate chat stream in API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")