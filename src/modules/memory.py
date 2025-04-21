# src/modules/memory.py
import logging
import json
import time
from typing import List, Optional, Dict, Any
from redis.asyncio import Redis as AsyncRedis # redis 라이브러리 사용 (asyncio 지원)

from src.schemas.common import Message # Message 스키마 임포트 (role, content 필드 가정)
from src.core.config import settings # TTL 설정 등 가져오기

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Redis를 사용하여 대화 기록을 관리하는 클래스 (비동기).
    각 대화는 Redis List에 JSON 문자열 형태로 저장됩니다.
    """
    def __init__(self, redis_client: AsyncRedis, prefix: str = "convo:", ttl_seconds: int = settings.REDIS_TTL_SECONDS):
        """
        ConversationMemory 초기화.
        :param redis_client: 비동기 Redis 클라이언트 인스턴스.
        :param prefix: Redis 키 접두사.
        :param ttl_seconds: 대화 기록 유지 시간 (초).
        """
        self.redis: AsyncRedis = redis_client
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        logger.info(f"ConversationMemory initialized with prefix '{prefix}' and TTL {ttl_seconds}s.")

    def _get_key(self, conversation_id: str) -> str:
        """대화 ID에 해당하는 Redis 키 생성"""
        return f"{self.prefix}{conversation_id}"

    async def add_message(self, conversation_id: str, message: Message):
        """
        대화에 메시지를 추가하고 TTL을 갱신합니다.
        메시지는 JSON 문자열로 직렬화되어 저장됩니다.
        """
        if not conversation_id or not message or not message.content:
            logger.warning("Attempted to add an empty message or use empty conversation_id. Skipping.")
            return

        key = self._get_key(conversation_id)
        try:
            # 메시지 객체를 딕셔너리로 변환하고 타임스탬프 추가
            message_dict = message.model_dump() 
            message_dict['timestamp'] = time.time()

            # 딕셔너리를 JSON 문자열로 직렬화
            message_json = json.dumps(message_dict, ensure_ascii=False)

            # Redis List의 오른쪽에 메시지 추가 (RPUSH)
            await self.redis.rpush(key, message_json)

            # TTL 갱신 (키가 존재할 때만 TTL 설정)
            # 참고: RPUSH는 키가 없으면 새로 생성하므로, 항상 TTL 설정 가능
            await self.redis.expire(key, self.ttl_seconds)

            logger.debug(f"Added message to conversation '{conversation_id}'. Role: {message.role}, Content: {message.content[:30]}...")

        except Exception as e:
            logger.error(f"Error adding message to Redis for conversation '{conversation_id}': {e}")
            # 필요시 예외 재발생 또는 특정 처리

    async def get_history(self, conversation_id: str, k: int = settings.CONTEXT_TURNS) -> List[Message]:
        """
        최근 k개의 대화 기록을 시간 순서대로 가져옵니다.
        오래된 메시지가 리스트 앞쪽에 위치합니다.
        """
        if not conversation_id or k <= 0:
            return []

        key = self._get_key(conversation_id)
        history: List[Message] = []
        try:
            # Redis List의 끝에서부터 k개의 요소 가져오기 (LRANGE 사용, 인덱스 주의)
            # LRANGE key start stop (stop 포함)
            # 끝에서 k개를 가져오려면 start = -k, stop = -1
            # 예: 끝에서 3개 -> LRANGE key -3 -1
            messages_json = await self.redis.lrange(key, -k, -1)

            if messages_json:
                logger.debug(f"Retrieved {len(messages_json)} raw messages for conversation '{conversation_id}'.")
                for msg_json in messages_json:
                    try:
                        # JSON 문자열을 딕셔너리로 역직렬화
                        msg_dict = json.loads(msg_json)
                        # 딕셔너리를 Message 객체로 변환 (타임스탬프 제외 가능)
                        # Pydantic 모델 생성 시 extra='ignore' 설정 필요할 수 있음
                        history.append(Message(role=msg_dict.get('role'), content=msg_dict.get('content')))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode message JSON from Redis: {msg_json}")
                    except Exception as parse_e: # Pydantic 유효성 검사 오류 등
                         logger.warning(f"Failed to parse message object: {parse_e} - Data: {msg_dict}")

                logger.info(f"Returning {len(history)} parsed messages for conversation '{conversation_id}'.")
            else:
                 logger.info(f"No history found for conversation '{conversation_id}'.")


        except Exception as e:
            logger.error(f"Error getting history from Redis for conversation '{conversation_id}': {e}")
            # 오류 발생 시 빈 리스트 반환

        return history

    async def clear_history(self, conversation_id: str):
        """특정 대화 기록 삭제"""
        key = self._get_key(conversation_id)
        try:
            await self.redis.delete(key)
            logger.info(f"Cleared history for conversation '{conversation_id}'.")
        except Exception as e:
            logger.error(f"Error clearing history for conversation '{conversation_id}': {e}")

    # --- 추가 가능 메서드 ---
    # async def get_last_message(self, conversation_id: str) -> Optional[Message]: ...
    # async def count_messages(self, conversation_id: str) -> int: ...