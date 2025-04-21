from pydantic import BaseModel
from enum import Enum
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class StreamData(BaseModel):
    token: str
    is_final: bool = False
    error: bool = False

class ChatFlowType(str, Enum):
    OFF_TOPIC = "OFF_TOPIC"
    GENERATE_DIRECT = "GENERATE_DIRECT"
    GENERATE_STANDARD = "GENERATE_STANDARD"
    HANDLE_ERROR = "HANDLE_ERROR"
