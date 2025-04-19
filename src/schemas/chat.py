from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class StreamData(BaseModel):
    token: str
    is_final: bool = False
    error: bool = False
