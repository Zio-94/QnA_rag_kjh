# src/chatbot_api/schemas/common.py
from pydantic import BaseModel, Field
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: str = Field(..., min_length=1) 


    model_config = {
        "extra": "ignore" 
    }
