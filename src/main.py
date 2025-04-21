
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.lifespan import lifespan
from src.api.chat import router as chat_router
import uvicorn


app = FastAPI(title="Initial Chatbot API", version="0.0.1", lifespan=lifespan)
app.include_router(chat_router)

@app.get("/")
async def read_root():
    return {"message": "API is running"}