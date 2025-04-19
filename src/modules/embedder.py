from openai import OpenAI
from typing import List



class Embedder:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, model=model_name)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return await self.client.embeddings.create(
            input=documents,
            model=self.model_name
        )