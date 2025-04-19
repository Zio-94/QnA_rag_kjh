from chromadb import Client, PersistentClient

class VectorStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = PersistentClient(path=self.db_path)
    
    async def get_collection(self):
        return self.client.get_or_create_collection(name = "faq")

