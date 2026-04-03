import uuid
import chromadb

class MemoryManager:
    def __init__(self, db_path="data/vectordb"):
        # Short-term Memory: List untuk menyimpan chat history (role user/assistant)
        self.chat_history = []
        
        # Long-term Memory (RAG): ChromaDB Persistent Client
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(name="personal_memory")

    def clear_history(self):
        """Mereset short-term memory (chat history) menjadi kosong."""
        self.chat_history = []

    def add_personal_data(self, text: str):
        """Menyimpan teks ke dalam long-term memory (ChromaDB)."""
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            ids=[doc_id]
        )
        return doc_id

    def query_personal_data(self, query_text: str, n_results: int = 1):
        """Mencari data yang relevan dari long-term memory."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
