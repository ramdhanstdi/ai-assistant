import yaml
import os
import chromadb
from chromadb.utils import embedding_functions

class VectorDBManager:
    def __init__(self):
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            if os.path.exists(alt_path):
                config_path = alt_path

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            
        memory_config = config.get("memory", {})
        persist_directory = memory_config.get("persist_directory")
        collection_name = memory_config.get("collection_name")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
        )

    def save_fact(self, text: str, id: str):
        self.collection.upsert(documents=[text], ids=[id])

    def search_context(self, query: str, n_results: int = 2) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        
        documents = results.get('documents')
        if documents and len(documents) > 0 and documents[0]:
            return " ".join(documents[0])
            
        return ""
