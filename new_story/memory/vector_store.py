import faiss
import numpy as np
from config import ark_embedding_dim, use_model
from api_client import get_client

class VectorStore:
    def __init__(self, dim=None):
        self.dim = dim if dim else (ark_embedding_dim if use_model == "ark" else 1536)
        self.index = faiss.IndexFlatL2(self.dim)
        self.memory_embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.embedding_client = get_client()

    def add_vectors(self, vectors):
        if len(vectors) == 0:
            return
        vectors = np.array(vectors, dtype=np.float32)
        self.memory_embeddings = np.vstack([self.memory_embeddings, vectors])
        self.index.reset()
        self.index.add(self.memory_embeddings)

    def search(self, query_vector, k=3):
        if len(self.memory_embeddings) == 0:
            return [], []
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(k * 2, len(self.memory_embeddings)))
        return scores[0], indices[0]

    def get_vector(self, index):
        if 0 <= index < len(self.memory_embeddings):
            return self.memory_embeddings[index]
        return None

    def get_size(self):
        return len(self.memory_embeddings)

    def clear(self):
        self.index.reset()
        self.memory_embeddings = np.zeros((0, self.dim), dtype=np.float32)

    def get_embedding(self, text):
        if use_model == "ark":
            try:
                response = self.embedding_client.embeddings.create(
                    model="doubao-embedding-vision", input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.dim).astype(np.float32)
        else:
            try:
                response = self.embedding_client.embeddings.create(
                    model="text-embedding-ada-002", input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.dim).astype(np.float32)
