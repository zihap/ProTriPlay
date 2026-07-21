from .vector_store import VectorStore

class MemoryManager:
    def __init__(self, max_memories=20):
        self.max_memories = max_memories
        self.memories = []
        self.vector_store = VectorStore()

    def add_memory(self, memory_text):
        if memory_text in self.memories:
            return -1

        embedding = self.vector_store.get_embedding(memory_text)
        self.memories.append(memory_text)
        self.vector_store.add_vectors([embedding])

        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
            self.vector_store.clear()
            for memory in self.memories:
                embedding = self.vector_store.get_embedding(memory)
                self.vector_store.add_vectors([embedding])

        return len(self.memories) - 1

    def retrieve_relevant_memories(self, query, k=3):
        if len(self.memories) == 0:
            return []

        query_embedding = self.vector_store.get_embedding(query)
        scores, indices = self.vector_store.search(query_embedding, k)

        results = []
        seen_texts = set()

        for idx in indices:
            memory_text = self.memories[idx]

            if memory_text in seen_texts:
                continue

            is_duplicate = False
            for existing in results:
                memory_words = set(memory_text.split())
                existing_words = set(existing.split())
                if memory_words and existing_words:
                    similarity = len(memory_words & existing_words) / len(memory_words)
                    if similarity > 0.7:
                        is_duplicate = True
                        break

            if not is_duplicate:
                results.append(memory_text)
                seen_texts.add(memory_text)

            if len(results) >= k:
                break

        return results

    def get_all_memories(self):
        return self.memories

    def get_memory_count(self):
        return len(self.memories)

    def clear_memories(self):
        self.memories = []
        self.vector_store.clear()
