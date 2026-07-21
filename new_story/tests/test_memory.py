import pytest
from memory import MemoryManager, VectorStore


def test_vector_store_init():
    vs = VectorStore()
    assert vs.dim > 0
    assert vs.get_size() == 0


def test_memory_manager_add_memory():
    mm = MemoryManager(max_memories=5)
    result = mm.add_memory("test memory")
    assert result == 0
    assert mm.get_memory_count() == 1


def test_memory_manager_add_duplicate_memory():
    mm = MemoryManager()
    mm.add_memory("test memory")
    result = mm.add_memory("test memory")
    assert result == -1
    assert mm.get_memory_count() == 1


def test_memory_manager_memory_limit():
    mm = MemoryManager(max_memories=3)
    mm.add_memory("memory 1")
    mm.add_memory("memory 2")
    mm.add_memory("memory 3")
    mm.add_memory("memory 4")
    assert mm.get_memory_count() == 3
    memories = mm.get_all_memories()
    assert "memory 1" not in memories
    assert "memory 4" in memories


def test_memory_manager_retrieve_relevant():
    mm = MemoryManager()
    mm.add_memory("我喜欢吃苹果")
    mm.add_memory("我不喜欢吃香蕉")
    mm.add_memory("苹果是红色的")
    results = mm.retrieve_relevant_memories("苹果", k=2)
    assert len(results) <= 2


def test_memory_manager_clear():
    mm = MemoryManager()
    mm.add_memory("test memory")
    mm.clear_memories()
    assert mm.get_memory_count() == 0
    assert mm.get_all_memories() == []
