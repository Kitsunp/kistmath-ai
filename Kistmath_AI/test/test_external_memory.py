import pytest
import tensorflow as tf
import numpy as np
from Kistmath_AI.models.external_memory import (
    ExternalMemory, FormulativeMemory, ConceptualMemory,
    ShortTermMemory, LongTermMemory, InferenceMemory
)

@pytest.fixture
def memory_sizes():
    return {
        'external': 100,
        'formulative': 1000,
        'short_term': 100,
        'long_term': 10000,
        'inference': 500
    }

@pytest.fixture
def key_value_sizes():
    return {'key': 64, 'value': 128}

class TestExternalMemory:
    def test_initialization(self, memory_sizes, key_value_sizes):
        mem = ExternalMemory(memory_sizes['external'], key_value_sizes['key'], key_value_sizes['value'])
        assert mem.memory_size == memory_sizes['external']
        assert mem.key_size == key_value_sizes['key']
        assert mem.value_size == key_value_sizes['value']

    def test_query(self, memory_sizes, key_value_sizes):
        mem = ExternalMemory(memory_sizes['external'], key_value_sizes['key'], key_value_sizes['value'])
        query = tf.random.normal([1, key_value_sizes['key']])
        result = mem.query(query)
        assert result.shape == (1, key_value_sizes['value'])

    def test_update(self, memory_sizes, key_value_sizes):
        mem = ExternalMemory(memory_sizes['external'], key_value_sizes['key'], key_value_sizes['value'])
        key = tf.random.normal([1, key_value_sizes['key']])
        value = tf.random.normal([1, key_value_sizes['value']])
        mem.update(key, value)
        # Check if the usage has been updated
        assert tf.reduce_sum(mem.usage) > 0

class TestFormulativeMemory:
    def test_initialization(self, memory_sizes):
        mem = FormulativeMemory(memory_sizes['formulative'])
        assert mem.max_formulas == memory_sizes['formulative']
        assert mem.formula_embeddings.shape == (0, 64)

    def test_add_formula(self):
        mem = FormulativeMemory(10)
        formula = tf.random.normal([1, 64])
        terms = ["term1", "term2"]
        mem.add_formula(formula, terms)
        assert mem.formula_embeddings.shape == (1, 64)
        assert len(mem.formula_terms) == 1
        assert "term1" in mem.inverted_index and "term2" in mem.inverted_index

    def test_query_similar_formulas(self):
        mem = FormulativeMemory(10)
        for i in range(5):
            formula = tf.random.normal([1, 64])
            terms = [f"term{i}"]
            mem.add_formula(formula, terms)
        
        query = tf.random.normal([1, 64])
        query_terms = ["term1", "term3"]
        result, indices = mem.query_similar_formulas(query, query_terms, top_k=3)
        assert result.shape[0] <= 3
        assert len(indices) <= 3

class TestConceptualMemory:
    def test_add_and_query_concept(self):
        mem = ConceptualMemory()
        key = tf.random.normal([1, 64])
        concept = tf.random.normal([1, 128])
        mem.add_concept(key, concept)
        result = mem.query_similar_concepts(key, top_k=1)
        assert result.shape == (1, 64)

    def test_get_concept(self):
        mem = ConceptualMemory()
        key = tf.random.normal([1, 64])
        concept = tf.random.normal([1, 128])
        mem.add_concept(key, concept)
        retrieved_concept = mem.get_concept(key)
        assert tf.reduce_all(tf.equal(concept, retrieved_concept))

class TestShortTermMemory:
    def test_add_and_get_memory(self, memory_sizes):
        mem = ShortTermMemory(memory_sizes['short_term'])
        data = tf.random.normal([1, 64])
        mem.add_memory(data)
        assert len(mem.get_memory()) == 1

    def test_capacity_limit(self):
        mem = ShortTermMemory(5)
        for _ in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data)
        assert len(mem.get_memory()) == 5

    def test_query_recent_memories(self):
        mem = ShortTermMemory(10)
        for _ in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data)
        query = tf.random.normal([1, 64])
        result = mem.query_recent_memories(query, top_k=5)
        assert result.shape == (5, 64)

class TestLongTermMemory:
    def test_add_and_get_memory(self, memory_sizes):
        mem = LongTermMemory(memory_sizes['long_term'])
        data = tf.random.normal([1, 64])
        mem.add_memory(data)
        assert len(mem.get_memory()) == 1

    def test_importance_based_storage(self):
        mem = LongTermMemory(5)
        for i in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data, importance=float(i))
        memories = mem.get_memory()
        assert len(memories) == 5
        assert all(tf.reduce_mean(mem) > 4.0 for mem in memories)  # Only high importance memories should remain

    def test_query_important_memories(self):
        mem = LongTermMemory(10)
        for _ in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data, importance=np.random.rand())
        query = tf.random.normal([1, 64])
        result = mem.query_important_memories(query, top_k=5)
        assert result.shape == (5, 64)

class TestInferenceMemory:
    def test_add_and_get_inference(self, memory_sizes):
        mem = InferenceMemory(memory_sizes['inference'])
        inference = tf.random.normal([1, 64])
        mem.add_inference(inference, confidence=0.8)
        assert len(mem.get_inferences()) == 1

    def test_confidence_based_storage(self):
        mem = InferenceMemory(5)
        for i in range(10):
            inference = tf.random.normal([1, 64])
            mem.add_inference(inference, confidence=i/10)
        inferences = mem.get_inferences()
        assert len(inferences) == 5

    def test_query_confident_inferences(self):
        mem = InferenceMemory(10)
        for _ in range(10):
            inference = tf.random.normal([1, 64])
            mem.add_inference(inference, confidence=np.random.rand())
        query = tf.random.normal([1, 64])
        result = mem.query_confident_inferences(query, top_k=5)
        assert result.shape[0] <= 5
        assert result.shape[1] == 64

if __name__ == "__main__":
    pytest.main([__file__])