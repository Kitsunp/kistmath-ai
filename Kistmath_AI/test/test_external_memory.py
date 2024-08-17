import pytest
import tensorflow as tf
import numpy as np
from Kistmath_AI.models.external_memory import (
    ExternalMemory, FormulativeMemory, ConceptualMemory,
    ShortTermMemory, LongTermMemory, InferenceMemory
)
from Kistmath_AI.models.kistmat_ai import Kistmat_AI
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
        assert mem.keys.shape == (memory_sizes['external'], key_value_sizes['key'])
        assert mem.values.shape == (memory_sizes['external'], key_value_sizes['value'])
        assert mem.usage.shape == (memory_sizes['external'],)

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
        assert tf.reduce_max(mem.usage) > 0

    def test_memory_overflow(self, memory_sizes, key_value_sizes):
        mem = ExternalMemory(2, key_value_sizes['key'], key_value_sizes['value'])
        for _ in range(5):  # Overflow the memory
            key = tf.random.normal([1, key_value_sizes['key']])
            value = tf.random.normal([1, key_value_sizes['value']])
            mem.update(key, value)
        assert tf.reduce_min(mem.usage) > 0  # All slots should be used

class TestFormulativeMemory:
    def test_initialization(self, memory_sizes):
        mem = FormulativeMemory(memory_sizes['formulative'])
        assert mem.max_formulas == memory_sizes['formulative']
        assert mem.formula_embeddings.shape[1] == 64

    def test_add_formula(self, memory_sizes):
        mem = FormulativeMemory(memory_sizes['formulative'])
        formula = tf.random.normal([1, 64])
        initial_size = mem.formula_embeddings.shape[0]
        mem.add_formula(formula)
        assert mem.formula_embeddings.shape[0] == initial_size + 1

    def test_query_similar_formulas(self, memory_sizes):
        mem = FormulativeMemory(memory_sizes['formulative'])
        for _ in range(10):
            formula = tf.random.normal([1, 64])
            mem.add_formula(formula)
        query = tf.random.normal([1, 64])
        result = mem.query_similar_formulas(query, top_k=5)
        assert result.shape == (5, 64)

    def test_memory_overflow(self, memory_sizes):
        mem = FormulativeMemory(5)
        for _ in range(10):  # Overflow the memory
            formula = tf.random.normal([1, 64])
            mem.add_formula(formula)
        assert mem.formula_embeddings.shape[0] == 5

class TestConceptualMemory:
    def test_add_and_query_concept(self):
        mem = ConceptualMemory()
        key = tf.random.normal([1, 64])
        concept = tf.random.normal([1, 128])
        mem.add_concept(key, concept)
        result = mem.query_similar_concepts(key, top_k=1)
        assert tf.reduce_mean(tf.abs(result - concept)) < 1e-6

    def test_multiple_concepts(self):
        mem = ConceptualMemory()
        for _ in range(10):
            key = tf.random.normal([1, 64])
            concept = tf.random.normal([1, 128])
            mem.add_concept(key, concept)
        query = tf.random.normal([1, 64])
        result = mem.query_similar_concepts(query, top_k=5)
        assert result.shape == (5, 64)

class TestShortTermMemory:
    def test_add_and_get_memory(self, memory_sizes):
        mem = ShortTermMemory(memory_sizes['short_term'])
        data = tf.random.normal([1, 64])
        mem.add_memory(data)
        assert len(mem.get_memory()) == 1

    def test_capacity_limit(self, memory_sizes):
        mem = ShortTermMemory(5)
        for _ in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data)
        assert len(mem.get_memory()) == 5

    def test_query_recent_memories(self, memory_sizes):
        mem = ShortTermMemory(memory_sizes['short_term'])
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

    def test_importance_based_storage(self, memory_sizes):
        mem = LongTermMemory(5)
        for i in range(10):
            data = tf.random.normal([1, 64])
            mem.add_memory(data, importance=float(i))
        memories = mem.get_memory()
        assert len(memories) == 5
        assert all(tf.reduce_mean(mem) > 4.0 for mem in memories)  # Only high importance memories should remain

    def test_query_important_memories(self, memory_sizes):
        mem = LongTermMemory(memory_sizes['long_term'])
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

    def test_confidence_based_storage(self, memory_sizes):
        mem = InferenceMemory(5)
        for i in range(10):
            inference = tf.random.normal([1, 64])
            mem.add_inference(inference, confidence=i/10)
        inferences = mem.get_inferences()
        assert len(inferences) == 5
        assert all(tf.reduce_mean(inf) > 0.5 for inf in inferences)  # Only high confidence inferences should remain

    def test_query_confident_inferences(self, memory_sizes):
        mem = InferenceMemory(memory_sizes['inference'])
        for _ in range(10):
            inference = tf.random.normal([1, 64])
            mem.add_inference(inference, confidence=np.random.rand())
        query = tf.random.normal([1, 64])
        result = mem.query_confident_inferences(query, top_k=5)
        assert result.shape == (5, 64)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])