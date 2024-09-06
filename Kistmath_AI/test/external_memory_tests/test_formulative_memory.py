import unittest
import tensorflow as tf
from models.external_memory import FormulativeMemory

class TestFormulativeMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = FormulativeMemory(max_formulas=10000, embedding_size=64)

    def test_add_and_query_with_real_formulas(self):
        formulas = [
            ("E = mc^2", ["energy", "mass", "speed of light"]),
            ("F = ma", ["force", "mass", "acceleration"]),
            ("PV = nRT", ["pressure", "volume", "temperature", "gas constant"]),
        ]

        for formula_text, terms in formulas:
            formula_embedding = tf.strings.to_hash_bucket_fast([formula_text], num_buckets=2**63-1)
            formula_embedding = tf.cast(formula_embedding, tf.float32) / (2**63-1)
            formula_embedding = tf.tile(formula_embedding, [1, 64])
            self.memory.add((formula_embedding, terms))

        query_embedding = tf.strings.to_hash_bucket_fast(["energy equation"], num_buckets=2**63-1)
        query_embedding = tf.cast(query_embedding, tf.float32) / (2**63-1)
        query_embedding = tf.tile(query_embedding, [1, 64])
        query_terms = ["energy", "mass"]

        result, indices = self.memory.query((query_embedding, query_terms))
        self.assertEqual(result.shape[0], 1)  # Should return at least one result
        
        # Verify that the returned formula is E = mc^2
        retrieved_formula = self.memory.formula_terms[indices[0]]
        self.assertIn("energy", retrieved_formula)
        self.assertIn("mass", retrieved_formula)

    def test_concept_drift(self):
        # Simulate concept drift by adding formulas over time
        for i in range(1000):
            formula_embedding = tf.random.normal([1, 64])
            terms = [f"term{i}", f"concept{i//100}"]
            self.memory.add((formula_embedding, terms))

        # Query for early concept
        query_embedding = tf.random.normal([1, 64])
        early_query_terms = ["term10", "concept0"]
        early_result, _ = self.memory.query((query_embedding, early_query_terms))

        # Query for later concept
        late_query_terms = ["term990", "concept9"]
        late_result, _ = self.memory.query((query_embedding, late_query_terms))

        # Check if later concepts are prioritized
        self.assertGreater(tf.reduce_sum(late_result), tf.reduce_sum(early_result))

if __name__ == '__main__':
    unittest.main()