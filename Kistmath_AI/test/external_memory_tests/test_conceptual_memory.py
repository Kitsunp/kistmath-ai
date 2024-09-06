import unittest
import tensorflow as tf
from models.external_memory import ConceptualMemory

class TestConceptualMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = ConceptualMemory(embedding_size=64)

    def test_concept_hierarchy(self):
        # Create a simple concept hierarchy
        concepts = [
            ("animal", tf.random.normal([1, 64])),
            ("mammal", tf.random.normal([1, 64])),
            ("dog", tf.random.normal([1, 64])),
            ("cat", tf.random.normal([1, 64])),
        ]

        # Add concepts to the memory
        for concept, embedding in concepts:
            self.memory.add((embedding, concept))

        # Query for a specific concept
        dog_query = concepts[2][1]  # Dog embedding
        results = self.memory.query(dog_query, top_k=3)

        # Check if the hierarchy is preserved in the results
        retrieved_concepts = [self.memory.concepts[tf.reduce_sum(emb).numpy()] for emb in results]
        self.assertIn("dog", retrieved_concepts)
        self.assertIn("mammal", retrieved_concepts)
        self.assertIn("animal", retrieved_concepts)

    def test_concept_generalization(self):
        # Add some animal concepts
        animals = ["dog", "cat", "elephant", "lion", "tiger"]
        for animal in animals:
            embedding = tf.random.normal([1, 64])
            self.memory.add((embedding, animal))

        # Create a query that's a mix of known animal embeddings
        query = tf.reduce_mean([self.memory.concept_embeddings[i] for i in range(len(animals))], axis=0)
        query = tf.reshape(query, [1, 64])
        # Query the memory
        result = self.memory.query(query)
        retrieved_concept = self.memory.concepts[tf.reduce_sum(result).numpy()]
        # Check if the retrieved concept is one of the known animals    
        # This test will fail if the order of the animals list changes
        self.assertIn(retrieved_concept, animals, "Retrieved concept should be one of the known animals")

if __name__ == '__main__':
    unittest.main()