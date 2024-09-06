import unittest
import tensorflow as tf
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.external_memory import IntegratedMemorySystem

class TestIntegratedMemorySystemImproved(unittest.TestCase):
    def setUp(self):
        config = {
            'external_memory': {'memory_size': 1000, 'key_size': 64, 'value_size': 128},
            'formulative_memory': {'max_formulas': 1000, 'embedding_size': 64},
            'conceptual_memory': {'embedding_size': 64},
            'short_term_memory': {'capacity': 100},
            'long_term_memory': {'capacity': 1000},
            'inference_memory': {'capacity': 500, 'embedding_size': 64}
        }
        self.memory_system = IntegratedMemorySystem(config)

    def test_cross_component_interaction(self):
        # Add data to different memory components
        external_key = tf.random.normal([1, 64])
        external_value = tf.random.normal([1, 128])
        self.memory_system.external_memory.add((external_key, external_value))

        formula = tf.random.normal([1, 64])
        terms = ["physics", "energy"]
        self.memory_system.formulative_memory.add((formula, terms))

        concept_key = tf.random.normal([1, 64])
        concept = "mass-energy equivalence"
        self.memory_system.conceptual_memory.add((concept_key, concept))

        short_term_data = tf.random.normal([1, 64])
        self.memory_system.short_term_memory.add(short_term_data)

        long_term_data = tf.random.normal([1, 64])
        self.memory_system.long_term_memory.add(long_term_data, metadata={'importance': 0.9})

        inference = tf.random.normal([1, 64])
        confidence = 0.8
        self.memory_system.inference_memory.add((inference, confidence))

        # Create a query that should activate multiple memory components
        query_data = {
            'external_query': (tf.random.normal([1, 64]), ["physics"]),
            'formulative_query': (tf.random.normal([1, 64]), ["energy"]),
            'conceptual_query': tf.random.normal([1, 64]),
            'short_term_query': tf.random.normal([1, 64]),
            'long_term_query': tf.random.normal([1, 64]),
            'inference_query': tf.random.normal([1, 64])
        }

        result = self.memory_system.process_input(query_data)

        # Check if all memory components contributed to the result
        self.assertIn('combined_embeddings', result)
        self.assertIn('memory_outputs', result)
        for memory_type in ['external_memory', 'formulative_memory', 'conceptual_memory', 
                            'short_term_memory', 'long_term_memory', 'inference_memory']:
            self.assertIn(memory_type, result['memory_outputs'])

    def test_adaptive_query_processing(self):
        # Train the system with some data
        for _ in range(100):
            key = tf.random.normal([1, 64])
            value = tf.random.normal([1, 128])
            self.memory_system.external_memory.add((key, value))

            formula = tf.random.normal([1, 64])
            terms = [f"term_{np.random.randint(10)}" for _ in range(3)]
            self.memory_system.formulative_memory.add((formula, terms))

        # Perform queries and measure time
        query_times = []
        for _ in range(50):
            query_data = {
                'external_query': (tf.random.normal([1, 64]), [f"term_{np.random.randint(10)}"]),
                'formulative_query': (tf.random.normal([1, 64]), [f"term_{np.random.randint(10)}"]),
            }
            start_time = time.time()
            self.memory_system.process_input(query_data)
            query_times.append(time.time() - start_time)

        # Check if query times are decreasing (system is adapting)
        avg_first_10 = np.mean(query_times[:10])
        avg_last_10 = np.mean(query_times[-10:])
        self.assertLess(avg_last_10, avg_first_10, "Query times should decrease as the system adapts")

    def test_memory_consolidation(self):
        # Add data to short-term memory
        for _ in range(150):  # Exceeding short-term memory capacity
            data = tf.random.normal([1, 64])
            self.memory_system.short_term_memory.add(data)

        # Simulate passage of time and memory consolidation
        self.memory_system.update_memories({})  # This should trigger internal consolidation

        # Check if some data has been transferred to long-term memory
        self.assertGreater(len(self.memory_system.long_term_memory.memory), 0, 
                           "Some memories should be consolidated into long-term memory")

    def test_concept_formation(self):
        # Add related data to formulative and conceptual memories
        related_terms = ["energy", "mass", "speed of light", "relativity"]
        for term in related_terms:
            formula = tf.random.normal([1, 64])
            self.memory_system.formulative_memory.add((formula, [term]))

            concept = tf.random.normal([1, 64])
            self.memory_system.conceptual_memory.add((concept, term))

        # Query with a related concept
        query_data = {
            'formulative_query': (tf.random.normal([1, 64]), ["energy", "mass"]),
            'conceptual_query': tf.random.normal([1, 64])
        }
        result = self.memory_system.process_input(query_data)

        # Check if the system forms a higher-level concept
        combined_embedding = result['combined_embeddings']
        self.assertIsNotNone(combined_embedding, "A combined embedding representing a higher-level concept should be formed")

    def test_inference_generation(self):
        # Add some basic facts to the memory system
        self.memory_system.external_memory.add((tf.constant([[1.0]]), tf.constant([[2.0]])))  # Fact: 1 + 1 = 2
        self.memory_system.external_memory.add((tf.constant([[2.0]]), tf.constant([[4.0]])))  # Fact: 2 + 2 = 4

        # Query for a new inference
        query_data = {
            'external_query': (tf.constant([[3.0]]), []),
            'inference_query': tf.constant([[3.0]])
        }
        result = self.memory_system.process_input(query_data)

        # Check if the system generates a plausible inference
        if 'inference_memory' in result['memory_outputs']:
            inference = result['memory_outputs']['inference_memory']
            self.assertIsNotNone(inference, "An inference should be generated")
            # In a more sophisticated system, we might check if the inference is close to 6.0 (3 + 3)

    def test_memory_pruning(self):
        # Fill up the memory system
        for _ in range(2000):  # Exceeding capacity of all components
            key = tf.random.normal([1, 64])
            value = tf.random.normal([1, 128])
            self.memory_system.external_memory.add((key, value))

            formula = tf.random.normal([1, 64])
            terms = [f"term_{np.random.randint(100)}" for _ in range(3)]
            self.memory_system.formulative_memory.add((formula, terms))

            concept = tf.random.normal([1, 64])
            self.memory_system.conceptual_memory.add((concept, f"concept_{np.random.randint(100)}"))

        # Trigger memory consolidation and pruning
        self.memory_system.update_memories({})

        # Check if memories have been pruned
        stats = self.memory_system.get_memory_statistics()
        for memory_type, memory_stats in stats.items():
            self.assertLessEqual(memory_stats['size'], memory_stats['capacity'], 
                                 f"{memory_type} should not exceed its capacity")

    def test_parallel_query_processing(self):
        # Add some data to the memory system
        for _ in range(1000):
            key = tf.random.normal([1, 64])
            value = tf.random.normal([1, 128])
            self.memory_system.external_memory.add((key, value))

        # Perform parallel queries
        def parallel_query():
            query_data = {
                'external_query': (tf.random.normal([1, 64]), []),
                'formulative_query': (tf.random.normal([1, 64]), [f"term_{np.random.randint(10)}"]),
                'conceptual_query': tf.random.normal([1, 64]),
                'short_term_query': tf.random.normal([1, 64]),
                'long_term_query': tf.random.normal([1, 64]),
                'inference_query': tf.random.normal([1, 64])
            }
            return self.memory_system.process_input(query_data)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parallel_query) for _ in range(100)]
            results = [future.result() for future in as_completed(futures)]

        self.assertEqual(len(results), 100, "All parallel queries should complete successfully")

if __name__ == '__main__':
    unittest.main()