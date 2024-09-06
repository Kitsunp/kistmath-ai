import unittest
import tensorflow as tf
from models.external_memory import InferenceMemory

class TestInferenceMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = InferenceMemory(capacity=500, embedding_size=64)

    def test_confidence_update(self):
        # Add an inference
        initial_inference = tf.random.normal([1, 64])
        initial_confidence = 0.6
        self.memory.add((initial_inference, initial_confidence))

        # Update the same inference with higher confidence
        updated_confidence = 0.9
        self.memory.add((initial_inference, updated_confidence))

        # Query and check if the updated confidence is used
        query = initial_inference
        result = self.memory.query(query)
        
        self.assertEqual(len(self.memory.inferences), 1, "Should have only one inference")
        self.assertAlmostEqual(self.memory.confidence_scores[0], updated_confidence, places=6)

    def test_inference_evolution(self):
        # Simulate evolving inferences
        base_inference = tf.random.normal([1, 64])
        
        for i in range(100):
            evolved_inference = base_inference + tf.random.normal([1, 64]) * 0.1 * i
            confidence = min(0.5 + i * 0.005, 1.0)  # Increasing confidence over time
            self.memory.add((evolved_inference, confidence))

        # Query with the latest evolved inference
        query = base_inference + tf.random.normal([1, 64]) * 0.1 * 100
        result = self.memory.query(query, top_k=5)

        # Check if the results show a progression of the inference
        confidences = [self.memory.confidence_scores[self.memory.index.get_nns_by_vector(r.numpy(), 1)[0]] for r in result]
        self.assertEqual(len(confidences), 5)
        self.assertTrue(all(confidences[i] <= confidences[i+1] for i in range(len(confidences)-1)), "Confidences should be in increasing order")

if __name__ == '__main__':
    unittest.main()