import unittest
import tensorflow as tf
import numpy as np
from models.external_memory import LongTermMemory

class TestLongTermMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = LongTermMemory(capacity=1000)

    def test_importance_based_retention(self):
        # Add items with varying importance
        for i in range(1500):  # Exceed capacity
            importance = np.random.rand()
            self.memory.add(tf.constant([[float(i)]]), metadata={'importance': importance})

        # Check if high importance items are retained
        high_importance_items = [item for item, score in zip(self.memory.memory, self.memory.importance_scores) if score > 0.8]
        self.assertGreater(len(high_importance_items), 0, "High importance items should be retained")

        # Check if low importance items are removed
        low_importance_items = [item for item, score in zip(self.memory.memory, self.memory.importance_scores) if score < 0.2]
        self.assertLess(len(low_importance_items), len(high_importance_items), "Fewer low importance items should be retained")

    def test_memory_consolidation(self):
        # Simulate memory consolidation over time
        for day in range(30):
            for i in range(50):
                importance = np.random.rand()
                self.memory.add(tf.constant([[float(day * 50 + i)]]), metadata={'importance': importance})
            
            # Simulate consolidation by slightly increasing importance of retained memories
            for i in range(len(self.memory.importance_scores)):
                self.memory.importance_scores[i] *= 1.01

            self.memory._sort_by_importance()

        # Check if older, important memories are still retained
        old_memories = [item for item in self.memory.memory if item[0][0] < 500]
        self.assertGreater(len(old_memories), 0, "Some old, important memories should be retained")

if __name__ == '__main__':
    unittest.main()