import unittest
import tensorflow as tf
import time
from models.external_memory import ShortTermMemory

class TestShortTermMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = ShortTermMemory(capacity=100)

    def test_recency_effect(self):
        # Add 150 items to exceed capacity
        for i in range(150):
            self.memory.add(tf.constant([[float(i)]]))

        # Query for recent and old items
        recent_query = tf.constant([[149.0]])
        old_query = tf.constant([[50.0]])

        recent_result = self.memory.query(recent_query)
        old_result = self.memory.query(old_query)

        # Recent items should be retrieved more accurately
        self.assertAlmostEqual(recent_result[0][0].numpy(), 149.0, places=1)
        self.assertNotAlmostEqual(old_result[0][0].numpy(), 50.0, places=1)

    def test_rapid_update(self):
        update_frequency = 10  # milliseconds
        num_updates = 1000

        start_time = time.time()
        for i in range(num_updates):
            self.memory.add(tf.constant([[float(i)]]))
            time.sleep(update_frequency / 1000)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Performed {num_updates} rapid updates in {total_time:.2f} seconds")
        self.assertLess(total_time, num_updates * update_frequency / 1000 * 1.1)  # Allow 10% overhead

if __name__ == '__main__':
    unittest.main()