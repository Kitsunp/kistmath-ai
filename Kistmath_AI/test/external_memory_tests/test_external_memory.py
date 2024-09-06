import unittest
import tensorflow as tf
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.external_memory import ExternalMemory

class TestExternalMemoryImproved(unittest.TestCase):
    def setUp(self):
        self.memory = ExternalMemory(memory_size=1000, key_size=64, value_size=128)

    def test_add_and_query_large_scale(self):
        num_entries = 10000
        keys = tf.random.normal([num_entries, 64])
        values = tf.random.normal([num_entries, 128])
        
        for i in range(num_entries):
            self.memory.add((tf.expand_dims(keys[i], 0), tf.expand_dims(values[i], 0)))

        query = tf.random.normal([1, 64])
        result, _ = self.memory.query(query)

        self.assertEqual(result.shape, (1, 128))
        self.assertEqual(tf.shape(self.memory.memory_embeddings)[0], 1000)  # Check if capacity is maintained

    def test_concurrent_access(self):
        def add_entry():
            key = tf.random.normal([1, 64])
            value = tf.random.normal([1, 128])
            self.memory.add((key, value))

        def query_memory():
            query = tf.random.normal([1, 64])
            return self.memory.query(query)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(100):
                futures.append(executor.submit(add_entry))
                futures.append(executor.submit(query_memory))

            for future in as_completed(futures):
                future.result()  # This will raise an exception if one occurred during the calls

    def test_memory_efficiency(self):
        import psutil
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss
        
        num_entries = 10000
        keys = tf.random.normal([num_entries, 64])
        values = tf.random.normal([num_entries, 128])
        
        for i in range(num_entries):
            self.memory.add((tf.expand_dims(keys[i], 0), tf.expand_dims(values[i], 0)))

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after {num_entries} entries: {memory_increase / 1024 / 1024:.2f} MB")
        self.assertLess(memory_increase, 100 * 1024 * 1024, "Memory usage exceeds expected threshold")

if __name__ == '__main__':
    unittest.main()