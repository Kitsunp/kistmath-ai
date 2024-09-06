import unittest
import tensorflow as tf
import numpy as np
import os
import pickle
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.external_memory import BTree, MemoryError
import random

class TestBTreeImproved(unittest.TestCase):
    def setUp(self):
        self.btree = BTree(t=2)

    def test_insert_and_search(self):
        self.btree.insert((5.0, "data5"))
        self.btree.insert((3.0, "data3"))
        self.btree.insert((7.0, "data7"))

        self.assertEqual(self.btree.search(tf.constant(5.0)).numpy().decode('utf-8'), "data5")
        self.assertEqual(self.btree.search(tf.constant(3.0)).numpy().decode('utf-8'), "data3")
        self.assertEqual(self.btree.search(tf.constant(7.0)).numpy().decode('utf-8'), "data7")
        self.assertIsNone(self.btree.search(tf.constant(1.0)))

    def test_delete(self):
        self.btree.insert((5.0, "data5"))
        self.btree.insert((3.0, "data3"))
        self.btree.delete(tf.constant(3.0))
        self.assertIsNone(self.btree.search(tf.constant(3.0)))
        self.assertEqual(self.btree.search(tf.constant(5.0)).numpy().decode('utf-8'), "data5")

    def test_delete_complex(self):
        for i in range(10):
            self.btree.insert((float(i), f"data{i}"))

        self.btree.delete(tf.constant(2.0))
        self.assertIsNone(self.btree.search(tf.constant(2.0)))

        self.btree.delete(tf.constant(1.0))
        self.assertIsNone(self.btree.search(tf.constant(1.0)))

        self.btree.delete(tf.constant(5.0))
        self.assertIsNone(self.btree.search(tf.constant(5.0)))

        for i in [0, 3, 4, 6, 7, 8, 9]:
            result = self.btree.search(tf.constant(float(i)))
            self.assertIsNotNone(result)
            self.assertEqual(result.numpy().decode('utf-8'), f"data{i}")

        for i in [1, 2, 5]:
            self.assertIsNone(self.btree.search(tf.constant(float(i))))

    def test_performance(self, num_operations=10000, batch_size=10000):
        keys = [float(i) for i in range(num_operations)]
        values = [f"data{i}" for i in range(num_operations)]
        
        insert_times = []
        search_times = []
        delete_times = []

        for i in range(0, num_operations, batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_values = values[i:i+batch_size]

            # Medir el rendimiento de la inserción
            start_time = time.perf_counter()
            for k, v in zip(batch_keys, batch_values):
                self.btree.insert((k, v))
            insert_times.append(time.perf_counter() - start_time)

            # Medir el rendimiento de la búsqueda
            start_time = time.perf_counter()
            for k in batch_keys:
                self.btree.search(tf.constant(k))
            search_times.append(time.perf_counter() - start_time)

            # Medir el rendimiento de la eliminación
            start_time = time.perf_counter()
            for k in batch_keys:
                self.btree.delete(tf.constant(k))
            delete_times.append(time.perf_counter() - start_time)

        avg_insert_time = statistics.mean(insert_times)
        avg_search_time = statistics.mean(search_times)
        avg_delete_time = statistics.mean(delete_times)

        insert_rate = num_operations / sum(insert_times)
        search_rate = num_operations / sum(search_times)
        delete_rate = num_operations / sum(delete_times)
        print(f"\nResultados de la Prueba de Rendimiento (Operaciones: {num_operations}):")
        print(f"Inserción: Tiempo promedio por operación: {avg_insert_time*1e6:.2f} µs, Tasa: {insert_rate:.2f} ops/seg")
        print(f"Búsqueda: Tiempo promedio por operación: {avg_search_time*1e6:.2f} µs, Tasa: {search_rate:.2f} ops/seg")
        print(f"Eliminación: Tiempo promedio por operación: {avg_delete_time*1e6:.2f} µs, Tasa: {delete_rate:.2f} ops/seg")

        # Afirmaciones para asegurar que los tiempos promedio están dentro de los límites aceptables
        self.assertLess(avg_insert_time, 1e-4, "El tiempo promedio de inserción excede el umbral")
        self.assertLess(avg_search_time, 1e-4, "El tiempo promedio de búsqueda excede el umbral")
        self.assertLess(avg_delete_time, 1e-4, "El tiempo promedio de eliminación excede el umbral")


    def test_caching(self):
        for i in range(1000):
            self.btree.insert((float(i), f"value_{i}"))

        num_searches = 1000
        cache_total_time = 0
        tree_total_time = 0

        for i in range(num_searches):
            key = i % 100

            start_time = time.perf_counter()
            self.btree.search(tf.constant(float(key)))
            tree_total_time += time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.btree.search(tf.constant(float(key)))
            cache_total_time += time.perf_counter() - start_time

        cache_search_time = cache_total_time / num_searches
        tree_search_time = tree_total_time / num_searches

        print(f"Average cache search time: {cache_search_time}")
        print(f"Average tree search time: {tree_search_time}")

        self.assertLess(cache_search_time, tree_search_time, "Cache search should be faster than tree search")

    def test_error_handling(self):
        with self.assertRaises(MemoryError):
            BTree(t=1)

        with self.assertRaises(MemoryError):
            self.btree.insert(5.0)

    def test_edge_cases(self):
        self.btree.insert((3.14, "pi"))
        result = self.btree.search(tf.constant(3.14))
        self.assertIsNotNone(result, "Search for 3.14 returned None")
        self.assertEqual(result.numpy().decode('utf-8'), "pi")

        self.btree.insert((1e10, "large"))
        self.btree.insert((1e-10, "small"))
        
        result = self.btree.search(tf.constant(1e10))
        self.assertIsNotNone(result, "Search for 1e10 returned None")
        self.assertEqual(result.numpy().decode('utf-8'), "large")
        
        result = self.btree.search(tf.constant(1e-10))
        self.assertIsNotNone(result, "Search for 1e-10 returned None")
        self.assertEqual(result.numpy().decode('utf-8'), "small")

        result = self.btree.search(tf.constant(3.14159))
        self.assertIsNotNone(result, "Search for 3.14159 returned None")
        self.assertEqual(result.numpy().decode('utf-8'), "pi")

        result = self.btree.search(tf.constant(3.14 + 1e-9))
        self.assertIsNotNone(result, "Search with small tolerance failed")
        self.assertEqual(result.numpy().decode('utf-8'), "pi")

        result = self.btree.search(tf.constant(3.14 - 1e-9))
        self.assertIsNotNone(result, "Search with small tolerance failed")
        self.assertEqual(result.numpy().decode('utf-8'), "pi")

    def test_query_method(self):
        self.btree.insert((5.0, "data5"))
        self.btree.insert((3.0, "data3"))
        self.btree.insert((7.0, "data7"))

        result = self.btree.query(5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], b"data5")

        result = self.btree.query(1.0)
        self.assertEqual(len(result), 0)

    def test_memory_component_interface(self):
        self.btree.add((5.0, "data5"))
        self.assertEqual(self.btree.search(tf.constant(5.0)).numpy().decode('utf-8'), "data5")

        shared_data = self.btree.get_shareable_data()
        new_btree = BTree(t=2)
        new_btree.update_from_shared_data(shared_data)
        self.assertEqual(new_btree.search(tf.constant(5.0)).numpy().decode('utf-8'), "data5")

        self.btree.save("test_btree.pkl")
        loaded_btree = BTree(t=2)
        loaded_btree.load("test_btree.pkl")
        self.assertEqual(loaded_btree.search(tf.constant(5.0)).numpy().decode('utf-8'), "data5")

    def test_concurrent_operations(self):
        def insert_operation(start, end):
            for i in range(start, end):
                self.btree.insert((float(i), f"data{i}"))

        def search_operation(start, end):
            for i in range(start, end):
                self.btree.search(tf.constant(float(i)))

        def delete_operation(start, end):
            for i in range(start, end):
                if random.random() < 0.5:
                    self.btree.delete(tf.constant(float(i)))

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for i in range(4):
                futures.append(executor.submit(insert_operation, i*250, (i+1)*250))
                futures.append(executor.submit(search_operation, i*250, (i+1)*250))
                futures.append(executor.submit(delete_operation, i*250, (i+1)*250))

            for future in as_completed(futures):
                future.result()

        for i in range(1000):
            result = self.btree.search(tf.constant(float(i)))
            if result is not None:
                self.assertEqual(result.numpy().decode('utf-8'), f"data{i}")

    def test_node_splitting(self):
        for i in range(100):
            self.btree.insert((float(i), f"data{i}"))

        for i in range(100):
            result = self.btree.search(tf.constant(float(i)))
            self.assertIsNotNone(result)
            self.assertEqual(result.numpy().decode('utf-8'), f"data{i}")

        self.assertGreater(len(self.btree.root.keys), 1, "Root should have split at least once")

    def test_serialization(self):
        for i in range(100):
            self.btree.insert((float(i), f"data{i}"))

        serialized_data = pickle.dumps(self.btree)
        deserialized_btree = pickle.loads(serialized_data)

        for i in range(100):
            result = deserialized_btree.search(tf.constant(float(i)))
            self.assertIsNotNone(result)
            self.assertEqual(result.numpy().decode('utf-8'), f"data{i}")

    def test_large_scale_operations(self):
        num_elements = 10000
        batch_size = 100

        for i in range(0, num_elements, batch_size):
            batch = [(float(j), f"data{j}") for j in range(i, min(i + batch_size, num_elements))]
            for key, value in batch:
                self.btree.insert((key, value))

        for i in range(0, num_elements, batch_size):
            batch = [float(j) for j in range(i, min(i + batch_size, num_elements))]
            for key in batch:
                result = self.btree.search(tf.constant(key))
                self.assertIsNotNone(result)
                self.assertEqual(result.numpy().decode('utf-8'), f"data{int(key)}")

        for i in range(0, num_elements, 2):
            self.btree.delete(tf.constant(float(i)))

        for i in range(num_elements):
            result = self.btree.search(tf.constant(float(i)))
            if i % 2 == 0:
                self.assertIsNone(result)
            else:
                self.assertIsNotNone(result)
                self.assertEqual(result.numpy().decode('utf-8'), f"data{i}")

    def tearDown(self):
        del self.btree

if __name__ == '__main__':
    unittest.main()