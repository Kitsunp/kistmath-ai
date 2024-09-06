import unittest
import numpy as np
import time
from utils import add, subtract, multiply, divide, tokenize, evaluate_expression

class TestUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(3, 5), 8)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        self.assertAlmostEqual(add(3.14, 2.86), 6.0)
        self.assertEqual(add(1+2j, 3+4j), 4+6j)

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertAlmostEqual(subtract(3.14, 2.14), 1.0)
        self.assertEqual(subtract(1+2j, 3+4j), -2-2j)

    def test_multiply(self):
        self.assertEqual(multiply(3, 5), 15)
        self.assertEqual(multiply(-1, 1), -1)
        self.assertAlmostEqual(multiply(3.14, 2), 6.28)
        self.assertEqual(multiply(1+2j, 3+4j), -5+10j)

    def test_divide(self):
        self.assertEqual(divide(6, 3), 2)
        self.assertAlmostEqual(divide(1, 3), 0.3333333333, places=7)
        self.assertEqual(divide(1+2j, 1-2j), -0.6+0.8j)

        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)

    def test_edge_cases(self):
        self.assertTrue(np.isnan(add(float('nan'), 1)))
        self.assertTrue(np.isinf(multiply(float('inf'), 2)))
        self.assertTrue(np.isnan(divide(0, 0)))

    def test_tokenize(self):
        expression = "3 + 4 * (2 - 1)"
        tokens = tokenize(expression)
        expected = ['3', '+', '4', '*', '(', '2', '-', '1', ')']
        self.assertEqual(tokens, expected)

    def test_evaluate_expression(self):
        self.assertEqual(evaluate_expression("3 + 4 * 2"), 11)
        self.assertEqual(evaluate_expression("(3 + 4) * 2"), 14)
        self.assertAlmostEqual(evaluate_expression("sin(pi/2)"), 1.0)

        with self.assertRaises(ValueError):
            evaluate_expression("3 + * 4")

    def test_performance(self):
        start_time = time.time()
        for _ in range(1000000):
            add(3, 5)
        end_time = time.time()
        self.assertLess(end_time - start_time, 1)  # Should take less than 1 second

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            add("a", 5)
        with self.assertRaises(TypeError):
            subtract(3, "b")
        with self.assertRaises(TypeError):
            multiply({}, 2)
        with self.assertRaises(TypeError):
            divide([], 3)

if __name__ == '__main__':
    unittest.main()