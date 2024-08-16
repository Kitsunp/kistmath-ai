import unittest
from Kistmath_AI.utils import add

class TestUtils(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(3, 5), 8)
        self.assertEqual(add(10, 20), 30)

    def test_add_non_numeric_arguments(self):
        with self.assertRaises(TypeError):
            add("a", 5)
        with self.assertRaises(TypeError):
            add(3, "b")

if __name__ == '__main__':
    unittest.main()