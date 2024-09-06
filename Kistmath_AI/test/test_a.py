import unittest
from test_external_memory import TestBTreeImproved

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBTreeImproved)
    unittest.TextTestRunner(verbosity=2).run(suite)