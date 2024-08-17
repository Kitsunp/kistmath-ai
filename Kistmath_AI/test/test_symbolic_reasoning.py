import unittest
from Kistmath_AI.models.symbolic_reasoning import SymbolicReasoning
import sympy as sp

class TestSymbolicReasoning(unittest.TestCase):
    def setUp(self):
        self.symbolic_reasoning = SymbolicReasoning()

    def test_solve_equation(self):
        equation = sp.Eq(sp.Symbol('x') + 2, 5)
        solution = self.symbolic_reasoning.solve_equation(equation)
        self.assertEqual(solution, {sp.Symbol('x'): 3})

        equation = sp.Eq(sp.Symbol('y') - 3, 7)
        solution = self.symbolic_reasoning.solve_equation(equation)
        self.assertEqual(solution, {sp.Symbol('y'): 10})

if __name__ == '__main__':
    unittest.main()