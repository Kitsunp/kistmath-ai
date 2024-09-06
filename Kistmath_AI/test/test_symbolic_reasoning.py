import unittest
import time
from models.symbolic_reasoning import SymbolicReasoning
import sympy as sp

class TestSymbolicReasoning(unittest.TestCase):
    def setUp(self):
        self.symbolic_reasoning = SymbolicReasoning()

    def test_solve_simple_equation(self):
        equation = sp.Eq(sp.Symbol('x') + 2, 5)
        solution = self.symbolic_reasoning.solve_equation(equation)
        self.assertEqual(solution, {sp.Symbol('x'): 3})

    def test_solve_quadratic_equation(self):
        equation = sp.Eq(sp.Symbol('x')**2 + 5*sp.Symbol('x') + 6, 0)
        solution = self.symbolic_reasoning.solve_equation(equation)
        self.assertEqual(set(solution), {-2, -3})

    def test_solve_system_of_equations(self):
        x, y = sp.symbols('x y')
        eq1 = sp.Eq(x + y, 5)
        eq2 = sp.Eq(2*x - y, 2)
        solution = self.symbolic_reasoning.solve_system_of_equations([eq1, eq2])
        self.assertEqual(solution, {x: sp.Rational(7, 3), y: sp.Rational(8, 3)})

    def test_simplify_expression(self):
        expr = (x + 1)**2 - (x**2 + 2*x + 1)
        simplified = self.symbolic_reasoning.simplify_expression(expr)
        self.assertEqual(simplified, 0)

    def test_calculate_derivative(self):
        expr = x**3 + 2*x**2 - 5*x + 3
        derivative = self.symbolic_reasoning.calculate_derivative(expr, x)
        self.assertEqual(derivative, 3*x**2 + 4*x - 5)

    def test_calculate_integral(self):
        expr = 2*x + 1
        integral = self.symbolic_reasoning.calculate_integral(expr, x)
        self.assertEqual(integral, x**2 + x)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.symbolic_reasoning.solve_equation("not an equation")
        
        with self.assertRaises(ValueError):
            self.symbolic_reasoning.calculate_derivative(x**2, y)

    def test_performance(self):
        start_time = time.time()
        complex_expr = sum([x**i for i in range(1000)])
        self.symbolic_reasoning.simplify_expression(complex_expr)
        end_time = time.time()
        self.assertLess(end_time - start_time, 5)  # Should take less than 5 seconds

    def test_symbolic_manipulation(self):
        expr1 = (x + y)**2
        expr2 = x**2 + 2*x*y + y**2
        self.assertTrue(self.symbolic_reasoning.are_expressions_equal(expr1, expr2))

        factor_expr = x**2 - y**2
        factored = self.symbolic_reasoning.factor_expression(factor_expr)
        self.assertEqual(factored, (x+y)*(x-y))

if __name__ == '__main__':
    unittest.main()