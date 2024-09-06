import numpy as np
import sympy as sp

class MathProblem:
    def __init__(self, problem, solution, difficulty, concept):
        self.problem = problem
        self.solution = solution
        self.difficulty = difficulty
        self.concept = concept

def generate_dataset(num_problems, stage, difficulty):
    problems = []
    if stage == 'elementary1':  # 1st-2nd grade
        for _ in range(num_problems):
            a, b = np.random.randint(1, int(10 * difficulty), size=2)
            op = np.random.choice(['+', '-'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
    elif stage == 'elementary2':  # 3rd-4th grade
        for _ in range(num_problems):
            a, b = np.random.randint(1, int(20 * difficulty), size=2)
            op = np.random.choice(['+', '-', '*'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
    elif stage == 'elementary3':  # 5th-6th grade
        for _ in range(num_problems):
            a, b = np.random.randint(1, int(30 * difficulty), size=2)
            op = np.random.choice(['+', '-', '*', '/'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
    elif stage == 'junior_high1':  # 7th-8th grade
        for _ in range(num_problems):
            a, b, c = np.random.randint(-int(10 * difficulty), int(10 * difficulty) + 1, size=3)
            if a == 0:
                a = 1
            problem = f"{a}x + {b} = {c}"
            solution = complex((c - b) / a)
            problems.append(MathProblem(problem, solution, difficulty, 'linear_equation'))
    elif stage == 'junior_high2':  # 9th grade
        for _ in range(num_problems):
            a, b, c = np.random.randint(-int(5 * difficulty), int(5 * difficulty) + 1, size=3)
            if a == 0:
                a = 1
            problem = f"{a}x^2 + {b}x + {c} = 0"
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                solution = (-b + np.sqrt(discriminant)) / (2*a)
            else:
                solution = complex(-b / (2*a), np.sqrt(-discriminant) / (2*a))
            problems.append(MathProblem(problem, solution, difficulty, 'quadratic'))
    elif stage == 'high_school1':  # 10th grade
        concepts = ['logarithm', 'complex_numbers', 'conic_sections', 'matrices']
        for _ in range(num_problems):
            concept = np.random.choice(concepts)
            if concept == 'logarithm':
                base = np.random.randint(2, 5)
                exponent = np.random.randint(1, int(5 * difficulty))
                problem = f"log_{base}(x) = {exponent}"
                solution = base ** exponent
            elif concept == 'complex_numbers':
                a, b = np.random.randint(-5, 6, size=2)
                c, d = np.random.randint(-5, 6, size=2)
                problem = f"({a} + {b}i) * ({c} + {d}i)"
                solution = complex(a*c - b*d, a*d + b*c)
            elif concept == 'conic_sections':
                h, k = np.random.randint(-5, 6, size=2)
                a = np.random.randint(1, 6)
                problem = f"(x - {h})^2 + (y - {k})^2 = {a}^2"
                solution = f"Circle with center ({h}, {k}) and radius {a}"
            else:  # matrices
                A = np.random.randint(-5, 6, size=(2, 2))
                B = np.random.randint(-5, 6, size=(2, 2))
                problem = f"A = {A.tolist()} B = {B.tolist()}. Calculate det(A * B)"
                solution = np.linalg.det(np.dot(A, B))
            problems.append(MathProblem(problem, solution, difficulty, concept))
    elif stage == 'high_school2':  # 11th grade
        concepts = ['trigonometry', 'vectors', 'combinatorics', 'sequences']
        for _ in range(num_problems):
            concept = np.random.choice(concepts)
            if concept == 'trigonometry':
                angle = np.random.randint(0, 360)
                func = np.random.choice(['sin', 'cos', 'tan'])
                problem = f"{func}({angle}°)"
                if func == 'sin':
                    solution = np.sin(np.radians(angle))
                elif func == 'cos':
                    solution = np.cos(np.radians(angle))
                else:
                    solution = np.tan(np.radians(angle))
                solution = complex(solution)
            elif concept == 'vectors':
                v1 = np.random.randint(-5, 6, size=3)
                v2 = np.random.randint(-5, 6, size=3)
                problem = f"v1 = {v1.tolist()}, v2 = {v2.tolist()}. Calculate v1 • v2"
                solution = np.dot(v1, v2)
            elif concept == 'combinatorics':
                n = np.random.randint(10, 21)
                r = np.random.randint(1, n+1)
                problem = f"C({n}, {r})"
                solution = sp.binomial(n, r)
            else:  # sequences
                a1 = np.random.randint(1, 10)
                d = np.random.randint(1, 5)
                n = np.random.randint(10, 21)
                problem = f"Find the {n}th term of the arithmetic sequence with a1 = {a1} and d = {d}"
                solution = a1 + (n - 1) * d
            problems.append(MathProblem(problem, solution, difficulty, concept))
    elif stage == 'high_school3':  # 12th grade
        concepts = ['limits', 'derivatives', 'integrals', 'differential_equations']
        for _ in range(num_problems):
            concept = np.random.choice(concepts)
            x = sp.Symbol('x')
            if concept == 'limits':
                a = np.random.randint(1, int(3 * difficulty))
                expr = sp.sin(a*x) / x
                problem = f"lim(x->0) ({expr})"
                solution = sp.limit(expr, x, 0)
            elif concept == 'derivatives':
                max_degree = int(3 * difficulty)
                num_terms = np.random.randint(1, max_degree + 1)
                coeffs = np.random.randint(1, int(5 * difficulty), size=num_terms)
                exponents = np.random.randint(1, max_degree + 1, size=num_terms)
                expr = sum(coeff * x**exp for coeff, exp in zip(coeffs, exponents))
                problem = f"d/dx ({expr})"
                solution = sp.diff(expr, x)
            elif concept == 'integrals':
                max_degree = int(2 * difficulty)
                num_terms = np.random.randint(1, max_degree + 1)
                coeffs = np.random.randint(1, int(3 * difficulty), size=num_terms)
                exponents = np.random.randint(0, max_degree + 1, size=num_terms)
                expr = sum(coeff * x**exp for coeff, exp in zip(coeffs, exponents))
                problem = f"∫ ({expr}) dx"
                solution = sp.integrate(expr, x)
            else:  # differential equations
                a = np.random.randint(1, 5)
                problem = f"y' + {a}y = 0"
                y = sp.Function('y')
                eq = sp.Eq(y(x).diff(x) + a*y(x), 0)
                solution = sp.dsolve(eq, y(x))
            problems.append(MathProblem(str(problem), str(solution), difficulty, concept))
    elif stage == 'university':  # University level
        concepts = ['multivariable_calculus', 'linear_algebra', 'complex_analysis', 'abstract_algebra']
        for _ in range(num_problems):
            concept = np.random.choice(concepts)
            if concept == 'multivariable_calculus':
                x, y, z = sp.symbols('x y z')
                expr = x**2 * y + y**2 * z + z**2 * x
                problem = f"∇f where f(x,y,z) = {expr}"
                solution = sp.derive_by_array(expr, (x, y, z))
            elif concept == 'linear_algebra':
                A = sp.Matrix(np.random.randint(-5, 6, size=(3, 3)))
                problem = f"Find the eigenvalues of A = {A}"
                solution = A.eigenvals()
            elif concept == 'complex_analysis':
                z = sp.Symbol('z')
                expr = z**3 + 2*z**2 + 3*z + 4
                problem = f"Find the residues of f(z) = 1 / ({expr})"
                solution = sp.residue(1 / expr, z)
            else:  # abstract algebra
                n = np.random.randint(10, 21)
                problem = f"Find the number of elements of order 2 in Z_{n}"
                solution = sum(1 for k in range(1, n) if sp.gcd(k, n) == 1 and pow(k, 2, n) == 1)
            problems.append(MathProblem(str(problem), str(solution), difficulty, concept))
    return problems
