import numpy as np

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
        for _ in range(num_problems):
            base = np.random.randint(2, 5)
            exponent = np.random.randint(1, int(5 * difficulty))
            problem = f"log_{base}(x) = {exponent}"
            solution = base ** exponent
            problems.append(MathProblem(problem, solution, difficulty, 'logarithm'))
    elif stage == 'high_school2':  # 11th grade
        for _ in range(num_problems):
            angle = np.random.randint(0, 360)
            func = np.random.choice(['sin', 'cos', 'tan'])
            problem = f"{func}({angle}°)"
            if func == 'sin':
                solution = np.sin(np.radians(angle))
            elif func == 'cos':
                solution = np.cos(np.radians(angle))
            else:
                solution = np.tan(np.radians(angle))
            problems.append(MathProblem(problem, complex(solution), difficulty, 'trigonometry'))
    elif stage == 'high_school3':  # 12th grade
        for _ in range(num_problems):
            a = np.random.randint(1, int(3 * difficulty))
            problem = f"lim(x->0) (sin({a}x) / x)"
            solution = a
            problems.append(MathProblem(problem, solution, difficulty, 'limits'))
    elif stage == 'university':  # University level
        for _ in range(num_problems):
            max_degree = int(3 * difficulty)
            num_terms = np.random.randint(1, max_degree + 1)
            coeffs = np.random.randint(1, int(5 * difficulty), size=num_terms)
            exponents = np.random.randint(1, max_degree + 1, size=num_terms)
            
            problem_str = "d/dx ("
            solution = 0
            for coeff, exp in zip(coeffs, exponents):
                problem_str += f"{coeff}x^{exp} + "
                solution += coeff * exp * (exp - 1)
            problem_str = problem_str.rstrip(" + ") + ")"
            
            problems.append(MathProblem(problem_str, complex(solution), difficulty, 'derivatives'))
    return problems
