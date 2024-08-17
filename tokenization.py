import numpy as np
from config.settings import VOCAB_SIZE, MAX_LENGTH, MAX_TERMS

def tokenize_problem(problem, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    tokens = problem.lower().split()
    tokens = [hash(token) % vocab_size for token in tokens]
    tokens = tokens[:max_length]
    tokens += [0] * (max_length - len(tokens))
    return tokens

def tokenize_calculus_problem(problem, max_terms=MAX_TERMS):
    func_str = problem.split("d/dx ")[1].strip("()")
    terms = func_str.replace("-", "+-").split("+")
    
    coeffs = np.zeros(max_terms)
    exponents = np.zeros(max_terms)
    
    for i, term in enumerate(terms[:max_terms]):
        if 'x^' in term:
            coeff, exp = term.split('x^')
            coeffs[i] = float(coeff) if coeff else 1
            exponents[i] = float(exp)
        elif 'x' in term:
            coeff = term.split('x')[0]
            coeffs[i] = float(coeff) if coeff else 1
            exponents[i] = 1
        else:
            coeffs[i] = float(term)
            exponents[i] = 0
    
    coeffs = coeffs / np.max(np.abs(coeffs)) if np.max(np.abs(coeffs)) > 0 else coeffs
    exponents = exponents / np.max(exponents) if np.max(exponents) > 0 else exponents
    
    return np.pad(np.concatenate([coeffs, exponents]), (0, MAX_LENGTH - 2*max_terms))