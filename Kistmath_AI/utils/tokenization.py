import numpy as np
import tensorflow as tf
from config.settings import VOCAB_SIZE, MAX_LENGTH, MAX_TERMS

def tokenize_problem(problem, stage, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    if isinstance(stage, tf.Tensor):
        stage = stage.numpy().decode('utf-8')
    elif isinstance(stage, bytes):
        stage = stage.decode('utf-8')
    elif not isinstance(stage, str):
        raise ValueError(f"Unexpected type for stage: {type(stage)}")
    if stage == 'university':
        return tokenize_calculus_problem(problem, max_length)
    elif stage.startswith('high_school'):
        return tokenize_advanced_problem(problem, vocab_size, max_length)
    else:
        return tokenize_basic_problem(problem, vocab_size, max_length)

def tokenize_basic_problem(problem, vocab_size, max_length):
    tokens = problem.lower().split()
    tokens = [hash(token) % vocab_size for token in tokens]
    return pad_tokens(tokens, max_length)

def tokenize_advanced_problem(problem, vocab_size, max_length):
    # Tokenización más avanzada para problemas de secundaria
    tokens = []
    for token in problem.replace('(', ' ( ').replace(')', ' ) ').split():
        if token.isalpha():
            tokens.append(hash(token) % vocab_size)
        elif token.isdigit() or token in '+-*/^()':
            tokens.append(ord(token))
        else:
            # Manejar números decimales y otros casos especiales
            try:
                tokens.append(int(float(token) * 100))  # Preservar 2 decimales
            except ValueError:
                tokens.append(hash(token) % vocab_size)
    return pad_tokens(tokens, max_length)

def tokenize_calculus_problem(problem, max_length):
    func_str = problem.split("d/dx ")[1].strip("()")
    terms = func_str.replace("-", "+-").split("+")
    
    coeffs = []
    exponents = []
    
    for term in terms:
        if 'x^' in term:
            coeff, exp = term.split('x^')
            coeffs.append(float(coeff) if coeff else 1)
            exponents.append(float(exp))
        elif 'x' in term:
            coeff = term.split('x')[0]
            coeffs.append(float(coeff) if coeff else 1)
            exponents.append(1)
        else:
            coeffs.append(float(term))
            exponents.append(0)
    
    # Normalizar coeficientes y exponentes
    max_coeff = max(abs(c) for c in coeffs) or 1
    max_exp = max(exponents) or 1
    normalized_coeffs = [c / max_coeff for c in coeffs]
    normalized_exps = [e / max_exp for e in exponents]
    
    result = normalized_coeffs + normalized_exps
    return pad_tokens(result, max_length)

def pad_tokens(tokens, max_length):
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + [0] * (max_length - len(tokens))