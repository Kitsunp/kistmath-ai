import numpy as np
from utils.tokenization import tokenize_problem
from utils.logging import log_message
import traceback

def safe_tokenize_problem(problem, learning_stage):
    try:
        tokenized = tokenize_problem(problem, learning_stage)
        log_message(f"Problema tokenizado exitosamente: {problem}")
        return tokenized
    except Exception as e:
        error_message = f"Error al tokenizar el problema: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)
        raise ValueError("No se pudo tokenizar el problema")

def prepare_data(problems, model):
    X = []
    y = []
    for problem in problems:
        try:
            tokenized = safe_tokenize_problem(problem.problem, model.get_learning_stage())
            X.append(tokenized)
            if isinstance(problem.solution, complex):
                y.append([problem.solution.real, problem.solution.imag])
            else:
                y.append([problem.solution, 0])
        except ValueError:
            # Ignoramos los problemas que no se pueden tokenizar
            continue
    
    X = np.array(X)
    y = np.array(y, dtype=np.float32)
    
    # Asegurarse de que X es 3D y y es 2D
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
    
    return X, y