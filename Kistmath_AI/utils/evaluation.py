import numpy as np
from utils.tokenization import tokenize_problem, tokenize_calculus_problem
from utils.logging import log_message

class EvaluationError(Exception):
    """Base exception class for evaluation-related errors."""
    pass

class EvaluationWarning(EvaluationError):
    """Exception class for non-critical evaluation warnings."""
    pass

class EvaluationCritical(EvaluationError):
    """Exception class for critical evaluation errors that prevent normal operation."""
    pass

def evaluate_readiness(model, problems, threshold):
    try:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        current_stage = model.get_learning_stage()
        if current_stage == 'university':
            X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
            y = np.array([p.solution for p in problems])
        else:
            X = np.array([tokenize_problem(p.problem, current_stage) for p in problems])
            y_real = np.array([p.solution.real for p in problems])
            y_imag = np.array([p.solution.imag for p in problems])
            y = np.column_stack((y_real, y_imag))
        
        # Ensure input shape is correct
        input_shape = model.input_shape[1:]
        X = X.reshape((-1,) + input_shape)
        
        predictions = model.predict(X)
        
        # Ensure predictions and y have the same shape
        if predictions.shape != y.shape:
            log_message(f"Predictions shape {predictions.shape} doesn't match y shape {y.shape}. Squeezing predictions.", is_error=True)
            predictions = predictions.squeeze()
        
        if predictions.shape != y.shape:
            raise ValueError(f"Predictions shape {predictions.shape} doesn't match y shape {y.shape} after squeezing")
        
        mse = np.mean(np.square(y - predictions))
        r2 = 1 - (np.sum(np.square(y - predictions)) / np.sum(np.square(y - np.mean(y))))
        
        log_message(f"Evaluation results: MSE = {mse}, R2 = {r2}")
        return r2 > threshold

    except AttributeError as e:
        log_message(f"Invalid model or problem attribute: {e}", is_error=True)
        raise EvaluationError(f"Invalid model or problem attribute: {e}")
    except ValueError as e:
        log_message(f"Shape mismatch or invalid value: {e}", is_error=True)
        raise EvaluationError(f"Shape mismatch or invalid value: {e}")
    except TypeError as e:
        log_message(f"Invalid type in evaluation: {e}", is_error=True)
        raise EvaluationError(f"Invalid type in evaluation: {e}")
    except Exception as e:
        log_message(f"Unexpected error in evaluate_readiness: {e}", is_error=True)
        raise EvaluationCritical(f"Critical error in evaluate_readiness: {e}")

def evaluate_model(model, X_test, y_test, test_problems):
    try:
        log_message("Realizando predicciones")
        predictions = model.predict(X_test)
        
        log_message(f"Predicciones completadas. Shape: {predictions.shape}")

        # Calcular el error cuadrático medio para números complejos
        mse = np.mean(np.abs(y_test[:, 0] + 1j*y_test[:, 1] - (predictions[:, 0] + 1j*predictions[:, 1]))**2)
        log_message(f"Error cuadrático medio: {mse}")

        log_message("\nMuestra de predicciones:")
        sample_size = min(5, len(test_problems))
        for i in range(sample_size):
            log_message(f"Problema: {test_problems[i].problem}")
            log_message(f"Predicción: {predictions[i][0] + 1j*predictions[i][1]}")
            log_message(f"Solución real: {test_problems[i].solution}")
        
        return predictions, mse
    except Exception as e:
        log_message(f"Error al hacer predicciones: {str(e)}", is_error=True)
        return None, None