import numpy as np
from utils.tokenization import tokenize_problem, tokenize_calculus_problem
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            logger.warning(f"Predictions shape {predictions.shape} doesn't match y shape {y.shape}. Squeezing predictions.")
            predictions = predictions.squeeze()
        
        if predictions.shape != y.shape:
            raise ValueError(f"Predictions shape {predictions.shape} doesn't match y shape {y.shape} after squeezing")
        
        mse = np.mean(np.square(y - predictions))
        r2 = 1 - (np.sum(np.square(y - predictions)) / np.sum(np.square(y - np.mean(y))))
        
        logger.info(f"Evaluation results: MSE = {mse}, R2 = {r2}")
        return r2 > threshold

    except AttributeError as e:
        logger.error(f"Invalid model or problem attribute: {e}")
        raise EvaluationError(f"Invalid model or problem attribute: {e}")
    except ValueError as e:
        logger.error(f"Shape mismatch or invalid value: {e}")
        raise EvaluationError(f"Shape mismatch or invalid value: {e}")
    except TypeError as e:
        logger.error(f"Invalid type in evaluation: {e}")
        raise EvaluationError(f"Invalid type in evaluation: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in evaluate_readiness: {e}")
        raise EvaluationCritical(f"Critical error in evaluate_readiness: {e}")