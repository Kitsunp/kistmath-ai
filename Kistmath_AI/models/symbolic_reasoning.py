import tensorflow as tf
import numpy as np
import logging
from typing import Union, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SymbolicReasoning:
    def __init__(self):
        logger.info("Initializing SymbolicReasoning")

    def solve_linear_equation(self, coefficients: tf.Tensor, constants: tf.Tensor) -> tf.Tensor:
        try:
            logger.info("Attempting to solve linear equation")
            logger.debug(f"Coefficients: {coefficients}, Constants: {constants}")
            
            # Ensure inputs are tensors
            coefficients = tf.convert_to_tensor(coefficients, dtype=tf.float32)
            constants = tf.convert_to_tensor(constants, dtype=tf.float32)
            
            # Check shapes
            if coefficients.shape[-1] != constants.shape[-1]:
                raise ValueError("The last dimension of coefficients and constants must match")
            
            solution = tf.linalg.solve(coefficients, constants)
            logger.info("Linear equation solved successfully")
            logger.debug(f"Solution: {solution}")
            return solution
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid argument error in solve_linear_equation: {e}")
            raise ValueError(f"Invalid input to solve_linear_equation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in solve_linear_equation: {e}")
            raise

    def reason(self, input_data: tf.Tensor) -> tf.Tensor:
        try:
            logger.info("Performing symbolic reasoning")
            logger.debug(f"Input data: {input_data}")
            
            # Example reasoning: if input > 0, double it; else, halve it
            result = tf.where(input_data > 0, input_data * 2, input_data / 2)
            
            logger.info("Symbolic reasoning completed successfully")
            logger.debug(f"Reasoning result: {result}")
            return result
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid argument error in reason: {e}")
            raise ValueError(f"Invalid input to reason: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in reason: {e}")
            raise

    def solve_equation(self, equation: str) -> Union[float, List[float]]:
        try:
            logger.info(f"Attempting to solve equation: {equation}")
            
            # Simple equation parser (for demonstration purposes)
            parts = equation.split('=')
            if len(parts) != 2:
                raise ValueError("Equation must contain exactly one '=' sign")
            
            left, right = parts
            left = left.strip()
            right = right.strip()
            
            # Handle simple linear equations (ax + b = c)
            if 'x' in left:
                a, b = self._parse_linear(left)
                c = float(right)
                solution = (c - b) / a
            elif 'x' in right:
                a, b = self._parse_linear(right)
                c = float(left)
                solution = (c - b) / a
            else:
                raise ValueError("Equation must contain 'x'")
            
            logger.info(f"Equation solved successfully. Solution: {solution}")
            return solution
        except ValueError as e:
            logger.error(f"Value error in solve_equation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in solve_equation: {e}")
            raise

    def _parse_linear(self, expression: str) -> Tuple[float, float]:
        try:
            parts = expression.split('x')
            if len(parts) != 2:
                raise ValueError("Invalid linear expression")
            
            a = float(parts[0]) if parts[0] and parts[0] != '-' else (-1 if parts[0] == '-' else 1)
            b = float(parts[1]) if parts[1] else 0
            
            return a, b
        except ValueError as e:
            logger.error(f"Value error in _parse_linear: {e}")
            raise ValueError(f"Invalid linear expression: {expression}")
        except Exception as e:
            logger.error(f"Unexpected error in _parse_linear: {e}")
            raise