import numpy as np
import tensorflow as tf
from config.settings import VOCAB_SIZE, MAX_LENGTH, MAX_TERMS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenizationError(Exception):
    """Base exception class for tokenization-related errors."""
    pass

class TokenizationWarning(TokenizationError):
    """Exception class for non-critical tokenization warnings."""
    pass

class TokenizationCritical(TokenizationError):
    """Exception class for critical tokenization errors that prevent normal operation."""
    pass

def tokenize_problem(problem, stage, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    try:
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
    except ValueError as e:
        logger.error(f"Invalid stage type in tokenize_problem: {e}")
        raise TokenizationError(f"Invalid stage type: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in tokenize_problem: {e}")
        raise TokenizationCritical(f"Critical error in tokenize_problem: {e}")

def tokenize_basic_problem(problem, vocab_size, max_length):
    try:
        tokens = problem.lower().split()
        tokens = [hash(token) % vocab_size for token in tokens]
        return pad_tokens(tokens, max_length)
    except AttributeError as e:
        logger.error(f"Invalid problem type in tokenize_basic_problem: {e}")
        raise TokenizationError(f"Problem must be a string: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in tokenize_basic_problem: {e}")
        raise TokenizationCritical(f"Critical error in tokenize_basic_problem: {e}")

def tokenize_advanced_problem(problem, vocab_size, max_length):
    try:
        tokens = []
        for token in problem.replace('(', ' ( ').replace(')', ' ) ').split():
            if token.isalpha():
                tokens.append(hash(token) % vocab_size)
            elif token.isdigit() or token in '+-*/^()':
                tokens.append(ord(token))
            else:
                try:
                    tokens.append(int(float(token) * 100))  # Preservar 2 decimales
                except ValueError:
                    tokens.append(hash(token) % vocab_size)
        return pad_tokens(tokens, max_length)
    except AttributeError as e:
        logger.error(f"Invalid problem type in tokenize_advanced_problem: {e}")
        raise TokenizationError(f"Problem must be a string: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in tokenize_advanced_problem: {e}")
        raise TokenizationCritical(f"Critical error in tokenize_advanced_problem: {e}")

def tokenize_calculus_problem(problem, max_length):
    try:
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
        
        if not coeffs:
            raise ValueError("No valid terms found in the calculus problem")

        max_coeff = max(abs(c) for c in coeffs) or 1
        max_exp = max(exponents) or 1
        normalized_coeffs = [c / max_coeff for c in coeffs]
        normalized_exps = [e / max_exp for e in exponents]
        
        result = normalized_coeffs + normalized_exps
        return pad_tokens(result, max_length)
    except IndexError as e:
        logger.error(f"Invalid calculus problem format: {e}")
        raise TokenizationError(f"Calculus problem must contain 'd/dx': {e}")
    except ValueError as e:
        logger.error(f"Invalid term in calculus problem: {e}")
        raise TokenizationError(f"Invalid term in calculus problem: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in tokenize_calculus_problem: {e}")
        raise TokenizationCritical(f"Critical error in tokenize_calculus_problem: {e}")

def pad_tokens(tokens, max_length):
    try:
        if len(tokens) > max_length:
            logger.warning(f"Tokens truncated from {len(tokens)} to {max_length}")
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))
    except TypeError as e:
        logger.error(f"Invalid token type in pad_tokens: {e}")
        raise TokenizationError(f"Tokens must be a list: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in pad_tokens: {e}")
        raise TokenizationCritical(f"Critical error in pad_tokens: {e}")