import numpy as np
import os
import tensorflow as tf
import keras
from tensorflow import keras
import sympy as sp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import io
import multiprocessing
import queue
import logging
from keras import ops
from tensorflow.keras.utils import register_keras_serializable
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.ops.nn")

try:
    # Remove forced CPU configuration
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Configure TensorFlow to use all available CPU cores
    os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())

    # Configure TensorFlow to use memory more efficiently
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Detect if GPU is available
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Try to use the first available GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f"Detected {len(gpus)} physical GPUs and {len(logical_gpus)} logical GPUs")
        except RuntimeError as e:
            # GPU memory error or other problems
            logger.error(f"Error configuring GPU: {e}")
            logger.info("Using CPU")
    else:
        logger.info("No GPUs detected. Using CPU")

    # Enable eager execution
    tf.config.run_functions_eagerly(True)

    # Enable debug mode for tf.data
    tf.data.experimental.enable_debug_mode()

    # Verify which device is being used
    logger.info(f"Device being used: {tf.device('/GPU:0' if gpus else '/CPU:0')}")

except Exception as e:
    logger.error(f"Error during TensorFlow configuration: {e}")
    raise

# Constants
VOCAB_SIZE = 1000
MAX_LENGTH = 10
MAX_TERMS = 5

class ExternalMemory:
    def __init__(self, memory_size=100, key_size=64, value_size=128):
        try:
            if not isinstance(memory_size, int) or memory_size <= 0:
                raise ValueError("memory_size must be a positive integer")
            if not isinstance(key_size, int) or key_size <= 0:
                raise ValueError("key_size must be a positive integer")
            if not isinstance(value_size, int) or value_size <= 0:
                raise ValueError("value_size must be a positive integer")
            
            self.memory_size = memory_size
            self.key_size = key_size
            self.value_size = value_size
            
            try:
                self.keys = tf.Variable(tf.random.normal([memory_size, key_size], dtype=tf.float32))
            except tf.errors.ResourceExhaustedError:
                logger.error("Not enough memory to allocate keys")
                raise MemoryError("Not enough memory to allocate keys")
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when creating keys: {e}")
                raise ValueError(f"Invalid argument when creating keys: {e}")
            
            try:
                self.values = tf.Variable(tf.zeros([memory_size, value_size], dtype=tf.float32))
            except tf.errors.ResourceExhaustedError:
                logger.error("Not enough memory to allocate values")
                raise MemoryError("Not enough memory to allocate values")
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when creating values: {e}")
                raise ValueError(f"Invalid argument when creating values: {e}")
            
            try:
                self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))
            except tf.errors.ResourceExhaustedError:
                logger.error("Not enough memory to allocate usage")
                raise MemoryError("Not enough memory to allocate usage")
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when creating usage: {e}")
                raise ValueError(f"Invalid argument when creating usage: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in ExternalMemory initialization: {e}")
            raise

    @tf.function
    def query(self, query_key):
        try:
            if not isinstance(query_key, tf.Tensor):
                raise TypeError("query_key must be a TensorFlow tensor")
            
            query_key = tf.cast(query_key, tf.float32)
            
            try:
                similarities = tf.matmul(query_key, self.keys, transpose_b=True)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument in query matmul operation: {e}")
                raise ValueError(f"Invalid argument in query matmul operation: {e}")
            except tf.errors.ResourceExhaustedError:
                logger.error("Not enough resources for query matmul operation")
                raise MemoryError("Not enough resources for query matmul operation")
            
            try:
                weights = tf.nn.sigmoid(similarities)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument in sigmoid operation: {e}")
                raise ValueError(f"Invalid argument in sigmoid operation: {e}")
            
            try:
                return tf.matmul(weights, self.values)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument in final matmul operation: {e}")
                raise ValueError(f"Invalid argument in final matmul operation: {e}")
            except tf.errors.ResourceExhaustedError:
                logger.error("Not enough resources for final matmul operation")
                raise MemoryError("Not enough resources for final matmul operation")
            
        except Exception as e:
            logger.error(f"Unexpected error in query operation: {e}")
            raise

    @tf.function
    def update(self, key, value):
        try:
            if not isinstance(key, tf.Tensor):
                raise TypeError("key must be a TensorFlow tensor")
            if not isinstance(value, tf.Tensor):
                raise TypeError("value must be a TensorFlow tensor")
            
            key = tf.cast(key, tf.float32)
            value = tf.cast(value, tf.float32)
            
            try:
                key = tf.reshape(key, [-1, self.key_size])
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when reshaping key: {e}")
                raise ValueError(f"Invalid argument when reshaping key: {e}")
            
            try:
                value = tf.reshape(value, [-1, self.value_size])
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when reshaping value: {e}")
                raise ValueError(f"Invalid argument when reshaping value: {e}")
            
            try:
                index = tf.argmin(self.usage)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when finding minimum usage: {e}")
                raise ValueError(f"Invalid argument when finding minimum usage: {e}")
            
            try:
                self.keys[index].assign(key[0])
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when assigning key: {e}")
                raise ValueError(f"Invalid argument when assigning key: {e}")
            except IndexError:
                logger.error("Index out of range when assigning key")
                raise IndexError("Index out of range when assigning key")
            
            try:
                self.values[index].assign(value[0])
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when assigning value: {e}")
                raise ValueError(f"Invalid argument when assigning value: {e}")
            except IndexError:
                logger.error("Index out of range when assigning value")
                raise IndexError("Index out of range when assigning value")
            
            try:
                self.usage[index].assign(1.0)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when assigning usage: {e}")
                raise ValueError(f"Invalid argument when assigning usage: {e}")
            except IndexError:
                logger.error("Index out of range when assigning usage")
                raise IndexError("Index out of range when assigning usage")
            
            try:
                self.usage.assign(self.usage * 0.99)
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Invalid argument when decaying usage: {e}")
                raise ValueError(f"Invalid argument when decaying usage: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in update operation: {e}")
            raise


class MathProblem:
    def __init__(self, problem, solution, difficulty, concept):
        try:
            if not isinstance(problem, str):
                raise TypeError("Problem must be a string")
            if not isinstance(solution, (int, float, complex, np.ndarray)):
                raise TypeError("Solution must be a number or numpy array")
            if not isinstance(difficulty, (int, float)):
                raise TypeError("Difficulty must be a number")
            if not isinstance(concept, str):
                raise TypeError("Concept must be a string")
            
            self.problem = problem
            self.solution = solution
            self.difficulty = difficulty
            self.concept = concept
        except Exception as e:
            logger.error(f"Error initializing MathProblem: {e}")
            raise

@register_keras_serializable()
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        try:
            super(Kistmat_AI, self).__init__(name=name, **kwargs)
            
            if not isinstance(input_shape, tuple):
                raise TypeError("input_shape must be a tuple")
            if not isinstance(output_shape, int):
                raise TypeError("output_shape must be an integer")
            if not isinstance(vocab_size, int) or vocab_size <= 0:
                raise ValueError("vocab_size must be a positive integer")
            
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.vocab_size = vocab_size
            
            # Define layers
            try:
                self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=64)
            except ValueError as e:
                logger.error(f"Error creating Embedding layer: {e}")
                raise ValueError(f"Invalid parameters for Embedding layer: {e}")
            
            try:
                self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
                self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
            except ValueError as e:
                logger.error(f"Error creating LSTM layers: {e}")
                raise ValueError(f"Invalid parameters for LSTM layers: {e}")
            
            try:
                self.dropout = keras.layers.Dropout(0.5)
            except ValueError as e:
                logger.error(f"Error creating Dropout layer: {e}")
                raise ValueError(f"Invalid dropout rate: {e}")
            
            try:
                self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)
            except ValueError as e:
                logger.error(f"Error creating MultiHeadAttention layer: {e}")
                raise ValueError(f"Invalid parameters for MultiHeadAttention layer: {e}")
            
            try:
                self.memory = ExternalMemory(key_size=64, value_size=128)
            except Exception as e:
                logger.error(f"Error creating ExternalMemory: {e}")
                raise RuntimeError(f"Failed to initialize ExternalMemory: {e}")
            
            try:
                self.memory_query = keras.layers.Dense(64, dtype='float32')
            except ValueError as e:
                logger.error(f"Error creating memory query Dense layer: {e}")
                raise ValueError(f"Invalid parameters for memory query Dense layer: {e}")
            
            try:
                self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
            except ValueError as e:
                logger.error(f"Error creating reasoning Dense layer: {e}")
                raise ValueError(f"Invalid parameters for reasoning Dense layer: {e}")
            
            # Output layers for different learning stages
            self.output_layers = {}
            stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                      'high_school1', 'high_school2', 'high_school3', 'university']
            try:
                for stage in stages:
                    self.output_layers[stage] = keras.layers.Dense(128, activation='linear')
            except ValueError as e:
                logger.error(f"Error creating output Dense layer for stage {stage}: {e}")
                raise ValueError(f"Invalid parameters for output Dense layer: {e}")
            
            try:
                self.final_output = keras.layers.Dense(output_shape, activation='linear')
            except ValueError as e:
                logger.error(f"Error creating final output Dense layer: {e}")
                raise ValueError(f"Invalid parameters for final output Dense layer: {e}")

            try:
                self.dropout = keras.layers.Dropout(0.5)
            except ValueError as e:
                logger.error(f"Error creating Dropout layer: {e}")
                raise ValueError(f"Invalid dropout rate: {e}")
            
            try:
                self.batch_norm = keras.layers.BatchNormalization()
            except ValueError as e:
                logger.error(f"Error creating BatchNormalization layer: {e}")
                raise ValueError(f"Invalid parameters for BatchNormalization layer: {e}")
            
            try:
                self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)
            except ValueError as e:
                logger.error(f"Error creating learning stage Variable: {e}")
                raise ValueError(f"Invalid parameters for learning stage Variable: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in Kistmat_AI initialization: {e}")
            raise
    
    def get_learning_stage(self):
        try:
            return self._learning_stage.numpy().decode()
        except tf.errors.OpError as e:
            logger.error(f"TensorFlow operation error in get_learning_stage: {e}")
            raise RuntimeError(f"Failed to get learning stage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_learning_stage: {e}")
            raise

    def set_learning_stage(self, stage):
        try:
            if not isinstance(stage, str):
                raise TypeError("Learning stage must be a string")
            if stage not in self.output_layers:
                raise ValueError(f"Invalid learning stage: {stage}")
            self._learning_stage.assign(stage.encode())
        except tf.errors.OpError as e:
            logger.error(f"TensorFlow operation error in set_learning_stage: {e}")
            raise RuntimeError(f"Failed to set learning stage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in set_learning_stage: {e}")
            raise
    
    @tf.function
    def call(self, inputs, training=False):
        try:
            current_stage = self.get_learning_stage()
            if current_stage == 'university':
                x = inputs
            else:
                try:
                    x = self.embedding(inputs)
                except ValueError as e:
                    logger.error(f"Error in embedding layer: {e}")
                    raise ValueError(f"Invalid input for embedding layer: {e}")
                
                try:
                    x = self.lstm1(x)
                    x = self.lstm2(x)
                except ValueError as e:
                    logger.error(f"Error in LSTM layers: {e}")
                    raise ValueError(f"Invalid input for LSTM layers: {e}")
                
                try:
                    x_reshaped = tf.expand_dims(x, axis=1)
                    context = self.attention(x_reshaped, x_reshaped)
                    context = tf.squeeze(context, axis=1)
                except tf.errors.InvalidArgumentError as e:
                    logger.error(f"Error in attention mechanism: {e}")
                    raise ValueError(f"Invalid input for attention mechanism: {e}")
                
                try:
                    query = self.memory_query(x)
                    memory_output = self.memory.query(query)
                except Exception as e:
                    logger.error(f"Error in memory query: {e}")
                    raise RuntimeError(f"Failed to query memory: {e}")
                
                try:
                    x = tf.concat([context, memory_output], axis=-1)
                except tf.errors.InvalidArgumentError as e:
                    logger.error(f"Error in concatenation: {e}")
                    raise ValueError(f"Invalid shapes for concatenation: {e}")
            
            try:
                x = self.reasoning_layer(x)
            except ValueError as e:
                logger.error(f"Error in reasoning layer: {e}")
                raise ValueError(f"Invalid input for reasoning layer: {e}")
            
            if training:
                try:
                    x = self.dropout(x)
                except ValueError as e:
                    logger.error(f"Error in dropout layer: {e}")
                    raise ValueError(f"Invalid input for dropout layer: {e}")
            
            try:
                x = self.batch_norm(x)
            except ValueError as e:
                logger.error(f"Error in batch normalization: {e}")
                raise ValueError(f"Invalid input for batch normalization: {e}")
            
            try:
                x = self.output_layers[current_stage](x)
            except KeyError:
                logger.error(f"Invalid learning stage: {current_stage}")
                raise ValueError(f"Invalid learning stage: {current_stage}")
            except ValueError as e:
                logger.error(f"Error in output layer: {e}")
                raise ValueError(f"Invalid input for output layer: {e}")
            
            if training and current_stage != 'university':
                try:
                    self.memory.update(query, x)
                except Exception as e:
                    logger.error(f"Error updating memory: {e}")
                    raise RuntimeError(f"Failed to update memory: {e}")
            
            try:
                return self.final_output(x)
            except ValueError as e:
                logger.error(f"Error in final output layer: {e}")
                raise ValueError(f"Invalid input for final output layer: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error in call method: {e}")
            raise

    def get_config(self):
        try:
            config = super().get_config()
            config.update({
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "vocab_size": self.vocab_size,
                "learning_stage": self.get_learning_stage()
            })
            return config
        except Exception as e:
            logger.error(f"Error in get_config: {e}")
            raise RuntimeError(f"Failed to get model configuration: {e}")

    @classmethod
    def from_config(cls, config):
        try:
            input_shape = config.pop("input_shape", None)
            output_shape = config.pop("output_shape", None)
            vocab_size = config.pop("vocab_size", VOCAB_SIZE)
            learning_stage = config.pop("learning_stage", "elementary1")
            
            if input_shape is None:
                raise ValueError("input_shape is missing from config")
            if output_shape is None:
                raise ValueError("output_shape is missing from config")
            
            instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
            instance.set_learning_stage(learning_stage)
            return instance
        except KeyError as e:
            logger.error(f"Missing key in config: {e}")
            raise ValueError(f"Missing key in config: {e}")
        except Exception as e:
            logger.error(f"Error in from_config: {e}")
            raise RuntimeError(f"Failed to create model from configuration: {e}")

class SymbolicReasoner:
    def __init__(self):
        self.symbols = {}
        self.rules = []
    
    def add_symbol(self, name):
        try:
            if not isinstance(name, str):
                raise TypeError("Symbol name must be a string")
            if name in self.symbols:
                raise ValueError(f"Symbol '{name}' already exists")
            self.symbols[name] = sp.Symbol(name)
        except Exception as e:
            logger.error(f"Error in add_symbol: {e}")
            raise
    
    def add_rule(self, rule):
        try:
            if not callable(rule):
                raise TypeError("Rule must be a callable")
            self.rules.append(rule)
        except Exception as e:
            logger.error(f"Error in add_rule: {e}")
            raise
    
    def apply_rules(self, expression):
        try:
            if not isinstance(expression, sp.Expr):
                raise TypeError("Expression must be a SymPy expression")
            for rule in self.rules:
                try:
                    expression = expression.replace(rule)
                except Exception as e:
                    logger.warning(f"Error applying rule {rule}: {e}")
            return expression
        except Exception as e:
            logger.error(f"Error in apply_rules: {e}")
            raise
    
    def simplify(self, expression):
        try:
            if not isinstance(expression, sp.Expr):
                raise TypeError("Expression must be a SymPy expression")
            return sp.simplify(expression)
        except sp.SympifyError as e:
            logger.error(f"Error simplifying expression: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in simplify: {e}")
            raise


def tokenize_problem(problem, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    try:
        if not isinstance(problem, str):
            raise TypeError("Problem must be a string")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        
        try:
            tokens = problem.lower().split()
        except AttributeError:
            logger.error("Problem is not a valid string")
            raise ValueError("Problem is not a valid string")
        
        try:
            tokens = [hash(token) % vocab_size for token in tokens]
        except TypeError:
            logger.error("Invalid token type encountered")
            raise ValueError("Invalid token type encountered")
        
        tokens = tokens[:max_length]
        tokens += [0] * (max_length - len(tokens))
        
        return tokens
    except Exception as e:
        logger.error(f"Unexpected error in tokenize_problem: {e}")
        raise

def tokenize_calculus_problem(problem, max_terms=MAX_TERMS):
    try:
        if not isinstance(problem, str):
            raise TypeError("Problem must be a string")
        if not isinstance(max_terms, int) or max_terms <= 0:
            raise ValueError("max_terms must be a positive integer")
        
        try:
            func_str = problem.split("d/dx ")[1].strip("()")
        except IndexError:
            logger.error("Invalid problem format: missing 'd/dx'")
            raise ValueError("Invalid problem format: missing 'd/dx'")
        
        terms = func_str.replace("-", "+-").split("+")
        
        coeffs = np.zeros(max_terms)
        exponents = np.zeros(max_terms)
        
        for i, term in enumerate(terms[:max_terms]):
            try:
                if 'x' in term:
                    parts = term.split('x')
                    coeff = float(parts[0]) if parts[0] and parts[0] != '-' else (-1 if parts[0] == '-' else 1)
                    exp = float(parts[1][1:]) if parts[1] else 1
                else:
                    coeff = float(term)
                    exp = 0
            except ValueError:
                logger.error(f"Invalid term format: {term}")
                raise ValueError(f"Invalid term format: {term}")
            except IndexError:
                logger.error(f"Invalid term structure: {term}")
                raise ValueError(f"Invalid term structure: {term}")
            
            coeffs[i] = coeff
            exponents[i] = exp
        
        try:
            max_coeff = np.max(np.abs(coeffs))
            if max_coeff > 0:
                coeffs = coeffs / max_coeff
            
            max_exp = np.max(exponents)
            if max_exp > 0:
                exponents = exponents / max_exp
        except FloatingPointError:
            logger.error("Floating point error during normalization")
            raise ArithmeticError("Floating point error during normalization")
        except ValueError:
            logger.error("Error during array operations")
            raise ValueError("Error during array operations")
        
        try:
            return np.pad(np.concatenate([coeffs, exponents]), (0, MAX_LENGTH - 2*max_terms))
        except ValueError:
            logger.error("Error during array concatenation or padding")
            raise ValueError("Error during array concatenation or padding")
    except Exception as e:
        logger.error(f"Unexpected error in tokenize_calculus_problem: {e}")
        raise

def generate_dataset(num_problems, stage, difficulty):
    try:
        if not isinstance(num_problems, int) or num_problems <= 0:
            raise ValueError("num_problems must be a positive integer")
        if not isinstance(stage, str):
            raise TypeError("stage must be a string")
        if not isinstance(difficulty, (int, float)) or difficulty <= 0:
            raise ValueError("difficulty must be a positive number")

        problems = []
        if stage == 'elementary1':  # 1st-2nd grade
            for _ in range(num_problems):
                try:
                    a, b = np.random.randint(1, int(10 * difficulty), size=2)
                    op = np.random.choice(['+', '-'])
                    problem = f"{a} {op} {b}"
                    solution = complex(eval(problem))
                    problems.append(MathProblem(problem, solution, difficulty, op))
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'elementary2':  # 3rd-4th grade
            for _ in range(num_problems):
                try:
                    a, b = np.random.randint(1, int(20 * difficulty), size=2)
                    op = np.random.choice(['+', '-', '*'])
                    problem = f"{a} {op} {b}"
                    solution = complex(eval(problem))
                    problems.append(MathProblem(problem, solution, difficulty, op))
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'elementary3':  # 5th-6th grade
            for _ in range(num_problems):
                try:
                    a, b = np.random.randint(1, int(30 * difficulty), size=2)
                    op = np.random.choice(['+', '-', '*', '/'])
                    problem = f"{a} {op} {b}"
                    solution = complex(eval(problem))
                    problems.append(MathProblem(problem, solution, difficulty, op))
                except ZeroDivisionError:
                    logger.warning("Division by zero, skipping this problem")
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'junior_high1':  # 7th-8th grade
            for _ in range(num_problems):
                try:
                    a, b, c = np.random.randint(-int(10 * difficulty), int(10 * difficulty) + 1, size=3)
                    if a == 0:
                        a = 1
                    problem = f"{a}x + {b} = {c}"
                    solution = complex((c - b) / a)
                    problems.append(MathProblem(problem, solution, difficulty, 'linear_equation'))
                except ZeroDivisionError:
                    logger.warning("Division by zero, skipping this problem")
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'junior_high2':  # 9th grade
            for _ in range(num_problems):
                try:
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
                except ZeroDivisionError:
                    logger.warning("Division by zero, skipping this problem")
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'high_school1':  # 10th grade
            for _ in range(num_problems):
                try:
                    base = np.random.randint(2, 5)
                    exponent = np.random.randint(1, int(5 * difficulty))
                    problem = f"log_{base}(x) = {exponent}"
                    solution = base ** exponent
                    problems.append(MathProblem(problem, solution, difficulty, 'logarithm'))
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'high_school2':  # 11th grade
            for _ in range(num_problems):
                try:
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
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'high_school3':  # 12th grade
            for _ in range(num_problems):
                try:
                    a = np.random.randint(1, int(3 * difficulty))
                    problem = f"lim(x->0) (sin({a}x) / x)"
                    solution = a
                    problems.append(MathProblem(problem, solution, difficulty, 'limits'))
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        elif stage == 'university':  # University level
            for _ in range(num_problems):
                try:
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
                    
                    problems.append(MathProblem(problem_str, solution, difficulty, 'derivatives'))
                except OverflowError:
                    logger.warning("Overflow occurred, skipping this problem")
                except ValueError:
                    logger.warning("Invalid value encountered, skipping this problem")
                except TypeError as e:
                    logger.warning(f"Type error occurred: {e}, skipping this problem")
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        if not problems:
            raise ValueError("No valid problems generated")
        
        return problems
    except ValueError as e:
        logger.error(f"Value error in generate_dataset: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error in generate_dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_dataset: {e}")
        raise

def evaluate_readiness(model, problems, threshold):
    try:
        if not isinstance(model, Kistmat_AI):
            raise TypeError("model must be an instance of Kistmat_AI")
        if not isinstance(problems, list) or not all(isinstance(p, MathProblem) for p in problems):
            raise TypeError("problems must be a list of MathProblem instances")
        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1")

        try:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise

        try:
            if model.get_learning_stage() == 'university':
                X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
                y = np.array([p.solution for p in problems])
            else:
                X = np.array([tokenize_problem(p.problem) for p in problems])
                y_real = np.array([p.solution.real for p in problems])
                y_imag = np.array([p.solution.imag for p in problems])
                y = np.column_stack((y_real, y_imag))
        except Exception as e:
            logger.error(f"Error preparing data for evaluation: {e}")
            raise
        
        try:
            predictions = model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
        
        try:
            mse = np.mean(np.square(y - predictions))
            r2 = 1 - (np.sum(np.square(y - predictions)) / np.sum(np.square(y - np.mean(y))))
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
        
        return r2 > threshold
    except Exception as e:
        logger.error(f"Error in evaluate_readiness: {e}")
        raise

def train_fold(fold_data):
    try:
        model_config, model_weights, train_problems, val_problems, epochs = fold_data
        
        try:
            model = Kistmat_AI.from_config(model_config)
        except Exception as e:
            logger.error(f"Error creating model from config: {e}")
            raise
        
        try:
            model.set_weights(model_weights)
        except ValueError as e:
            logger.error(f"Error setting model weights: {e}")
            raise
        
        try:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise
        
        try:
            if model.get_learning_stage() == 'university':
                X_train = np.array([tokenize_calculus_problem(p.problem) for p in train_problems])
                y_train = np.array([p.solution for p in train_problems])
            else:
                X_train = np.array([tokenize_problem(p.problem) for p in train_problems])
                y_train_real = np.array([p.solution.real for p in train_problems])
                y_train_imag = np.array([p.solution.imag for p in train_problems])
                y_train = np.column_stack((y_train_real, y_train_imag))
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
        
        try:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=0)
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
        
        return {'history': history.history, 'weights': model.get_weights()}
    except Exception as e:
        logger.error(f"Error in train_fold: {e}")
        raise

def parallel_train_model(model, problems, epochs=10, n_folds=3):
    try:
        if not isinstance(model, Kistmat_AI):
            raise ValueError("model must be an instance of Kistmat_AI")
        if not isinstance(problems, list) or not all(isinstance(p, MathProblem) for p in problems):
            raise ValueError("problems must be a list of MathProblem instances")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(n_folds, int) or n_folds <= 1:
            raise ValueError("n_folds must be an integer greater than 1")

        kf = KFold(n_splits=n_folds)
        fold_data = []
        
        model_config = model.get_config()
        model_weights = model.get_weights()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        for train_index, val_index in kf.split(problems):
            train_problems = [problems[i] for i in train_index]
            val_problems = [problems[i] for i in val_index]
            fold_data.append((model_config, model_weights, train_problems, val_problems, epochs))
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            fold_histories = pool.map(train_fold, fold_data)
        
        return fold_histories
    except Exception as e:
        logger.error(f"Error in parallel_train_model: {e}")
        raise

def reinforce_single(args):
    try:
        model, problem, prediction, true_solution = args
        if model.get_learning_stage() == 'university':
            inputs = tf.convert_to_tensor([tokenize_calculus_problem(problem.problem)])
            true_solution_tensor = tf.constant([[true_solution]], dtype=tf.float32)
        else:
            inputs = tf.convert_to_tensor([tokenize_problem(problem.problem)])
            true_solution_tensor = tf.constant([[true_solution.real, true_solution.imag]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predicted = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(true_solution_tensor - predicted))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss.numpy()
    except Exception as e:
        logger.error(f"Error in reinforce_single: {e}")
        raise

def parallel_reinforce_learning(model, problems, predictions, true_solutions, learning_rate=0.01):
    try:
        if not isinstance(model, Kistmat_AI):
            raise ValueError("model must be an instance of Kistmat_AI")
        if not isinstance(problems, list) or not all(isinstance(p, MathProblem) for p in problems):
            raise ValueError("problems must be a list of MathProblem instances")
        if not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a numpy array")
        if not isinstance(true_solutions, np.ndarray):
            raise ValueError("true_solutions must be a numpy array")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float")

        reinforce_data = [(model, problem, prediction, true_solution) 
                          for problem, prediction, true_solution in zip(problems, predictions, true_solutions)]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            losses = pool.map(reinforce_single, reinforce_data)
        
        return losses
    except Exception as e:
        logger.error(f"Error in parallel_reinforce_learning: {e}")
        raise

def plot_learning_curves(all_history):
    try:
        if not isinstance(all_history, list):
            raise ValueError("all_history must be a list")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        stages = ['basic', 'algebra', 'precalculus', 'calculus']
        
        for i, stage in enumerate(stages):
            ax = axes[i // 2, i % 2]
            stage_history = next((h for h in all_history if h['stage'] == stage), None)
            
            if stage_history is None:
                logger.warning(f"No history found for stage {stage}")
                continue
            
            losses = []
            maes = []
            for history in stage_history['fold_histories']:
                losses.extend(history['history']['loss'])
                maes.extend(history['history']['mae'])
            
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, label='Loss')
            ax.plot(epochs, maes, label='MAE')
            
            ax.set_title(f'{stage.capitalize()} Stage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric Value')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error in plot_learning_curves: {e}")
        raise

def real_time_plotter(plot_queue):
    """
    Plots real-time training progress with extensive error handling.
    
    Args:
    plot_queue: A multiprocessing Queue containing plot data
    """
    try:
        plt.switch_backend('agg')
    except ImportError as e:
        logger.error(f"Failed to switch matplotlib backend: {e}")
        return
    except RuntimeError as e:
        logger.error(f"Runtime error when switching matplotlib backend: {e}")
        return

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    except ValueError as e:
        logger.error(f"Invalid figure size: {e}")
        return
    except RuntimeError as e:
        logger.error(f"Failed to create subplots: {e}")
        return

    epochs, losses, maes = [], [], []
    
    while True:
        try:
            data = plot_queue.get(timeout=1)
            if data is None:
                break
            
            try:
                epochs.append(data['epoch'])
                losses.append(data['loss'])
                maes.append(data['mae'])
            except KeyError as e:
                logger.error(f"Missing key in data: {e}")
                continue
            except TypeError as e:
                logger.error(f"Invalid data type in queue: {e}")
                continue

            try:
                ax1.clear()
                ax2.clear()
            except ValueError as e:
                logger.error(f"Failed to clear axes: {e}")
                continue
            
            try:
                ax1.plot(epochs, losses, 'b-', label='Loss')
                ax2.plot(epochs, maes, 'g-', label='MAE')
            except ValueError as e:
                logger.error(f"Invalid data for plotting: {e}")
                continue

            try:
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
            except ValueError as e:
                logger.error(f"Failed to set labels or legend: {e}")
            
            try:
                ax1.set_title(f"Epoch {data['epoch']}/{data['total_epochs']}")
            except KeyError as e:
                logger.error(f"Missing 'total_epochs' key: {e}")
            
            try:
                for i, example in enumerate(data['examples']):
                    true_value = example['true']
                    predicted_value = example['predicted']
                    
                    if isinstance(true_value, np.ndarray):
                        true_str = f"[{', '.join(f'{v:.4f}' for v in true_value)}]"
                    else:
                        true_str = f"{true_value:.4f}"
                    
                    if isinstance(predicted_value, np.ndarray):
                        pred_str = f"[{', '.join(f'{v:.4f}' for v in predicted_value)}]"
                    else:
                        pred_str = f"{predicted_value:.4f}"
                    
                    ax2.text(0.05, 0.9 - i*0.15, 
                             f"Problem: {example['problem']}\nTrue: {true_str}, Predicted: {pred_str}", 
                             transform=ax2.transAxes, fontsize=8, verticalalignment='top')
            except KeyError as e:
                logger.error(f"Missing key in example data: {e}")
            except ValueError as e:
                logger.error(f"Invalid value in example data: {e}")
            except IndexError as e:
                logger.error(f"Index out of range in example data: {e}")
            
            try:
                fig.canvas.draw()
            except RuntimeError as e:
                logger.error(f"Failed to draw canvas: {e}")
                continue
            
            try:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
            except ValueError as e:
                logger.error(f"Invalid parameters for saving figure: {e}")
                continue
            except RuntimeError as e:
                logger.error(f"Failed to save figure to buffer: {e}")
                continue
            
            try:
                with open(f'training_progress_epoch_{data["epoch"]}.png', 'wb') as f:
                    f.write(buf.getvalue())
            except IOError as e:
                logger.error(f"Failed to write plot to file: {e}")
            except KeyError as e:
                logger.error(f"Missing 'epoch' key when saving file: {e}")
            
        except queue.Empty:
            logger.debug("Queue is empty, continuing...")
        except queue.Full:
            logger.warning("Queue is full, skipping this iteration")
        except AttributeError as e:
            logger.error(f"Attribute error in plotting: {e}")
        except TypeError as e:
            logger.error(f"Type error in plotting: {e}")
        except ValueError as e:
            logger.error(f"Value error in plotting: {e}")
        except MemoryError as e:
            logger.critical(f"Out of memory: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error in plotting: {e}")
    
    try:
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to close figure: {e}")

def smooth_curriculum_learning(model, stages, initial_problems=4000, max_problems=5000, difficulty_increase_rate=0.05, plot_queue=None):
    """
    Implements smooth curriculum learning for the Kistmat AI model with extensive error handling.
    """
    if not isinstance(model, Kistmat_AI):
        raise TypeError("model must be an instance of Kistmat_AI")
    if not isinstance(stages, list) or not all(isinstance(s, str) for s in stages):
        raise ValueError("stages must be a list of strings")
    if not isinstance(initial_problems, int) or initial_problems <= 0:
        raise ValueError("initial_problems must be a positive integer")
    if not isinstance(max_problems, int) or max_problems <= 0:
        raise ValueError("max_problems must be a positive integer")
    if not isinstance(difficulty_increase_rate, float) or difficulty_increase_rate <= 0:
        raise ValueError("difficulty_increase_rate must be a positive float")
    if plot_queue is not None and not isinstance(plot_queue, multiprocessing.Queue):
        raise TypeError("plot_queue must be a multiprocessing.Queue or None")

    all_history = []
    current_difficulty = 1.0

    readiness_thresholds = {
        'elementary1': 0.95, 'elementary2': 0.93, 'elementary3': 0.91,
        'junior_high1': 0.89, 'junior_high2': 0.87,
        'high_school1': 0.85, 'high_school2': 0.83, 'high_school3': 0.81,
        'university': 0.80
    }

    for stage in stages:
        try:
            logger.info(f"Entering learning stage: {stage}")
            model.set_learning_stage(stage)
        except ValueError as e:
            logger.error(f"Invalid learning stage: {e}")
            continue

        problems_solved = 0
        stage_history = []

        while problems_solved < max_problems:
            try:
                num_problems = min(initial_problems, max_problems - problems_solved)
                problems = generate_dataset(num_problems, stage, current_difficulty)
            except ValueError as e:
                logger.error(f"Failed to generate dataset: {e}")
                break

            try:
                fold_histories = parallel_train_model(model, problems, epochs=50)
                model.set_weights(fold_histories[-1]['weights'])
                stage_history.extend(fold_histories)
            except Exception as e:
                logger.error(f"Error during parallel training: {e}")
                break

            try:
                val_problems = problems[-len(problems)//5:]
                if evaluate_readiness(model, val_problems, readiness_thresholds[stage]):
                    logger.info("Model ready to advance!")
                    current_difficulty += difficulty_increase_rate
                    break
            except KeyError as e:
                logger.error(f"Invalid stage in readiness_thresholds: {e}")
            except Exception as e:
                logger.error(f"Error during readiness evaluation: {e}")

            problems_solved += num_problems

            if current_difficulty > 3.0 and stage != stages[-1]:
                logger.info(f"Advancing to next stage: {stages[stages.index(stage) + 1]}")
                break

            if plot_queue is not None:
                try:
                    plot_data = {
                        'epoch': problems_solved,
                        'total_epochs': max_problems,
                        'loss': fold_histories[-1]['history']['loss'][-1],
                        'mae': fold_histories[-1]['history']['mae'][-1],
                        'examples': [
                            {'problem': p.problem, 'true': p.solution, 'predicted': model.predict(np.array([tokenize_problem(p.problem)]))[0]}
                            for p in val_problems[:3]
                        ]
                    }
                    plot_queue.put(plot_data)
                except Exception as e:
                    logger.error(f"Error preparing plot data: {e}")

        all_history.append({
            'stage': stage,
            'fold_histories': stage_history
        })

        current_difficulty = max(1.0, current_difficulty - 0.5)

    return all_history

def main():
    """
    Main function to run the Kistmat AI training process with extensive error handling.
    """
    try:
        model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    except ValueError as e:
        logger.critical(f"Failed to initialize or compile model: {e}")
        return
    except RuntimeError as e:
        logger.critical(f"Runtime error during model initialization: {e}")
        return

    stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
              'high_school1', 'high_school2', 'high_school3', 'university']
    
    try:
        plot_queue = multiprocessing.Queue()
        plot_process = multiprocessing.Process(target=real_time_plotter, args=(plot_queue,))
        plot_process.start()
    except RuntimeError as e:
        logger.error(f"Failed to start plotting process: {e}")
        plot_queue = None
        plot_process = None
    
    all_history = None
    try:
        all_history = smooth_curriculum_learning(model, stages, plot_queue=plot_queue)
    except Exception as e:
        logger.critical(f"An error occurred during training: {e}")
    finally:
        if plot_queue is not None:
            try:
                plot_queue.put(None)
                plot_process.join(timeout=10)
                if plot_process.is_alive():
                    plot_process.terminate()
            except Exception as e:
                logger.error(f"Error shutting down plotting process: {e}")
    
    if all_history:
        try:
            plot_learning_curves(all_history)
        except Exception as e:
            logger.error(f"Failed to plot learning curves: {e}")
    
    try:
        test_problems = generate_dataset(100, 'university', difficulty=2.0)
        X_test = np.array([tokenize_calculus_problem(p.problem) for p in test_problems])
        y_test = np.array([p.solution for p in test_problems])
    except Exception as e:
        logger.error(f"Failed to generate test dataset: {e}")
        return

    try:
        predictions = model.predict(X_test)
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        return

    print("\nTest Results:")
    try:
        mse = np.mean(np.square(y_test - predictions))
        print(f"Mean Squared Error: {mse}")
    except Exception as e:
        logger.error(f"Failed to calculate MSE: {e}")

    print("\nSample predictions:")
    sample_size = min(5, len(test_problems))
    for i in range(sample_size):
        try:
            print(f"Problem: {test_problems[i].problem}")
            print(f"Prediction: {predictions[i][0]}")
            print(f"Actual solution: {test_problems[i].solution}")
        except IndexError as e:
            logger.error(f"Index error when printing sample predictions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when printing sample predictions: {e}")
    
    try:
        losses = parallel_reinforce_learning(model, test_problems[:sample_size], 
                                             predictions[:sample_size], y_test[:sample_size])
        for i, loss in enumerate(losses):
            print(f"Reinforcement learning loss for problem {i+1}: {loss}")
    except Exception as e:
        logger.error(f"Failed during reinforcement learning: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}")