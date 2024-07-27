import numpy as np
import os
# Removido: os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Configuración de TensorFlow
os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import keras
from tensorflow import keras
import sympy as sp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import io
from keras import ops
from tensorflow.keras.utils import register_keras_serializable
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.ops.nn")

# Habilitar ejecución ansiosa
tf.config.run_functions_eagerly(True)

# Habilitar modo de depuración para tf.data
tf.data.experimental.enable_debug_mode()

# Constantes
VOCAB_SIZE = 1000
MAX_LENGTH = 50
MAX_TERMS = 50

class ExternalMemory:
    def __init__(self, memory_size=100, key_size=64, value_size=128):
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.keys = tf.Variable(tf.random.normal([memory_size, key_size], dtype=tf.float32))
        self.values = tf.Variable(tf.zeros([memory_size, value_size], dtype=tf.float32))
        self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))

    @tf.function
    def query(self, query_key):
        query_key = tf.cast(query_key, tf.float32)
        similarities = tf.matmul(query_key, self.keys, transpose_b=True)
        weights = tf.nn.sigmoid(similarities)
        return tf.matmul(weights, self.values)

    @tf.function
    def update(self, key, value):
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)
        key = tf.reshape(key, [-1, self.key_size])
        value = tf.reshape(value, [-1, self.value_size])

        index = tf.argmin(self.usage)

        self.keys[index].assign(key[0])
        self.values[index].assign(value[0])
        self.usage[index].assign(1.0)

        # Decay usage
        self.usage.assign(self.usage * 0.99)
    def inspect_memory(self):
        return {
            'keys': self.keys.numpy(),
            'values': self.values.numpy(),
            'usage': self.usage.numpy()
        }
class MathProblem:
    def __init__(self, problem, solution, difficulty, concept):
        self.problem = problem
        self.solution = solution
        self.difficulty = difficulty
        self.concept = concept

@register_keras_serializable()
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        super(Kistmat_AI, self).__init__(name=name, **kwargs)

        # Define layers
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=64)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True,
                                                kernel_regularizer=keras.regularizers.l2(0.01)))
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(1024,
                                                kernel_regularizer=keras.regularizers.l2(0.01)))
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(0.5)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)

        self.memory = ExternalMemory(key_size=64, value_size=128)
        self.memory_query = keras.layers.Dense(64, dtype='float32')

        self.reasoning_layer = keras.layers.Dense(512, activation='relu',
                                                  kernel_regularizer=keras.regularizers.l2(0.01))
        self.batch_norm3 = keras.layers.BatchNormalization()

        # Output layers for different learning stages
        self.output_layers = {
            'elementary1': keras.layers.Dense(128, activation='linear'),
            'elementary2': keras.layers.Dense(128, activation='linear'),
            'elementary3': keras.layers.Dense(128, activation='linear'),
            'junior_high1': keras.layers.Dense(128, activation='linear'),
            'junior_high2': keras.layers.Dense(128, activation='linear'),
            'high_school1': keras.layers.Dense(128, activation='linear'),
            'high_school2': keras.layers.Dense(128, activation='linear'),
            'high_school3': keras.layers.Dense(128, activation='linear'),
            'university': keras.layers.Dense(128, activation='linear')
        }
        self.final_output = keras.layers.Dense(output_shape, activation='linear')

        self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)

    def get_learning_stage(self):
        return self._learning_stage.numpy().decode()

    def set_learning_stage(self, stage):
        self._learning_stage.assign(stage.encode())

    @tf.function
    def call(self, inputs, training=False):
        current_stage = self.get_learning_stage()
        if current_stage == 'university':
            x = inputs
        else:
            x = self.embedding(inputs)
            x = self.lstm1(x)
            x = self.batch_norm1(x, training=training)
            x = self.lstm2(x)
            x = self.batch_norm2(x, training=training)

            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped)
            context = tf.squeeze(context, axis=1)

            query = self.memory_query(x)
            memory_output = self.memory.query(query)

            x = tf.concat([context, memory_output], axis=-1)

        x = self.reasoning_layer(x)
        x = self.batch_norm3(x, training=training)

        if training:
            x = self.dropout(x)

        x = self.output_layers[current_stage](x)

        if training and current_stage != 'university':
            self.memory.update(query, x)

        return self.final_output(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "vocab_size": self.vocab_size,
            "learning_stage": self.get_learning_stage()
        })
        return config
    def inspect_memory(self):
        return self.memory.inspect_memory()
    @classmethod
    def from_config(cls, config):
        input_shape = config.pop("input_shape", None)
        output_shape = config.pop("output_shape", None)
        vocab_size = config.pop("vocab_size", VOCAB_SIZE)
        learning_stage = config.pop("learning_stage", "elementary1")

        if input_shape is None:
            input_shape = (MAX_LENGTH,)
        if output_shape is None:
            output_shape = 1

        instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
        instance.set_learning_stage(learning_stage)
        return instance

class SymbolicReasoner:
    def __init__(self):
        self.symbols = {}
        self.rules = []

    def add_symbol(self, name):
        self.symbols[name] = sp.Symbol(name)

    def add_rule(self, rule):
        self.rules.append(rule)

    def apply_rules(self, expression):
        for rule in self.rules:
            expression = expression.replace(rule)
        return expression

    def simplify(self, expression):
        return sp.simplify(expression)

# Utility functions

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
        elif 'x' in term:
            coeff, exp = term.split('x')
            exp = '1'
        else:
            coeff, exp = term, '0'

        coeffs[i] = float(coeff) if coeff else 1
        exponents[i] = float(exp)

    coeffs = coeffs / np.max(np.abs(coeffs)) if np.max(np.abs(coeffs)) > 0 else coeffs
    exponents = exponents / np.max(exponents) if np.max(exponents) > 0 else exponents

    return np.pad(np.concatenate([coeffs, exponents]), (0, MAX_LENGTH - 2*max_terms))

def generate_dataset(num_problems, stage, difficulty):
    problems = []
    for _ in range(num_problems):
        if stage == 'elementary1':
            a, b = np.random.randint(1, int(10 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'elementary2':
            a, b = np.random.randint(1, int(20 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-', '*'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'elementary3':
            a, b = np.random.randint(1, int(30 * difficulty) + 1, size=2)
            op = np.random.choice(['+', '-', '*', '/'])
            problem = f"{a} {op} {b}"
            solution = complex(eval(problem))
            problems.append(MathProblem(problem, solution, difficulty, op))
        elif stage == 'junior_high1':
            a, b, c = np.random.randint(-int(10 * difficulty), int(10 * difficulty) + 1, size=3)
            if a == 0:
                a = 1
            problem = f"{a}x + {b} = {c}"
            solution = complex((c - b) / a)
            problems.append(MathProblem(problem, solution, difficulty, 'linear_equation'))
        elif stage == 'junior_high2':
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
        elif stage == 'high_school1':
            base = np.random.randint(2, 5)
            exponent = np.random.randint(1, int(5 * difficulty) + 1)
            problem = f"log_{base}(x) = {exponent}"
            solution = base ** exponent
            problems.append(MathProblem(problem, solution, difficulty, 'logarithm'))
        elif stage == 'high_school2':
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
        elif stage == 'high_school3':
            a = np.random.randint(1, int(3 * difficulty) + 1)
            problem = f"lim(x->0) (sin({a}x) / x)"
            solution = a
            problems.append(MathProblem(problem, solution, difficulty, 'limits'))
        elif stage == 'university':
            max_degree = max(1, int(3 * difficulty))  # Ensure max_degree is at least 1
            num_terms = np.random.randint(1, max_degree + 1)
            coeffs = np.random.randint(1, int(5 * difficulty) + 1, size=num_terms)
            exponents = np.random.randint(0, max_degree + 1, size=num_terms)

            problem_str = "d/dx ("
            solution = 0
            for coeff, exp in zip(coeffs, exponents):
                if exp == 0:
                    problem_str += f"{coeff} + "
                elif exp == 1:
                    problem_str += f"{coeff}x + "
                    solution += coeff
                else:
                    problem_str += f"{coeff}x^{exp} + "
                    solution += coeff * exp * (exp - 1)
            problem_str = problem_str.rstrip(" + ") + ")"

            problems.append(MathProblem(problem_str, solution, difficulty, 'derivatives'))
    return problems

def evaluate_readiness(model, problems, threshold):
    if model.get_learning_stage() == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
        if y.ndim == 1:
            y = np.column_stack((y, np.zeros_like(y)))
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))

    predictions = model.predict(X)
    mse = np.mean(np.square(y - predictions))
    r2 = r2_score(y, predictions)

    print(f"Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

    return r2

def train_model(model, problems, epochs=10):
    # Create a new optimizer instance for each training session
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    if model.get_learning_stage() == 'university':
        X_train = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y_train = np.array([p.solution for p in problems])
        if y_train.ndim == 1:
            y_train = np.column_stack((y_train, np.zeros_like(y_train)))
    else:
        X_train = np.array([tokenize_problem(p.problem) for p in problems])
        y_train_real = np.array([p.solution.real for p in problems])
        y_train_imag = np.array([p.solution.imag for p in problems])
        y_train = np.column_stack((y_train_real, y_train_imag))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )

    return history

def plot_learning_curves(stage_history, stage_name):
    plt.figure(figsize=(10, 6))

    history = stage_history['history']

    if isinstance(history, list):
        for i, h in enumerate(history):
            losses = h.history['loss']
            val_losses = h.history.get('val_loss', [])
            epochs = range(1, len(losses) + 1)

            plt.plot(epochs, losses, label=f'Training Loss (Session {i+1})')
            if val_losses:
                plt.plot(epochs, val_losses, label=f'Validation Loss (Session {i+1})')
    else:
        losses = history.history['loss']
        val_losses = history.history.get('val_loss', [])
        epochs = range(1, len(losses) + 1)

        plt.plot(epochs, losses, label='Training Loss')
        if val_losses:
            plt.plot(epochs, val_losses, label='Validation Loss')

    plt.title(f'Learning Curves - {stage_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs('output', exist_ok=True)

    figure_path = os.path.join('output', f'learning_curves_{stage_name}.png')
    plt.savefig(figure_path)
    plt.close()

    print(f"Learning curves for {stage_name} saved to {figure_path}")

def evaluate_model(model, problems, stage):
    if stage == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))

    predictions = model.predict(X)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if y.shape[1] == 1 and predictions.shape[1] == 2:
        # If actual output is real but prediction is complex
        y = np.column_stack((y, np.zeros_like(y)))
    elif y.shape[1] == 2 and predictions.shape[1] == 1:
        # If actual output is complex but prediction is real
        predictions = np.column_stack((predictions, np.zeros_like(predictions)))

    mse = np.mean(np.square(y - predictions))

    if y.shape[1] == 2:
        r2_real = r2_score(y[:, 0], predictions[:, 0])
        r2_imag = r2_score(y[:, 1], predictions[:, 1])
        r2 = (r2_real + r2_imag) / 2
    else:
        r2 = r2_score(y, predictions)

    return mse, r2

def test_knowledge_transfer(model, source_stage, target_stage, num_problems=100):
    source_problems = generate_dataset(num_problems, source_stage, difficulty=1.5)
    target_problems = generate_dataset(num_problems, target_stage, difficulty=1.5)

    source_mse, _ = evaluate_model(model, source_problems, source_stage)
    target_mse, _ = evaluate_model(model, target_problems, target_stage)

    transfer_ratio = source_mse / target_mse

    return transfer_ratio

def test_symbolic_consistency(model, num_problems=100):
    def generate_equivalent_problems():
        a, b = np.random.randint(1, 20, size=2)
        return [f"{a} + {b}", f"{b} + {a}", f"{a+b}"]

    consistent_count = 0

    for _ in range(num_problems):
        problems = generate_equivalent_problems()
        tokenized_problems = [tokenize_problem(p) for p in problems]
        predictions = model.predict(np.array(tokenized_problems))

        if np.allclose(predictions, predictions[0], atol=1e-5):
            consistent_count += 1

    return consistent_count / num_problems

def test_long_term_memory(model, stages, num_problems=50):
    memory_scores = {}

    for stage in stages:
        model.set_learning_stage(stage)
        problems = generate_dataset(num_problems, stage, difficulty=1.5)

        mse1, _ = evaluate_model(model, problems, stage)
        mse2, _ = evaluate_model(model, problems, stage)

        memory_score = (mse1 - mse2) / mse1
        memory_scores[stage] = memory_score

    return memory_scores

def test_concept_generalization(model, stage, num_problems=100):
    standard_problems = generate_dataset(num_problems, stage, difficulty=1.5)
    complex_problems = generate_dataset(num_problems, stage, difficulty=3.0)

    standard_mse, _ = evaluate_model(model, standard_problems, stage)
    complex_mse, _ = evaluate_model(model, complex_problems, stage)

    generalization_score = standard_mse / complex_mse

    return generalization_score

def smooth_curriculum_learning(model, stages, initial_problems=1000, max_problems=2000,
                               initial_difficulty=0.2, max_difficulty=5.0,
                               difficulty_increase_rate=0.5, difficulty_decrease_rate=0.2,
                               readiness_threshold=0.8, max_attempts_per_stage=5):
    all_history = []
    current_difficulty = initial_difficulty
    evaluation_results = {}

    readiness_thresholds = {
        'elementary1': 0.95, 'elementary2': 0.93, 'elementary3': 0.91,
        'junior_high1': 0.89, 'junior_high2': 0.87,
        'high_school1': 0.85, 'high_school2': 0.83, 'high_school3': 0.81,
        'university': 0.80
    }

    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Entering learning stage: {stage}")
        print(f"{'='*50}")
        model.set_learning_stage(stage)

        # Create a new model instance for each stage to reset the optimizer state
        model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=2)

        problems_solved = 0
        stage_history = []
        attempts = 0
        consecutive_improvements = 0

        while problems_solved < max_problems and attempts < max_attempts_per_stage:
            num_problems = min(initial_problems, max_problems - problems_solved)
            print(f"\nGenerating {num_problems} problems with difficulty {current_difficulty:.2f}")
            problems = generate_dataset(num_problems, stage, current_difficulty)

            print(f"Training on {len(problems)} problems...")
            history = train_model(model, problems, epochs=20)
            stage_history.append(history)

            val_problems = problems[-len(problems)//5:]  # Use last 20% as validation
            print("Evaluating model on validation set...")
            readiness_score = evaluate_readiness(model, val_problems, readiness_thresholds[stage])

            print(f"Current difficulty: {current_difficulty:.2f}, Readiness score: {readiness_score:.4f}")

            if readiness_score > readiness_thresholds[stage]:
                consecutive_improvements += 1
                current_difficulty = min(current_difficulty + difficulty_increase_rate, max_difficulty)
                print(f"Increasing difficulty. New difficulty: {current_difficulty:.2f}")

                if consecutive_improvements >= 3:
                    print(f"Model consistently improving. Moving to next stage.")
                    break
            else:
                consecutive_improvements = 0
                current_difficulty = max(current_difficulty * (1 - difficulty_decrease_rate), initial_difficulty)
                print(f"Decreasing difficulty. New difficulty: {current_difficulty:.2f}")
                attempts += 1

            problems_solved += num_problems

        if attempts == max_attempts_per_stage:
            print(f"Max attempts reached for {stage}. Moving to next stage.")

        model.save(f'model_{stage}.keras')

        stage_data = {
            'stage': stage,
            'history': stage_history,
            'final_difficulty': current_difficulty
        }
        all_history.append(stage_data)

        plot_learning_curves(stage_data, stage)

        print("\nRunning additional tests...")
        transfer_ratio = test_knowledge_transfer(model, stages[max(0, stages.index(stage)-1)], stage)
        consistency_score = test_symbolic_consistency(model)
        memory_score = test_long_term_memory(model, [stage])[stage]
        generalization_score = test_concept_generalization(model, stage)

        evaluation_results[stage] = {
            'readiness_score': readiness_score,
            'transfer_ratio': transfer_ratio,
            'consistency_score': consistency_score,
            'memory_score': memory_score,
            'generalization_score': generalization_score,
            'memory_snapshot': model.inspect_memory()
        }

        print(f"\nStage {stage} completed.")
        print(f"Final Readiness Score: {readiness_score:.4f}")
        print(f"Transfer Ratio: {transfer_ratio:.4f}")
        print(f"Consistency Score: {consistency_score:.4f}")
        print(f"Memory Score: {memory_score:.4f}")
        print(f"Generalization Score: {generalization_score:.4f}")

    return all_history, evaluation_results

def main():
    print("Initializing Kistmat_AI model...")
    model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=2)

    stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2',
              'high_school1', 'high_school2', 'high_school3', 'university']

    print("Starting curriculum learning process...")
    all_history, evaluation_results = smooth_curriculum_learning(
        model,
        stages,
        initial_problems=1000,
        max_problems=2000,
        initial_difficulty=0.2,
        max_difficulty=5.0,
        difficulty_increase_rate=0.5,
        difficulty_decrease_rate=0.2,
        readiness_threshold=0.8,
        max_attempts_per_stage=5
    )

    print("\nGenerating final test problems...")
    test_problems = generate_dataset(100, 'university', difficulty=5.0)

    print("Evaluating model on final test set...")
    test_mse, test_r2 = evaluate_model(model, test_problems, 'university')

    print("\nFinal Test Results:")
    print(f"Mean Squared Error: {test_mse:.4f}")
    print(f"R-squared: {test_r2:.4f}")

    print("\nSample predictions:")
    sample_size = 5
    for i in range(sample_size):
        problem = test_problems[i]
        X = np.array([tokenize_calculus_problem(problem.problem)])
        prediction = model.predict(X)[0][0]
        print(f"Problem: {problem.problem}")
        print(f"Prediction: {prediction:.4f}")
        print(f"Actual solution: {problem.solution:.4f}")
        print()

    print("\nDetailed Evaluation Results:")
    for stage, results in evaluation_results.items():
        print(f"\n{'='*30}")
        print(f"{stage.upper()}:")
        print(f"{'='*30}")
        print(f"Readiness Score: {results['readiness_score']:.4f}")
        print(f"Transfer Ratio: {results['transfer_ratio']:.4f}")
        print(f"Consistency Score: {results['consistency_score']:.4f}")
        print(f"Memory Score: {results['memory_score']:.4f}")
        print(f"Generalization Score: {results['generalization_score']:.4f}")

    print("\nSaving final model...")
    model.save('final_kistmat_ai_model.keras')
    print("Final model saved as 'final_kistmat_ai_model.keras'")

    print("\nGenerating learning curve plots...")
    for stage_data in all_history:
        plot_learning_curves(stage_data, stage_data['stage'])

    print("\nKistmat_AI training and evaluation complete.")

if __name__ == "__main__":
    main()