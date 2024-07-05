import numpy as np
import os
import tensorflow as tf

# Elimina la configuración forzada de CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configura TensorFlow para usar todos los núcleos de CPU disponibles
os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())

# Configura TensorFlow para usar la memoria de manera más eficiente
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Detecta si hay GPU disponible
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Intenta usar la primera GPU disponible
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Se detectaron {len(gpus)} GPUs físicas y {len(logical_gpus)} GPUs lógicas")
    except RuntimeError as e:
        # Error de memoria de GPU u otros problemas
        print(f"Error al configurar GPU: {e}")
        print("Usando CPU")
else:
    print("No se detectaron GPUs. Usando CPU")

# El resto de tus importaciones y configuraciones
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
from tensorflow.keras import ops
from tensorflow.keras.utils import register_keras_serializable
tf.keras.utils.register_keras_serializable(package='Custom', name=None)
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.ops.nn")

# Habilita la ejecución eager
tf.config.run_functions_eagerly(True)

# Habilita el modo de depuración para tf.data
tf.data.experimental.enable_debug_mode()

# Verifica qué dispositivo está siendo utilizado
print("Dispositivo que se está utilizando:", tf.device("/GPU:0" if gpus else "/CPU:0"))

# Constants
VOCAB_SIZE = 1000
MAX_LENGTH = 10
MAX_TERMS = 5

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
        
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=64)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
        self.dropout = keras.layers.Dropout(0.5)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)
        
        self.memory = ExternalMemory(key_size=64, value_size=128)
        self.memory_query = keras.layers.Dense(64, dtype='float32')
        
        self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
        
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

        self.dropout = keras.layers.Dropout(0.5)
        self.batch_norm = keras.layers.BatchNormalization()
        
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
            x = self.lstm2(x)
            
            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped)
            context = tf.squeeze(context, axis=1)
            
            query = self.memory_query(x)
            memory_output = self.memory.query(query)
            
            x = tf.concat([context, memory_output], axis=-1)
        
        x = self.reasoning_layer(x)
        
        if training:
            x = self.dropout(x)
        x = self.batch_norm(x)
        
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

    @classmethod
    def from_config(cls, config):
        # Extraer los argumentos necesarios de la configuración
        input_shape = config.pop("input_shape", None)
        output_shape = config.pop("output_shape", None)
        vocab_size = config.pop("vocab_size", VOCAB_SIZE)
        learning_stage = config.pop("learning_stage", "elementary1")
        
        # Si input_shape o output_shape no están en la configuración, usar valores por defecto
        if input_shape is None:
            input_shape = (MAX_LENGTH,)
        if output_shape is None:
            output_shape = 1
        
        # Crear una nueva instancia con los argumentos extraídos
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
    # Extract the function from the problem string
    func_str = problem.split("d/dx ")[1].strip("()")
    
    # Parse the function into terms
    terms = func_str.replace("-", "+-").split("+")
    
    # Initialize arrays for coefficients and exponents
    coeffs = coeffs / np.max(np.abs(coeffs)) if np.max(np.abs(coeffs)) > 0 else coeffs
    exponents = exponents / np.max(exponents) if np.max(exponents) > 0 else exponents
    
    # Ensure the output has the same shape as the input for other stages
    return np.pad(np.concatenate([coeffs, exponents]), (0, MAX_LENGTH - 2*max_terms))

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
            
            problems.append(MathProblem(problem_str, solution, difficulty, 'derivatives'))
    return problems

# Modify evaluate_readiness function
def evaluate_readiness(model, problems, threshold):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    if model.get_learning_stage() == 'university':
        X = np.array([tokenize_calculus_problem(p.problem) for p in problems])
        y = np.array([p.solution for p in problems])
    else:
        X = np.array([tokenize_problem(p.problem) for p in problems])
        y_real = np.array([p.solution.real for p in problems])
        y_imag = np.array([p.solution.imag for p in problems])
        y = np.column_stack((y_real, y_imag))
    
    predictions = model.predict(X)
    mse = np.mean(np.square(y - predictions))
    r2 = 1 - (np.sum(np.square(y - predictions)) / np.sum(np.square(y - np.mean(y))))
    
    return r2 > threshold

def train_fold(fold_data):
    model_config, model_weights, train_problems, val_problems, epochs = fold_data
    model = Kistmat_AI.from_config(model_config)
    model.set_weights(model_weights)
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
    
    if model.get_learning_stage() == 'university':
        X_train = np.array([tokenize_calculus_problem(p.problem) for p in train_problems])
        y_train = np.array([p.solution for p in train_problems])
    else:
        X_train = np.array([tokenize_problem(p.problem) for p in train_problems])
        y_train_real = np.array([p.solution.real for p in train_problems])
        y_train_imag = np.array([p.solution.imag for p in train_problems])
        y_train = np.column_stack((y_train_real, y_train_imag))
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=0)
    
    return {'history': history.history, 'weights': model.get_weights()}

def parallel_train_model(model, problems, epochs=10, n_folds=3):
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

# Modify reinforce_single function
def reinforce_single(args):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Ajusta la tasa de aprendizaje
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss.numpy()

def parallel_reinforce_learning(model, problems, predictions, true_solutions, learning_rate=0.01):
    reinforce_data = [(model, problem, prediction, true_solution) 
                      for problem, prediction, true_solution in zip(problems, predictions, true_solutions)]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        losses = pool.map(reinforce_single, reinforce_data)
    
    return losses

# Visualization functions
def plot_learning_curves(all_history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    stages = ['basic', 'algebra', 'precalculus', 'calculus']
    
    for i, stage in enumerate(stages):
        ax = axes[i // 2, i % 2]
        stage_history = next(h for h in all_history if h['stage'] == stage)
        
        losses = []
        maes = []
        for history in stage_history['fold_histories']:
            losses.extend(history['loss'])
            maes.extend(history['mae'])
        
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

def real_time_plotter(plot_queue):
    plt.switch_backend('agg')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs, losses, maes = [], [], []
    
    while True:
        try:
            data = plot_queue.get(timeout=1)
            if data is None:
                break
            
            epochs.append(data['epoch'])
            losses.append(data['loss'])
            maes.append(data['mae'])
            
            ax1.clear()
            ax2.clear()
            
            ax1.plot(epochs, losses, 'b-', label='Loss')
            ax2.plot(epochs, maes, 'g-', label='MAE')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            
            ax1.set_title(f"Epoch {data['epoch']}/{data['total_epochs']}")
            
            for i, example in enumerate(data['examples']):
                true_value = example['true']
                predicted_value = example['predicted']
                
                # Convert numpy arrays to strings, handling both scalar and array cases
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
            
            fig.canvas.draw()
            
            # Convert plot to PNG image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Save the plot
            with open(f'training_progress_epoch_{data["epoch"]}.png', 'wb') as f:
                f.write(buf.getvalue())
            
        except queue.Empty:
            pass
    
    plt.close(fig)

# Modify smooth_curriculum_learning function
def smooth_curriculum_learning(model, stages, initial_problems=4000, max_problems=5000, difficulty_increase_rate=0.05, plot_queue=None):
    all_history = []
    current_difficulty = 1.0

    readiness_thresholds = {
        'elementary1': 0.95,
        'elementary2': 0.93,
        'elementary3': 0.91,
        'junior_high1': 0.89,
        'junior_high2': 0.87,
        'high_school1': 0.85,
        'high_school2': 0.83,
        'high_school3': 0.81,
        'university': 0.80
    }

    for stage in stages:
        print(f"\nEntering learning stage: {stage}")
        model.set_learning_stage(stage)

        problems_solved = 0
        stage_history = []

        while problems_solved < max_problems:
            num_problems = min(initial_problems, max_problems - problems_solved)
            problems = generate_dataset(num_problems, stage, current_difficulty)

            fold_histories = parallel_train_model(model, problems, epochs=50)  # Aumenta el número de épocas
            
            # Update the model with the new weights from the last fold
            model.set_weights(fold_histories[-1]['weights'])
            
            stage_history.extend(fold_histories)

            val_problems = problems[-len(problems)//5:]  # Use last 20% as validation
            if evaluate_readiness(model, val_problems, readiness_thresholds[stage]):
                print("Model ready to advance!")
                current_difficulty += difficulty_increase_rate
                break

            problems_solved += num_problems

            if current_difficulty > 3.0 and stage != stages[-1]:
                print(f"Advancing to next stage: {stages[stages.index(stage) + 1]}")
                break

        all_history.append({
            'stage': stage,
            'fold_histories': stage_history
        })

        current_difficulty = max(1.0, current_difficulty - 0.5)

    return all_history

# Modify main function
def main():
    model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
              'high_school1', 'high_school2', 'high_school3', 'university']
    
    plot_queue = multiprocessing.Queue()
    plot_process = multiprocessing.Process(target=real_time_plotter, args=(plot_queue,))
    plot_process.start()
    
    all_history = None
    try:
        all_history = smooth_curriculum_learning(model, stages, plot_queue=plot_queue)
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        plot_queue.put(None)
        plot_process.join()
    
    if all_history:
        plot_learning_curves(all_history)
    
    # Generate test problems
    test_problems = generate_dataset(100, 'university', difficulty=2.0)
    X_test = np.array([tokenize_calculus_problem(p.problem) for p in test_problems])
    y_test = np.array([p.solution for p in test_problems])

    predictions = model.predict(X_test)

    print("\nTest Results:")
    mse = np.mean(np.square(y_test - predictions))
    print(f"Mean Squared Error: {mse}")

    print("\nSample predictions:")
    sample_size = 5
    for i in range(sample_size):
        print(f"Problem: {test_problems[i].problem}")
        print(f"Prediction: {predictions[i][0]}")
        print(f"Actual solution: {test_problems[i].solution}")
    
    losses = parallel_reinforce_learning(model, test_problems[:sample_size], 
                                         predictions[:sample_size], y_test[:sample_size])
    for i, loss in enumerate(losses):
        print(f"Reinforcement learning loss for problem {i+1}: {loss}")

if __name__ == "__main__":
    main()
