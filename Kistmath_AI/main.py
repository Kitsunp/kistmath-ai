import os
import tensorflow as tf
import numpy as np
import multiprocessing
import traceback
from datetime import datetime
from models.kistmat_ai import Kistmat_AI
from models.external_memory import IntegratedMemorySystem
from training.curriculum_learning import smooth_curriculum_learning
from utils.data_generation import generate_dataset
from utils.evaluation import evaluate_readiness
from utils.tokenization import tokenize_problem
from training.parallel_training import parallel_reinforce_learning
from visualization.plotting import plot_learning_curves, real_time_plotter
from config.settings import STAGES, MAX_LENGTH, READINESS_THRESHOLDS

# Suprimir mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def log_message(message, is_error=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "ERROR" if is_error else "INFO"
    with open("program_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] {prefix}: {message}\n")

def safe_tokenize_problem(problem, learning_stage):
    try:
        tokenized = tokenize_problem(problem, learning_stage)
        log_message(f"Problema tokenizado exitosamente: {problem}")
        return tokenized
    except Exception as e:
        error_message = f"Error al tokenizar el problema: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)
        return np.zeros(MAX_LENGTH)

def main():
    try:
        log_message("Iniciando el programa principal")
        
        # Configure TensorFlow
        os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        log_message("Configuración de TensorFlow completada")

        # Initialize model
        model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        log_message("Modelo Kistmat_AI inicializado y compilado")

        # Start training
        plot_queue = multiprocessing.Queue()
        plot_process = multiprocessing.Process(target=real_time_plotter, args=(plot_queue,))
        plot_process.start()
        log_message("Proceso de visualización en tiempo real iniciado")

        all_history = None
        try:
            log_message("Iniciando entrenamiento con curriculum learning")
            all_history = smooth_curriculum_learning(model, STAGES)
            log_message("Entrenamiento completado exitosamente")
        except Exception as e:
            error_message = f"Error durante el entrenamiento: {str(e)}\n{traceback.format_exc()}"
            log_message(error_message, is_error=True)
        finally:
            plot_queue.put(None)
            plot_process.join()
            log_message("Proceso de visualización finalizado")

        if all_history:
            plot_learning_curves(all_history)
            log_message("Curvas de aprendizaje generadas")

        # Generate test problems and evaluate
        log_message("Generando problemas de prueba")
        test_problems = generate_dataset(100, 'university', difficulty=2.0)
        log_message(f"Generados {len(test_problems)} problemas de prueba")

        log_message("Tokenizando problemas de prueba")
        X_test = np.array([safe_tokenize_problem(p.problem, model.get_learning_stage()) for p in test_problems])
        y_test = np.array([p.solution for p in test_problems])

        log_message(f"Forma inicial de X_test: {X_test.shape}")
        log_message(f"Forma inicial de y_test: {y_test.shape}")

        if len(X_test.shape) == 1:
            X_test = np.expand_dims(X_test, axis=0)
            log_message("X_test expandido a 2D")

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
            log_message("y_test reformado a 2D")
        elif hasattr(y_test[0], 'real') and hasattr(y_test[0], 'imag'):
            y_test = np.array([[sol.real, sol.imag] for sol in y_test])
            log_message("y_test convertido a formato real-imaginario")

        log_message(f"Forma final de X_test: {X_test.shape}")
        log_message(f"Forma final de y_test: {y_test.shape}")

        try:
            log_message("Realizando predicciones")
            predictions = model.predict(X_test)
            log_message(f"Forma de las predicciones: {predictions.shape}")

            if predictions.shape[1] != 1:
                predictions = predictions.mean(axis=1, keepdims=True)
                log_message("Predicciones promediadas a una columna")

            log_message(f"Forma final de las predicciones: {predictions.shape}")

            mse = np.mean(np.square(np.abs(y_test - predictions)))
            log_message(f"Error cuadrático medio: {mse}")

            log_message("\nMuestra de predicciones:")
            sample_size = min(5, len(test_problems))
            for i in range(sample_size):
                log_message(f"Problema: {test_problems[i].problem}")
                log_message(f"Predicción: {predictions[i]}")
                log_message(f"Solución real: {test_problems[i].solution}")
            
            # Aprendizaje por refuerzo
            try:
                log_message("Iniciando aprendizaje por refuerzo")
                losses = parallel_reinforce_learning(model, test_problems[:sample_size],
                                                    predictions[:sample_size], y_test[:sample_size])
                for i, loss in enumerate(losses):
                    log_message(f"Pérdida de aprendizaje por refuerzo para el problema {i + 1}: {loss}")
            except Exception as e:
                error_message = f"Error durante el aprendizaje por refuerzo: {str(e)}\n{traceback.format_exc()}"
                log_message(error_message, is_error=True)

        except Exception as e:
            error_message = f"Error al hacer predicciones: {str(e)}\n{traceback.format_exc()}"
            log_message(error_message, is_error=True)

    except Exception as e:
        error_message = f"Error general en la ejecución del programa: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message = f"Error no manejado en la ejecución principal: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)