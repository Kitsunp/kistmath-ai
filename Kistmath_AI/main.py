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
# Configurar TensorFlow de manera más Flexible
def log_message(message, is_error=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "ERROR" if is_error else "INFO"
    with open("program_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] {prefix}: {message}\n")
#tokenize_problem seguro
def safe_tokenize_problem(problem, learning_stage):
    try:
        tokenized = tokenize_problem(problem, learning_stage)
        log_message(f"Problema tokenizado exitosamente: {problem}")
        return tokenized
    except Exception as e:
        error_message = f"Error al tokenizar el problema: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)
        # En lugar de devolver ceros, podríamos lanzar una excepción para manejar este caso de forma más explícita
        raise ValueError("No se pudo tokenizar el problema")
# Configurar TensorFlow
def configure_tensorflow():
    # Configurar TensorFlow de manera más flexible
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Usa todos los núcleos disponibles
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Usa todos los núcleos disponibles
    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(True)  # Habilita XLA JIT compilation
    
    # Configurar el crecimiento de memoria de GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            log_message(f"Error al configurar GPU: {str(e)}", is_error=True)

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

def main():
    try:
        log_message("Iniciando el programa principal")
        
        configure_tensorflow()
        log_message("Configuración de TensorFlow completada")

        # Inicializar modelo
        model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_dim=2)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Construir el modelo explícitamente
        model.build(input_shape=(None, MAX_LENGTH))
        
        log_message("Modelo Kistmat_AI inicializado, compilado y construido")
        # Iniciar entrenamiento
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

        # Generar problemas de prueba y evaluar
        log_message("Generando problemas de prueba")
        test_problems = generate_dataset(100, 'university', difficulty=2.0)
        log_message(f"Generados {len(test_problems)} problemas de prueba")

        X_test, y_test = prepare_data(test_problems, model)
        log_message(f"Datos de prueba preparados. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

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
            
            # Aprendizaje por refuerzo
            try:
                log_message("Iniciando aprendizaje por refuerzo")
                losses = parallel_reinforce_learning(model, test_problems, predictions, y_test)
                log_message(f"Pérdida media de aprendizaje por refuerzo: {np.mean(losses)}")
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
    main()