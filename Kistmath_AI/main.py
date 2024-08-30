import os
import tensorflow as tf
# Suprimir mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
tf.get_logger().setLevel('ERROR')  # or 'WARNING'
import numpy as np
import multiprocessing
from models.kistmat_ai import Kistmat_AI
from models.external_memory import IntegratedMemorySystem
from training.curriculum_learning import smooth_curriculum_learning
from utils.data_generation import generate_dataset
from utils.evaluation import evaluate_readiness
from utils.tokenization import tokenize_problem
from training.parallel_training import parallel_reinforce_learning
from visualization.plotting import plot_learning_curves, real_time_plotter
from config.settings import STAGES, MAX_LENGTH, READINESS_THRESHOLDS

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
# Habilitar ejecución ansiosa

def safe_tokenize_problem(problem, learning_stage):
    try:
        return tokenize_problem(problem, learning_stage)
    except Exception as e:
        print(f"Error al tokenizar el problema: {e}")
        return np.zeros(MAX_LENGTH)  # Devolver un array de ceros como fallback

def main():
    try:
        # Configure TensorFlow
        os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # Initialize model
        model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Start training
        plot_queue = multiprocessing.Queue()
        plot_process = multiprocessing.Process(target=real_time_plotter, args=(plot_queue,))
        plot_process.start()

        all_history = None
        try:
            all_history = smooth_curriculum_learning(model, STAGES)
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
        finally:
            plot_queue.put(None)
            plot_process.join()

        if all_history:
            plot_learning_curves(all_history)

        # Generate test problems and evaluate
        test_problems = generate_dataset(100, 'university', difficulty=2.0)

# Tokenizar problemas de prueba de manera segura
        X_test = np.array([safe_tokenize_problem(p.problem, model.get_learning_stage()) for p in test_problems])
        y_test = np.array([p.solution for p in test_problems])

        print(f"Forma inicial de X_test: {X_test.shape}")
        print(f"Forma inicial de y_test: {y_test.shape}")

        # Asegurarse de que X_test tiene la forma correcta
        if len(X_test.shape) == 1:
            X_test = np.expand_dims(X_test, axis=0)

        # Asegurarse de que y_test tiene la forma correcta
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        elif hasattr(y_test[0], 'real') and hasattr(y_test[0], 'imag'):
            y_test = np.array([[sol.real, sol.imag] for sol in y_test])

        print(f"Forma final de X_test: {X_test.shape}")
        print(f"Forma final de y_test: {y_test.shape}")

        try:
            predictions = model.predict(X_test)
            print(f"Forma de las predicciones: {predictions.shape}")

            # Asegurarse de que las predicciones tengan la forma correcta
            if predictions.shape[1] != 1:
                predictions = predictions.mean(axis=1, keepdims=True)

            print(f"Forma final de las predicciones: {predictions.shape}")

            # Calcula el MSE
            mse = np.mean(np.square(np.abs(y_test - predictions)))
            print(f"Error cuadrático medio: {mse}")

            print("\nMuestra de predicciones:")
            sample_size = min(5, len(test_problems))
            for i in range(sample_size):
                print(f"Problema: {test_problems[i].problem}")
                print(f"Predicción: {predictions[i]}")
                print(f"Solución real: {test_problems[i].solution}")

            # Aprendizaje por refuerzo
            try:
                losses = parallel_reinforce_learning(model, test_problems[:sample_size],
                                                    predictions[:sample_size], y_test[:sample_size])
                for i, loss in enumerate(losses):
                    print(f"Pérdida de aprendizaje por refuerzo para el problema {i + 1}: {loss}")
            except Exception as e:
                print(f"Error durante el aprendizaje por refuerzo: {e}")

        except Exception as e:
            print(f"Error al hacer predicciones: {e}")
            return

    except ValueError as e:
        print(f"Error de valor en la ejecución del programa: {e}")
        print(f"Tamaño de X_test: {X_test.size}")
        print(f"Tamaño de y_test: {y_test.size}")
        raise
    except Exception as e:
        print(f"Error general en la ejecución del programa: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error no manejado en la ejecución principal: {str(e)}")
        import traceback
        traceback.print_exc()