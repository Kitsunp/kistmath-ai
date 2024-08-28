import os
import tensorflow as tf
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

# Habilitar ejecución ansiosa
tf.config.run_functions_eagerly(True)

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

        # Asegurarse de que X_test tiene la forma correcta
        X_test = X_test.reshape((-1, MAX_LENGTH))

        # Asegurarse de que y_test tiene la forma correcta
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        try:
            predictions = model.predict(X_test)
        except Exception as e:
            print(f"Error al hacer predicciones: {e}")
            return

        # Asegurarse de que las predicciones tienen la misma forma que y_test
        if predictions.shape != y_test.shape:
            predictions = predictions.reshape(y_test.shape)

        print("\nResultados de la prueba:")
        mse = np.mean(np.square(y_test - predictions))
        print(f"Error cuadrático medio: {mse}")

        print("\nMuestra de predicciones:")
        sample_size = min(5, len(test_problems))
        for i in range(sample_size):
            print(f"Problema: {test_problems[i].problem}")
            print(f"Predicción: {predictions[i]}")
            print(f"Solución real: {test_problems[i].solution}")

        try:
            losses = parallel_reinforce_learning(model, test_problems[:sample_size], 
                                                 predictions[:sample_size], y_test[:sample_size])
            for i, loss in enumerate(losses):
                print(f"Pérdida de aprendizaje por refuerzo para el problema {i+1}: {loss}")
        except Exception as e:
            print(f"Error durante el aprendizaje por refuerzo: {e}")

    except Exception as e:
        print(f"Error general en la ejecución del programa: {e}")

if __name__ == "__main__":
    main()