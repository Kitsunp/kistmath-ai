import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from models.kistmat_ai import Kistmat_AI
from training.curriculum_learning import smooth_curriculum_learning
from utils.data_generation import generate_dataset
from utils.evaluation import evaluate_readiness
from utils.tokenization import tokenize_calculus_problem
from training.parallel_training import parallel_reinforce_learning
from visualization.plotting import plot_learning_curves, real_time_plotter
from config.settings import STAGES, MAX_LENGTH, READINESS_THRESHOLDS
import multiprocessing
from numpy import np

def main():
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
        all_history = smooth_curriculum_learning(model, STAGES, plot_queue=plot_queue)
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        plot_queue.put(None)
        plot_process.join()

    if all_history:
        plot_learning_curves(all_history)

    # Generate test problems and evaluate
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