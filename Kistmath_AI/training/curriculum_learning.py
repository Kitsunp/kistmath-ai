import numpy as np
import tensorflow as tf
from utils.data_generation import generate_dataset
from utils.evaluation import evaluate_readiness
from training.parallel_training import parallel_train_model
from config.settings import READINESS_THRESHOLDS
from visualization.real_time_plotter import RealTimePlotter
from utils.tokenization import tokenize_problem

def smooth_curriculum_learning(model, stages, initial_problems=4000, max_problems=5000, difficulty_increase_rate=0.05):
    all_history = []
    current_difficulty = 1.0

    plotter = RealTimePlotter()
    plotter.after(100, plotter.update)  # Start the Tkinter loop

    # Compile the model before starting the learning process
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    for stage in stages:
        print(f"\nEntering learning stage: {stage}")
        model.set_learning_stage(stage)

        problems_solved = 0
        stage_history = []

        while problems_solved < max_problems:
            num_problems = min(initial_problems, max_problems - problems_solved)
            problems = generate_dataset(num_problems, stage, current_difficulty)

            # Prepare data for training
            X = np.array([tokenize_problem(problem.problem, stage) for problem in problems])
            y = np.array([problem.solution for problem in problems])

            # Ensure X has the correct shape (batch_size, time_steps)
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)

            # Reshape y if necessary
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Ensure X and y have the same number of samples
            assert X.shape[0] == y.shape[0], f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match"

            fold_histories = parallel_train_model(model, problems, epochs=50)

            # Update real-time plot with fold history loss data
            for history in fold_histories:
                for loss in history['history']['loss']:
                    plotter.update_plot(loss)
                    plotter.update_idletasks()
                    plotter.update()

            model.set_weights(fold_histories[-1]['weights'])

            stage_history.extend(fold_histories)

            # Prepare validation data
            val_problems = problems[-len(problems)//5:]
            X_val = np.array([tokenize_problem(problem.problem, stage) for problem in val_problems])
            y_val = np.array([problem.solution for problem in val_problems])

            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

            if evaluate_readiness(model, X_val, y_val, READINESS_THRESHOLDS[stage]):
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

    plotter.destroy()  # Close the Tkinter window when done
    
    return all_history
