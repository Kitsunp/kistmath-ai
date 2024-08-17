from Kistmath_AI.utils.data_generation import generate_dataset
from Kistmath_AI.utils.evaluation import evaluate_readiness
from Kistmath_AI.training.parallel_training import parallel_train_model
from Kistmath_AI.config.settings import READINESS_THRESHOLDS
from Kistmath_AI.visualization.real_time_plotter import RealTimePlotter
from Kistmath_AI.visualization.real_time_plotter import RealTimePlotter

def smooth_curriculum_learning(model, stages, initial_problems=4000, max_problems=5000, difficulty_increase_rate=0.05):
    all_history = []
    current_difficulty = 1.0

    plotter = RealTimePlotter()
    plotter.after(100, plotter.update)  # Start the Tkinter loop

    for stage in stages:
        print(f"\nEntering learning stage: {stage}")
        model.set_learning_stage(stage)

        problems_solved = 0
        stage_history = []

        while problems_solved < max_problems:
            num_problems = min(initial_problems, max_problems - problems_solved)
            problems = generate_dataset(num_problems, stage, current_difficulty)

            fold_histories = parallel_train_model(model, problems, epochs=50)

            # Update real-time plot with fold history loss data
            for history in fold_histories:
                for loss in history['history']['loss']:
                    plotter.update_plot(loss)
                    plotter.update_idletasks()
                    plotter.update()

            model.set_weights(fold_histories[-1]['weights'])

            stage_history.extend(fold_histories)

            val_problems = problems[-len(problems)//5:]
            if evaluate_readiness(model, val_problems, READINESS_THRESHOLDS[stage]):
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