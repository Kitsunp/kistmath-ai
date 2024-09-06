from training.curriculum_learning import smooth_curriculum_learning
from training.parallel_training import parallel_reinforce_learning
from utils.logging import log_message
from visualization.plotting import plot_learning_curves
import numpy as np
import traceback

def run_training_pipeline(model, STAGES):
    all_history = None
    try:
        log_message("Iniciando entrenamiento con curriculum learning")
        all_history = smooth_curriculum_learning(model, STAGES)
        log_message("Entrenamiento completado exitosamente")
    except Exception as e:
        error_message = f"Error durante el entrenamiento: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)
    
    if all_history:
        plot_learning_curves(all_history)
        log_message("Curvas de aprendizaje generadas")
    
    return all_history

def run_reinforcement_learning(model, test_problems, predictions, y_test):
    try:
        log_message("Iniciando aprendizaje por refuerzo")
        losses = parallel_reinforce_learning(model, test_problems, predictions, y_test)
        log_message(f"Pérdida media de aprendizaje por refuerzo: {np.mean(losses)}")
        return losses
    except Exception as e:
        error_message = f"Error durante el aprendizaje por refuerzo: {str(e)}\n{traceback.format_exc()}"
        log_message(error_message, is_error=True)
        return None