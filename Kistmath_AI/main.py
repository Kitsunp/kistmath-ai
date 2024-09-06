import os
import tensorflow as tf
import numpy as np
from models.kistmat_ai import Kistmat_AI
from config.tensorflow_config import configure_tensorflow
from utils.logging import log_message
from utils.data_preparation import prepare_data
from utils.data_generation import generate_dataset
from utils.evaluation import evaluate_model, evaluate_readiness
from training.training_pipeline import run_training_pipeline, run_reinforcement_learning
from visualization.visualization_process import start_visualization_process, stop_visualization_process
from visualization.plotting import plot_learning_curves
from config.settings import STAGES, MAX_LENGTH, READINESS_THRESHOLDS

# Suprimir mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

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
        
        # Iniciar proceso de visualización
        plot_queue, plot_process = start_visualization_process()

        # Ejecutar pipeline de entrenamiento
        all_history = run_training_pipeline(model, STAGES)

        # Detener proceso de visualización
        stop_visualization_process(plot_queue, plot_process)

        if all_history:
            plot_learning_curves(all_history)
            log_message("Curvas de aprendizaje generadas")

        # Generar problemas de prueba y evaluar
        log_message("Generando problemas de prueba")
        test_problems = generate_dataset(100, 'university', difficulty=2.0)
        log_message(f"Generados {len(test_problems)} problemas de prueba")

        X_test, y_test = prepare_data(test_problems, model)
        log_message(f"Datos de prueba preparados. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Evaluar el modelo
        predictions, mse = evaluate_model(model, X_test, y_test, test_problems)
        
        if predictions is not None:
            # Aprendizaje por refuerzo
            losses = run_reinforcement_learning(model, test_problems, predictions, y_test)
            if losses is not None:
                log_message(f"Pérdida media de aprendizaje por refuerzo: {np.mean(losses)}")

    except Exception as e:
        log_message(f"Error general en la ejecución del programa: {str(e)}", is_error=True)

if __name__ == "__main__":
    main()