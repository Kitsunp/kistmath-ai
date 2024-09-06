import tensorflow as tf
from utils.logging import log_message

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