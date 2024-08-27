import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
from models.external_memory import IntegratedMemorySystem
from config.settings import VOCAB_SIZE, MAX_LENGTH
from models.symbolic_reasoning import SymbolicReasoning
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
@register_keras_serializable(package='Custom', name=None)
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        try:
            super(Kistmat_AI, self).__init__(name=name, **kwargs)
        except TypeError as e:
            print(f"Error en el constructor de Kistmat_AI: {e}")
            print("Asegúrate de que los argumentos de entrada son del tipo correcto.")
            raise
        except ValueError as e:
            print(f"Error en el constructor de Kistmat_AI: {e}")
            print("Asegúrate de que los argumentos de entrada tienen los valores adecuados.")
            raise
        except Exception as e:
            print(f"Error inesperado en el constructor de Kistmat_AI: {e}")
            print("Ha ocurrido un error desconocido. Por favor, revisa el código y los argumentos de entrada.")
            raise

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        self.symbolic_reasoning = SymbolicReasoning()

        try:
            self._init_layers()
            self._init_memory_components()
            self._init_output_layers()
        except ValueError as e:
            print(f"Error en la inicialización de los componentes: {e}")
            print("Asegúrate de que los parámetros de inicialización sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en la inicialización de los componentes: {e}")
            print("Ha ocurrido un error desconocido durante la inicialización. Por favor, revisa el código.")
            raise

        self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)

    def _init_layers(self):
        try:
            self.final_reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
            self.embedding = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=64)
            self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
            self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
            self.dropout = keras.layers.Dropout(0.5)
            self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, attention_axes=(1, 2))
            self.memory_query = keras.layers.Dense(64, dtype='float32')
            # Specify a fixed input shape for reasoning_layer
            self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(self.input_shape[-1],))
            self.batch_norm = keras.layers.BatchNormalization()
        except ValueError as e:
            print(f"Error en _init_layers: {e}")
            print("Asegúrate de que los valores de los parámetros de inicialización de las capas sean correctos.")
            raise
        except TypeError as e:
            print(f"Error en _init_layers: {e}")
            print("Asegúrate de que los tipos de los parámetros de inicialización de las capas sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en _init_layers: {e}")
            print("Ha ocurrido un error desconocido durante la inicialización de las capas. Por favor, revisa el código.")
            raise

    def _init_memory_components(self):
        
        try:
            self.memory_system = IntegratedMemorySystem({
                'external_memory': {'memory_size': 100, 'key_size': 64, 'value_size': 128},
                'formulative_memory': {'max_formulas': 1000},
                'conceptual_memory': {},
                'short_term_memory': {'capacity': 100},
                'long_term_memory': {'capacity': 10000},
                'inference_memory': {'capacity': 500, 'embedding_size': 64}
            })
        except ValueError as e:
            print(f"Error en _init_memory_components: {e}")
            print("Asegúrate de que los valores de los parámetros de inicialización de los componentes de memoria sean correctos.")
            raise
        except TypeError as e:
            print(f"Error en _init_memory_components: {e}")
            print("Asegúrate de que los tipos de los parámetros de inicialización de los componentes de memoria sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en _init_memory_components: {e}")
            print("Ha ocurrido un error desconocido durante la inicialización de los componentes de memoria. Por favor, revisa el código.")
            raise

    def _init_output_layers(self):
        try:
            stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                      'high_school1', 'high_school2', 'high_school3', 'university']
            self.output_layers = {stage: keras.layers.Dense(128, activation='linear') for stage in stages}
            self.final_output = keras.layers.Dense(self.output_shape, activation='sigmoid')
        except ValueError as e:
            print(f"Error en _init_output_layers: {e}")
            print("Asegúrate de que los valores de los parámetros de inicialización de las capas de salida sean correctos.")
            raise
        except TypeError as e:
            print(f"Error en _init_output_layers: {e}")
            print("Asegúrate de que los tipos de los parámetros de inicialización de las capas de salida sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en _init_output_layers: {e}")
            print("Ha ocurrido un error desconocido durante la inicialización de las capas de salida. Por favor, revisa el código.")
            raise

    def get_learning_stage(self):
        return tf.strings.strip(self._learning_stage)

    def set_learning_stage(self, stage):
        try:
            self._learning_stage.assign(stage.encode())
        except TypeError as e:
            print(f"Error en set_learning_stage: {e}")
            print("Asegúrate de que el argumento 'stage' sea del tipo correcto (una cadena de caracteres).")
            raise
        except Exception as e:
            print(f"Error inesperado en set_learning_stage: {e}")
            print("Ha ocurrido un error desconocido al establecer la etapa de aprendizaje. Por favor, revisa el código.")
            raise
    def _apply_reasoning(self, x, training):
        # Aplicar la capa de razonamiento inicial
        x = self.reasoning_layer(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)

        # Integrar razonamiento simbólico
        symbolic_out = self.symbolic_reasoning.reason(x)
        x = tf.concat([x, symbolic_out], axis=-1)

        # Aplicar atención multi-cabeza para capturar relaciones complejas
        x_reshaped = tf.expand_dims(x, axis=1)
        attention_out = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = tf.squeeze(attention_out, axis=1)

        # Consultar la memoria externa para integrar conocimiento previo
        memory_query = self.memory_query(x)
        memory_out = self.memory_system.process_input({
            'external_query': memory_query,
            'external_query_terms': None  # O proporciona los términos de consulta si están disponibles
        })['external_memory']
        x = tf.concat([x, memory_out[0]], axis=-1)

        # Capa final de integración
        x = self.final_reasoning_layer(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x
    @tf.function
    def call(self, inputs, training=False):
        try:
            current_stage = self.get_learning_stage()

            if isinstance(inputs, np.ndarray):
                inputs = self._numpy_to_tensor(inputs)

            if tf.equal(current_stage, tf.constant('university')):
                x = inputs
            else:
                x = self._process_input(inputs)
                x = self._apply_attention(x)
                x = self._query_memories(x)

            x = tf.ensure_shape(x, [None, self.input_shape[-1]])
            x = self._apply_reasoning(x, training)
            return self._generate_output(x, current_stage, training)
        except AttributeError as e:
            logger.error(f"Error en call: {e}")
            logger.error("Asegúrate de que todos los métodos y atributos estén correctamente definidos.")
            raise
        except Exception as e:
            logger.error(f"Error inesperado en call: {e}")
            raise

    def _process_input(self, inputs):
        try:
            x = self.embedding(inputs)
            x = self.lstm1(x)
            return self.lstm2(x)
        except TypeError as e:
            print(f"Error en _process_input: {e}")
            print("Asegúrate de que los argumentos de entrada sean del tipo correcto.")
            raise
        except ValueError as e:
            print(f"Error en _process_input: {e}")
            print("Asegúrate de que los valores de los argumentos de entrada sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en _process_input: {e}")
            print("Ha ocurrido un error desconocido durante el procesamiento de la entrada. Por favor, revisa el código.")
            raise


    def _apply_attention(self, x):
        try:
            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped, x_reshaped)
            return tf.squeeze(context, axis=1)
        except TypeError as e:
            print(f"Error en _apply_attention: {e}")
            print("Asegúrate de que los argumentos de entrada sean del tipo correcto (TensorFlow tensors).")
            raise
        except tf.errors.InvalidArgumentError as e:
            print(f"Error en _apply_attention: {e}")
            print("Parece que hay un problema con los tensores de entrada. Revisa que sean compatibles con las operaciones necesarias.")
            raise
        except Exception as e:
            print(f"Error inesperado en _apply_attention: {e}")
            print("Ha ocurrido un error desconocido durante la aplicación de la atención. Por favor, revisa el código.")
            raise

    @tf.function
    def _query_memories(self, x):
        try:
            query = self.memory_query(x)
            memory_results = self.memory_system.process_input({
                'external_query': (query, tf.constant([])),
                'formulative_query': (query, self._extract_relevant_terms(query)),
                'conceptual_query': query,
                'short_term_query': query,
                'long_term_memory': query,
                'inference_query': query
            })

            memory_outputs = [
                memory_results['external_memory'],
                memory_results['formulative_memory'],
                memory_results['conceptual_memory'],
                memory_results['short_term_memory'],
                memory_results['long_term_memory'],
                memory_results['inference_memory']
            ]
            memory_outputs = [tf.expand_dims(m, axis=1) if tf.rank(m) == 2 else m for m in memory_outputs]
            combined_memory = tf.concat(memory_outputs, axis=1)
            return tf.reduce_mean(combined_memory, axis=1)
        except KeyError as e:
            logger.error(f"Error en _query_memories: Clave no encontrada - {e}")
            return tf.zeros_like(x)
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Error en _query_memories: Argumento inválido - {e}")
            return tf.zeros_like(x)
        except Exception as e:
            logger.error(f"Error inesperado en _query_memories: {e}")
            return tf.zeros_like(x)
    def _init_layers(self):
        try:
            self.embedding = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=64)
            self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
            self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
            self.dropout = keras.layers.Dropout(0.5)
            self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, attention_axes=(1, 2))
            self.memory_query = keras.layers.Dense(64, dtype='float32')
            self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
            self.batch_norm = keras.layers.BatchNormalization()
        except ValueError as e:
            logger.error(f"Error en _init_layers: {e}")
            raise
        except TypeError as e:
            logger.error(f"Error en _init_layers: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado en _init_layers: {e}")
            raise

    def _generate_output(self, x, current_stage, training):
        try:
            def stage_fn(stage):
                return lambda: self.output_layers[stage](x)

            stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                    'high_school1', 'high_school2', 'high_school3', 'university']
            
            branch_fns = {tf.equal(current_stage, tf.constant(stage)): stage_fn(stage) for stage in stages}
            
            x = tf.case(branch_fns, default=stage_fn('elementary1'))

            if training:
                not_university = tf.not_equal(current_stage, tf.constant('university'))
                self._update_memories = tf.cond(not_university, lambda: self._update_memories(x), lambda: tf.no_op())
            
            return self.final_output(x)
        except TypeError as e:
            print(f"Error en _generate_output: {e}")
            print("Asegúrate de que los argumentos de entrada sean del tipo correcto.")
            raise
        except ValueError as e:
            print(f"Error en _generate_output: {e}")
            print("Asegúrate de que los valores de los argumentos de entrada sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en _generate_output: {e}")
            print("Ha ocurrido un error desconocido durante la generación de la salida. Por favor, revisa el código.")
            raise
    @tf.function
    def _extract_relevant_terms(self, query):
        current_stage = self.get_learning_stage()
        
        def extract_terms(term_indices):
            return tf.reduce_sum(tf.gather(query, term_indices, axis=-1), axis=-1)

        elementary_indices = tf.constant([0, 1, 2, 3, 4, 5])
        junior_high_indices = tf.constant([6, 7, 8, 9, 10, 11])
        high_school_indices = tf.constant([12, 13, 14, 15, 16, 17])
        university_indices = tf.constant([18, 19, 20, 21, 22, 23])

        return tf.case([
            (tf.equal(current_stage, tf.constant('elementary1')), lambda: extract_terms(elementary_indices)),
            (tf.equal(current_stage, tf.constant('elementary2')), lambda: extract_terms(elementary_indices)),
            (tf.equal(current_stage, tf.constant('elementary3')), lambda: extract_terms(elementary_indices)),
            (tf.equal(current_stage, tf.constant('junior_high1')), lambda: extract_terms(junior_high_indices)),
            (tf.equal(current_stage, tf.constant('junior_high2')), lambda: extract_terms(junior_high_indices)),
            (tf.equal(current_stage, tf.constant('high_school1')), lambda: extract_terms(high_school_indices)),
            (tf.equal(current_stage, tf.constant('high_school2')), lambda: extract_terms(high_school_indices)),
            (tf.equal(current_stage, tf.constant('high_school3')), lambda: extract_terms(high_school_indices)),
        ], default=lambda: extract_terms(university_indices))


    def get_config(self):
        try:
            config = super().get_config()
            config.update({
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "vocab_size": self.vocab_size,
                "learning_stage": tf.get_static_value(self.get_learning_stage()).decode('utf-8')
            })
            return config
        except TypeError as e:
            print(f"Error en get_config: {e}")
            print("Asegúrate de que los atributos del modelo sean del tipo correcto.")
            raise
        except ValueError as e:
            print(f"Error en get_config: {e}")
            print("Asegúrate de que los valores de los atributos del modelo sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en get_config: {e}")
            print("Ha ocurrido un error desconocido durante la obtención de la configuración del modelo. Por favor, revisa el código.")
            raise

    @classmethod
    def from_config(cls, config):
        try:
            input_shape = config.pop("input_shape", (MAX_LENGTH,))
            output_shape = config.pop("output_shape", 1)
            vocab_size = config.pop("vocab_size", VOCAB_SIZE)
            learning_stage = config.pop("learning_stage", "elementary1")

            instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
            instance.set_learning_stage(learning_stage)
            return instance
        except TypeError as e:
            print(f"Error en from_config: {e}")
            print("Asegúrate de que los argumentos de entrada sean del tipo correcto.")
            raise
        except ValueError as e:
            print(f"Error en from_config: {e}")
            print("Asegúrate de que los valores de los argumentos de entrada sean correctos.")
            raise
        except Exception as e:
            print(f"Error inesperado en from_config: {e}")
            print("Ha ocurrido un error desconocido durante la creación del modelo a partir de la configuración. Por favor, revisa el código.")
            raise

    def solve_problem(self, problem):
        try:
            if isinstance(problem, str):  # Assuming string problems are symbolic
                return self.symbolic_reasoning.solve_equation(problem)
            else:
                raise TypeError("El problema debe ser una cadena de caracteres (string).")
        except TypeError as e:
            print(f"Error en solve_problem: {e}")
            print("Asegúrate de que el argumento 'problem' sea una cadena de caracteres (string).")
            raise
        except Exception as e:
            print(f"Error inesperado en solve_problem: {e}")
            print("Ha ocurrido un error desconocido durante la resolución del problema. Por favor, revisa el código.")
            raise

    def save_memory_state(self, directory: str) -> None:
        try:
            self.memory_system.save_state(directory)
        except TypeError as e:
            print(f"Error en save_memory_state: {e}")
            print("Asegúrate de que el argumento 'directory' sea una cadena de caracteres (string).")
            raise
        except Exception as e:
            print(f"Error inesperado en save_memory_state: {e}")
            print("Ha ocurrido un error desconocido durante el guardado del estado de la memoria. Por favor, revisa el código.")
            raise

    def load_memory_state(self, directory: str) -> None:
        try:
            self.memory_system.load_state(directory)
        except TypeError as e:
            print(f"Error en load_memory_state: {e}")
            print("Asegúrate de que el argumento 'directory' sea una cadena de caracteres (string).")
            raise
        except Exception as e:
            print(f"Error inesperado en load_memory_state: {e}")
            print("Ha ocurrido un error desconocido durante la carga del estado de la memoria. Por favor, revisa el código.")
            raise

    def _numpy_to_tensor(self, input_data):
        try:
            if isinstance(input_data, np.ndarray):
                return tf.convert_to_tensor(input_data, dtype=tf.float32)
            else:
                return input_data
        except TypeError as e:
            print(f"Error en _numpy_to_tensor: {e}")
            print("Asegúrate de que el argumento 'input_data' sea un objeto NumPy o un objeto TensorFlow.")
            raise
        except Exception as e:
            print(f"Error inesperado en _numpy_to_tensor: {e}")
            print("Ha ocurrido un error desconocido durante la conversión de NumPy a TensorFlow. Por favor, revisa el código.")
            raise
    