import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
from models.external_memory import IntegratedMemorySystem
from config.settings import VOCAB_SIZE, MAX_LENGTH
from models.symbolic_reasoning import SymbolicReasoning
from utils.tokenization import tokenize_problem
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_keras_serializable(package='Custom', name=None)
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_dim, vocab_size=VOCAB_SIZE, name=None, **kwargs):
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

        self._input_shape = input_shape
        self._output_dim = output_dim
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

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_dim(self):
        return self._output_dim

    def _init_layers(self):
        try:
            self.final_reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='final_reasoning_layer')
            self.embedding = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=64, name='embedding')
            self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True), name='lstm1')
            self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=False), name='lstm2')
            self.dropout = keras.layers.Dropout(0.5, name='dropout')
            self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, attention_axes=(1, 2), name='attention')
            self.memory_query = keras.layers.Dense(64, dtype='float32', name='memory_query')
            self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='reasoning_layer')
            self.batch_norm = keras.layers.BatchNormalization(name='batch_norm')
            self.rule_dense = keras.layers.Dense(256, activation='relu', name='rule_dense')
            self.rule_output = keras.layers.Dense(64, activation='linear', name='rule_output')
            self.memory_attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, name='memory_attention')
            self.memory_dense = keras.layers.Dense(512, activation='relu', name='memory_dense')
            self.final_output = keras.layers.Dense(self._output_dim, activation='linear', name='final_output')
        except ValueError as e:
            logger.error(f"ValueError in _init_layers: {e}")
            raise ValueError(f"Error initializing layers: {e}")
        except TypeError as e:
            logger.error(f"TypeError in _init_layers: {e}")
            raise TypeError(f"Incorrect type for layer initialization: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in _init_layers: {e}")
            raise RuntimeError(f"Unexpected error during layer initialization: {e}")

    def _init_memory_components(self):
        try:
            memory_config = {
                'external_memory': {'memory_size': 1000, 'key_size': 64, 'value_size': 128},
                'formulative_memory': {'max_formulas': 5000, 'embedding_size': 64},
                'conceptual_memory': {'embedding_size': 64},
                'short_term_memory': {'capacity': 100},
                'long_term_memory': {'capacity': 10000},
                'inference_memory': {'capacity': 1000, 'embedding_size': 64}
            }
            self.memory_system = IntegratedMemorySystem(memory_config)
        except ValueError as e:
            logger.error(f"ValueError in _init_memory_components: {e}")
            raise ValueError(f"Invalid configuration for memory components: {e}")
        except TypeError as e:
            logger.error(f"TypeError in _init_memory_components: {e}")
            raise TypeError(f"Incorrect type in memory component configuration: {e}")
        except AttributeError as e:
            logger.error(f"AttributeError in _init_memory_components: {e}")
            raise AttributeError(f"Missing attribute in IntegratedMemorySystem: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in _init_memory_components: {e}")
            raise RuntimeError(f"Unexpected error initializing memory components: {e}")

    def _init_output_layers(self):
        try:
            stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                    'high_school1', 'high_school2', 'high_school3', 'university']
            self.output_layers = {stage: keras.layers.Dense(128, activation='linear') for stage in stages}
            self.final_output = keras.layers.Dense(self._output_dim, activation='linear')
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
        return self._learning_stage.numpy().decode('utf-8') if tf.executing_eagerly() else self._learning_stage

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

    @tf.function
    def _apply_reasoning(self, x, training):
        try:
            # Aseguramos que la forma de la entrada sea la correcta
            x = tf.ensure_shape(x, [None, self.input_shape[-1]])

            # Aplicar la capa de razonamiento inicial
            x = self.reasoning_layer(x)
            x = self.batch_norm(x, training=training)
            x = self.dropout(x, training=training)

            # Integrar razonamiento simbólico
            symbolic_out = self.symbolic_reasoning.reason(x)
            
            # Depuración: imprimir tipo y contenido de symbolic_out
            tf.print("Tipo de symbolic_out:", tf.shape(symbolic_out))
            tf.print("Contenido de symbolic_out:", symbolic_out)

            # Concatenar la salida simbólica
            x = tf.concat([x, symbolic_out], axis=-1)

            # Aplicar atención multi-cabeza para capturar relaciones complejas
            x_reshaped = tf.expand_dims(x, axis=1)
            attention_out = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = tf.squeeze(attention_out, axis=1)

            # Consultar la memoria externa para integrar conocimiento previo
            memory_query = self.memory_query(x)
            memory_out = self.memory_system.process_input({
                'external_query': memory_query,
                'external_query_terms': None
            })['external_memory']

            # Verificar el formato de salida de la memoria
            if not isinstance(memory_out, tuple) or len(memory_out) < 1:
                raise ValueError("Unexpected output format from memory system")

            x = tf.concat([x, memory_out[0]], axis=-1)

            # Aplicar razonamiento basado en reglas
            rule_based_out = self.apply_rule_based_reasoning(x, self.get_learning_stage())
            x = tf.concat([x, rule_based_out], axis=-1)

            # Capa final de integración
            x = self.final_reasoning_layer(x)
            x = self.batch_norm(x, training=training)
            x = self.dropout(x, training=training)

            return x
        except ValueError as e:
            # Manejo de errores específico para fallos de valor
            tf.print(f"Error de valor en _apply_reasoning: {e}")
            tf.print(f"Tipo de x: {tf.shape(x)}")
            tf.print(f"Forma de x: {x.shape}")
            raise
        except Exception as e:
            # Manejo de errores genérico
            tf.print(f"Error inesperado en _apply_reasoning: {e}")
            raise

    @tf.function
    def apply_rule_based_reasoning(self, x, current_stage):
        try:
            # Asegurar que current_stage es un tensor de strings
            if not isinstance(current_stage, tf.Tensor) or current_stage.dtype != tf.string:
                current_stage = tf.convert_to_tensor(current_stage, dtype=tf.string)

            stage = tf.strings.lower(current_stage)

            # Procesar la entrada a través de una capa densa
            x = self.rule_dense(x)

            # Aplicar reglas específicas según la etapa
            conditions = [
                (tf.strings.regex_full_match(stage, "elementary[123]"), lambda: tf.where(x > 0, x + 1, x - 1)),
                (tf.strings.regex_full_match(stage, "junior_high[12]"), lambda: tf.abs(x) * tf.sign(tf.reduce_sum(x, axis=-1, keepdims=True))),
                (tf.strings.regex_full_match(stage, "high_school[123]"), lambda: tf.where(tf.abs(x) > 1, x ** 2, tf.sqrt(tf.abs(x)))),
                (tf.equal(stage, "university"), lambda: tf.sin(x) + tf.cos(x))
            ]

            x = tf.case(conditions, default=lambda: x, exclusive=True)

            # Capa de salida para el razonamiento basado en reglas
            return self.rule_output(x)
        except Exception as e:
            logger.error(f"Error in apply_rule_based_reasoning: {e}")
            return x  # Return the input as is in case of error

    def _process_complex_output(self, x):
        # Asumimos que x tiene forma (batch_size, 2)
        return tf.complex(x[:, 0], x[:, 1])      

    @tf.function
    def call(self, inputs, training=False):
        try:
            current_stage = self.get_learning_stage()

            if isinstance(inputs, np.ndarray):
                inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

            if tf.equal(current_stage, tf.constant('university')):
                x = inputs
            else:
                x = self._process_input(inputs)
                x = self._apply_attention(x)
                x = self._query_memories(x)

            # Ensuring that x has the correct shape before further processing
            x = tf.ensure_shape(x, [None, self.input_shape[-1]])
            x = self._apply_reasoning(x, training)
            x = self._generate_output(x, current_stage, training)
            
            # Procesar salida compleja
            x = self._process_complex_output(x)
            
            # Ensure the output shape is correct (batch_size, 2)
            x = tf.reshape(x, [-1, self._output_dim])

            return x
        except Exception as e:
            logger.error(f"Error in call: {e}")
            logger.error(f"Input shape: {inputs.shape}")
            logger.error(f"Current stage: {current_stage}")
            raise  # Re-raise the exception after logging

    def _process_input(self, inputs):
        try:
            current_stage = self.get_learning_stage()
            if isinstance(inputs, np.ndarray):
                x = np.array([tokenize_problem(str(p), current_stage) for p in inputs])
            else:
                x = tf.numpy_function(
                    lambda p: tokenize_problem(p.decode() if isinstance(p, bytes) else str(p), current_stage),
                    [inputs],
                    tf.float32
                )

            x = tf.convert_to_tensor(x, dtype=tf.float32)
            tf.print(f"Forma de x después de la conversión: {tf.shape(x)}")

            # Asegurarse de que x tiene la forma correcta (batch_size, time_steps, 1)
            if len(tf.shape(x)) == 2:
                x = tf.expand_dims(x, axis=-1)
            
            # Asegurarse de que la forma es conocida antes de pasar a la capa de embedding
            x = tf.ensure_shape(x, [None, self._input_shape[0], 1])
            
            x = self.embedding(tf.cast(tf.squeeze(x, axis=-1), tf.int32))
            tf.print(f"Forma de x después de embedding: {tf.shape(x)}")

            # Ahora x debería tener la forma (batch_size, time_steps, embedding_dim)
            x = self.lstm1(x)
            x = self.lstm2(x)
            return x
        except Exception as e:
            tf.print(f"Error detallado en _process_input: {e}")
            tf.print(f"Tipo de inputs: {tf.shape(inputs)}")
            tf.print(f"Forma de inputs: {tf.shape(inputs)}")
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

    def build(self, input_shape):
        super().build(input_shape)
        # Asegurarse de que input_shape es un TensorShape
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        # Actualizar self._input_shape si es necesario
        if self._input_shape != input_shape[1:]:
            self._input_shape = input_shape[1:]
        # Inicializar capas
        self._init_layers()
        self._init_memory_components()
        self._init_output_layers()
        self.built = True
    @tf.function
    def _query_memories(self, x):
        try:
            logger.info(f"Iniciando _query_memories. Forma de x: {tf.shape(x)}")
            
            # Generar la consulta de memoria
            query = self.memory_query(x)
            logger.info(f"Forma de query después de memory_query: {tf.shape(query)}")
            
            # Asegurar que la consulta tenga la forma correcta
            query = tf.reshape(query, [-1, self.memory_system.external_memory.key_size])
            logger.info(f"Forma de query después de reshape: {tf.shape(query)}")
            
            # Preparar los embeddings de memoria para la multiplicación matricial
            memory_embeddings = tf.transpose(self.memory_system.external_memory.memory_embeddings)
            logger.info(f"Forma de memory_embeddings: {tf.shape(memory_embeddings)}")

            # Calcular similitudes
            similarities = tf.matmul(query, memory_embeddings)
            logger.info(f"Forma de similarities: {tf.shape(similarities)}")

            # Obtener los top_k resultados más similares
            top_k = tf.minimum(5, tf.shape(similarities)[1])
            _, top_indices = tf.nn.top_k(similarities[0], k=top_k)

            # Obtener los embeddings de memoria correspondientes
            retrieved_memories = tf.gather(self.memory_system.external_memory.memory_embeddings, top_indices)
            logger.info(f"Forma de retrieved_memories: {tf.shape(retrieved_memories)}")

            # Procesar los resultados de diferentes tipos de memoria
            memory_results = self.memory_system.process_input({
                'external_query': (query, tf.constant([])),
                'formulative_query': (query, self._extract_relevant_terms(query)),
                'conceptual_query': query,
                'short_term_query': query,
                'long_term_query': query,
                'inference_query': query
            })

            # Combinar los resultados de diferentes memorias
            memory_outputs = []
            for memory_type in ['external_memory', 'formulative_memory', 'conceptual_memory', 
                                'short_term_memory', 'long_term_memory', 'inference_memory']:
                if memory_type in memory_results:
                    result = memory_results[memory_type]
                    if isinstance(result, tuple):
                        result = result[0]  # Tomar solo el primer elemento si es una tupla
                    if tf.rank(result) == 2:
                        result = tf.expand_dims(result, axis=1)
                    memory_outputs.append(result)
                    logger.info(f"Forma de resultado de {memory_type}: {tf.shape(result)}")

            # Combinar todos los resultados de memoria
            combined_memory = tf.concat(memory_outputs, axis=1)
            logger.info(f"Forma de combined_memory: {tf.shape(combined_memory)}")

            # Aplicar atención sobre la memoria combinada
            x_expanded = tf.expand_dims(x, axis=1)
            attended_memory = self.memory_attention(x_expanded, combined_memory, combined_memory)
            attended_memory = tf.squeeze(attended_memory, axis=1)
            logger.info(f"Forma de attended_memory: {tf.shape(attended_memory)}")

            # Procesar la memoria atendida
            processed_memory = self.memory_dense(attended_memory)
            logger.info(f"Forma de processed_memory: {tf.shape(processed_memory)}")

            # Combinar la entrada original con la memoria procesada
            result = tf.concat([x, processed_memory], axis=-1)
            logger.info(f"Forma final del resultado: {tf.shape(result)}")

            return result

        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Error de argumento inválido en _query_memories: {e}")
            logger.error(f"Formas de las matrices involucradas:")
            logger.error(f"x: {tf.shape(x)}")
            logger.error(f"query: {tf.shape(query)}")
            logger.error(f"memory_embeddings: {tf.shape(self.memory_system.external_memory.memory_embeddings)}")
            return x  # Devolver la entrada original en caso de error
        except Exception as e:
            logger.error(f"Error inesperado en _query_memories: {e}")
            return x  # Devolver la entrada original en caso de error

    def _generate_output(self, x, current_stage, training):
        try:
            def stage_fn(stage):
                return lambda: self.output_layers[stage](x)

            stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 
                    'junior_high2', 'high_school1', 'high_school2', 'high_school3', 
                    'university']

            # Cambiar la manera de crear branch_fns para que utilizamos tf.constant
            branch_fns = {tf.constant(stage): stage_fn(stage) for stage in stages}

            # Crear la condición de ramificación de salida
            x = tf.case(
                [(tf.equal(current_stage, key), value) for key, value in branch_fns.items()], 
                default=stage_fn('elementary1')
            )

            if training:
                not_university = tf.not_equal(current_stage, tf.constant('university'))
                self._update_memories = tf.cond(not_university,
                                                lambda: self._update_memories(x), 
                                                lambda: tf.no_op())

            x = self.final_output(x)
            return tf.reshape(x, [-1, self._output_dim])

        except TypeError as e:
            print(f"Error de tipo en _generate_output: {e}")
            print(f"Tipo de current_stage: {type(current_stage)}")
            print(f"Contenido de current_stage: {current_stage}")
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
        try:
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
        
        except Exception as e:
            logger.error(f"Error en _extract_relevant_terms: {e}")
            return query  # Devolver la consulta original en caso de error

    def get_config(self):
        try:
            config = super().get_config()
            config.update({
                "input_shape": self._input_shape,
                "output_dim": self._output_dim,
                "vocab_size": self.vocab_size,
                "learning_stage": self.get_learning_stage()
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
            output_dim = config.pop("output_dim", 2)
            vocab_size = config.pop("vocab_size", VOCAB_SIZE)
            learning_stage = config.pop("learning_stage", "elementary1")

            instance = cls(input_shape=input_shape, output_dim=output_dim, vocab_size=vocab_size, **config)
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
        except Exception as e:
            logger.error(f"Error in _numpy_to_tensor: {e}")
            return input_data