import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
from external_memory import IntegratedMemorySystem
from config.settings import VOCAB_SIZE, MAX_LENGTH
from symbolic_reasoning import SymbolicReasoning

@register_keras_serializable(package='Custom', name=None)
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        super(Kistmat_AI, self).__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        self.symbolic_reasoning = SymbolicReasoning()

        self._init_layers()
        self._init_memory_components()
        self._init_output_layers()

        self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)

    def _init_layers(self):
        self.embedding = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=64)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
        self.dropout = keras.layers.Dropout(0.5)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.memory_query = keras.layers.Dense(64, dtype='float32')
        self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
        self.batch_norm = keras.layers.BatchNormalization()

    def _init_memory_components(self):
        self.memory_system = IntegratedMemorySystem({
            'external_memory': {'memory_size': 100, 'key_size': 64, 'value_size': 128},
            'formulative_memory': {'max_formulas': 1000},
            'conceptual_memory': {},
            'short_term_memory': {'capacity': 100},
        
            'long_term_memory': {'capacity': 10000},
            'inference_memory': {'capacity': 500, 'embedding_size': 64}
        })

    def _init_output_layers(self):
        stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                  'high_school1', 'high_school2', 'high_school3', 'university']
        self.output_layers = {stage: keras.layers.Dense(128, activation='linear') for stage in stages}
        self.final_output = keras.layers.Dense(self.output_shape, activation='sigmoid')

    def get_learning_stage(self):
        return tf.strings.strip(self._learning_stage)

    def set_learning_stage(self, stage):
        self._learning_stage.assign(stage.encode())

    @tf.function
    def call(self, inputs, training=False):
        current_stage = self.get_learning_stage()

        if current_stage == 'university':
            x = inputs
        else:
            x = self._process_input(inputs)
            x = self._apply_attention(x)
            x = self._query_memories(x)

        x = self._apply_reasoning(x, training)
        return self._generate_output(x, current_stage, training)

    def _process_input(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        return self.lstm2(x)

    def _apply_attention(self, x):
        x_reshaped = tf.expand_dims(x, axis=1)
        context = self.attention(x_reshaped, x_reshaped, x_reshaped)
        return tf.squeeze(context, axis=1)

    def _query_memories(self, x):
        query = self.memory_query(x)
        memory_results = self.memory_system.process_input({
            'external_query': query,
            'formulative_query': (query, self._extract_relevant_terms(query)),
            'conceptual_query': query,
            'short_term_query': query,
            'long_term_query': query,
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
        memory_outputs = [tf.expand_dims(m, axis=1) if len(m.shape) == 2 else m for m in memory_outputs]
        combined_memory = tf.concat(memory_outputs, axis=1)
        return tf.reduce_mean(combined_memory, axis=1)

    def _apply_reasoning(self, x, training):
        x = self.reasoning_layer(x)
        if training:
            self._update_memories(x)
        return self.batch_norm(x)

    def _generate_output(self, x, current_stage, training):
        x = self.output_layers[current_stage](x)
        if training and current_stage != 'university':
            self._update_memories(x)
        return self.final_output(x)

    def _update_memories(self, x):
        update_data = {
            'external_memory': {'data': (self.memory_query(x), x)},
            'formulative_memory': {'data': (x, self._extract_relevant_terms(x))},
            'conceptual_memory': {'data': (self.memory_query(x), x)},
            'short_term_memory': {'data': x},
            'long_term_memory': {'data': x, 'metadata': {'importance': tf.reduce_mean(tf.abs(x))}},
            'inference_memory': {'data': (x, tf.nn.sigmoid(tf.reduce_mean(x)))}
        }
        self.memory_system.update_memories(update_data)

    def _extract_relevant_terms(self, query):
        current_stage = self.get_learning_stage()
        
        def extract_terms(term_indices):
            return tf.reduce_sum(tf.gather(query, term_indices, axis=-1), axis=-1)

        elementary_indices = [0, 1, 2, 3, 4, 5]  # Indices corresponding to elementary terms
        junior_high_indices = [6, 7, 8, 9, 10, 11]  # Indices for junior high terms
        high_school_indices = [12, 13, 14, 15, 16, 17]  # Indices for high school terms
        university_indices = [18, 19, 20, 21, 22, 23]  # Indices for university terms

        return tf.switch_case(
            tf.math.logical_or(
                tf.equal(current_stage, 'elementary1'),
                tf.math.logical_or(
                    tf.equal(current_stage, 'elementary2'),
                    tf.equal(current_stage, 'elementary3')
                )
            ), {
                0: lambda: extract_terms(elementary_indices),
                1: lambda: extract_terms(junior_high_indices),
                2: lambda: extract_terms(high_school_indices)
            }, default=lambda: extract_terms(university_indices))

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "vocab_size": self.vocab_size,
            "learning_stage": self.get_learning_stage()
        })
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop("input_shape", (MAX_LENGTH,))
        output_shape = config.pop("output_shape", 1)
        vocab_size = config.pop("vocab_size", VOCAB_SIZE)
        learning_stage = config.pop("learning_stage", "elementary1")

        instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
        instance.set_learning_stage(learning_stage)
        return instance

    def solve_problem(self, problem):
        if isinstance(problem, str):  # Assuming string problems are symbolic
            return self.symbolic_reasoning.solve_equation(problem)

    def save_memory_state(self, directory: str) -> None:
        self.memory_system.save_state(directory)

    def load_memory_state(self, directory: str) -> None:
        self.memory_system.load_state(directory)