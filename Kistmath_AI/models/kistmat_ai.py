import tensorflow as tf

# Habilitar la modo de ejecución ansiosa para tf.data functions
tf.data.experimental.enable_debug_mode()

tf.config.run_functions_eagerly(True)  # Esto ya lo tenías y está bien

from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable

from Kistmath_AI.models.external_memory import ExternalMemory, FormulativeMemory, ConceptualMemory, ShortTermMemory, LongTermMemory, InferenceMemory
from Kistmath_AI.config.settings import VOCAB_SIZE, MAX_LENGTH
from Kistmath_AI.models.symbolic_reasoning import SymbolicReasoning
import numpy as np
import numpy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@register_keras_serializable(package='Custom', name=None)
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        self.symbolic_reasoning = SymbolicReasoning()
        super(Kistmat_AI, self).__init__(name=name, **kwargs)

        # Define layers
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=64)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
        self.dropout = keras.layers.Dropout(0.5)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        
        # Initialize enhanced memory components
        self.external_memory = ExternalMemory(memory_size=100, key_size=64, value_size=128)
        self.formulative_memory = FormulativeMemory(max_formulas=1000)
        self.conceptual_memory = ConceptualMemory()
        self.short_term_memory = ShortTermMemory(capacity=100)
        self.long_term_memory = LongTermMemory(capacity=10000)
        self.inference_memory = InferenceMemory(capacity=500)
        
        self.memory_query = keras.layers.Dense(64, dtype='float32')
        self.reasoning_layer = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))

        # Output layers for different learning stages
        self.output_layers = {
            'elementary1': keras.layers.Dense(128, activation='linear'),
            'elementary2': keras.layers.Dense(128, activation='linear'),
            'elementary3': keras.layers.Dense(128, activation='linear'),
            'junior_high1': keras.layers.Dense(128, activation='linear'),
            'junior_high2': keras.layers.Dense(128, activation='linear'),
            'high_school1': keras.layers.Dense(128, activation='linear'),
            'high_school2': keras.layers.Dense(128, activation='linear'),
            'high_school3': keras.layers.Dense(128, activation='linear'),
            'university': keras.layers.Dense(128, activation='linear')
        }
        self.final_output = keras.layers.Dense(output_shape, activation='sigmoid')

        self.dropout = keras.layers.Dropout(0.5)
        self.batch_norm = keras.layers.BatchNormalization()

        self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)
    def get_learning_stage(self):
        return self._learning_stage.numpy().decode()

    def set_learning_stage(self, stage):
        self._learning_stage.assign(stage.encode())
    @tf.function
    def call(self, inputs, training=False):
        current_stage = self.get_learning_stage()

        if current_stage == 'university':
            x = inputs
        else:
            x = self.embedding(inputs)
            x = self.lstm1(x)
            x = self.lstm2(x)
            
            # Reshape x for attention
            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped, x_reshaped)
            context = tf.squeeze(context, axis=1)

            query = self.memory_query(x)
            memory_output = self.external_memory.query(query)

            # Query other memory components
            similar_formulas = self.query_formulative_memory(query)
            similar_concepts = self.query_conceptual_memory(query)
            recent_memories = self.short_term_memory.query_recent_memories(query)
            important_memories = self.long_term_memory.query_important_memories(query)
            confident_inferences = self.inference_memory.query_confident_inferences(query)

            # Combine all memory outputs
            all_memory_outputs = [memory_output, similar_formulas, similar_concepts, recent_memories, important_memories, confident_inferences]
            combined_memory = tf.concat(all_memory_outputs, axis=-1)

            x = tf.concat([context, combined_memory], axis=-1)

        x = self.reasoning_layer(x)

        if training:
            # Update all memory components
            key = tf.reduce_sum(tf.cast(inputs, tf.float32), axis=-1, keepdims=True)
            self.update_memories(key, x)

        x = self.batch_norm(x)

        # Using appropriate activation in outputs
        x = self.output_layers[current_stage](x)

        if training and current_stage != 'university':
            self.external_memory.update(query, x)

        return self.final_output(x)

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
        input_shape = config.pop("input_shape", None)
        output_shape = config.pop("output_shape", None)
        vocab_size = config.pop("vocab_size", VOCAB_SIZE)
        learning_stage = config.pop("learning_stage", "elementary1")

        if input_shape is None:
            input_shape = (MAX_LENGTH,)
        if output_shape is None:
            output_shape = 1

        instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
        instance.set_learning_stage(learning_stage)
        return instance

    def add_formula(self, formula):
        self.formulative_memory.add_formula(formula)

    def solve_problem(self, problem):
        if isinstance(problem, str):  # Assuming string problems are symbolic
            return self.symbolic_reasoning.solve_equation(problem)

    def get_formulas(self):
        return self.formulative_memory.get_formulas()

    def add_concept(self, key, concept):
        self.conceptual_memory.add_concept(key, concept)

    def get_concept(self, key):
        return self.conceptual_memory.get_concept(key)

    def add_short_term_memory(self, data):
        self.short_term_memory.add_memory(data)

    def get_short_term_memory(self):
        return self.short_term_memory.get_memory()

    def add_long_term_memory(self, data, importance=1.0):
        self.long_term_memory.add_memory(data, importance)

    def get_long_term_memory(self):
        return self.long_term_memory.get_memory()

    def add_inference(self, inference, confidence=0.8):
        self.inference_memory.add_inference(inference, confidence)

    def get_inferences(self):
        return self.inference_memory.get_inferences()

    def query_similar_formulas(self, query, top_k=5):
        return self.formulative_memory.query_similar_formulas(query, top_k)

    def query_similar_concepts(self, query, top_k=5):
        return self.conceptual_memory.query_similar_concepts(query, top_k)

    def query_recent_memories(self, query, top_k=5):
        return self.short_term_memory.query_recent_memories(query, top_k)

    def query_important_memories(self, query, top_k=5):
        return self.long_term_memory.query_important_memories(query, top_k)

    def query_confident_inferences(self, query, top_k=5):
        return self.inference_memory.query_confident_inferences(query, top_k)
    def query_formulative_memory(self, query):
        # Convert query tensor to string representation
        query_str = tf.strings.as_string(tf.reduce_sum(query, axis=-1))
        query_str = tf.strings.as_string(tf.reduce_sum(query, axis=-1))
        similar_formulas = self.formulative_memory.query_similar_formulas(query_str.numpy().decode())
        return tf.concat(similar_formulas, axis=0)
    def query_conceptual_memory(self, query):
        # Convert query tensor to string representation
        query_str = tf.strings.as_string(tf.reduce_sum(query, axis=-1))
        similar_concepts = self.conceptual_memory.query_similar_concepts(query_str)
        return tf.concat(similar_concepts, axis=0)

    def update_memories(self, key, x):
        key_str = tf.strings.as_string(key)
        self.conceptual_memory.add_concept(key_str.decode(), x.numpy())
        self.formulative_memory.add_formula(key_str)
        self.short_term_memory.add_memory(x)
        
        # Calculate importance for long-term memory
        importance = tf.reduce_mean(tf.abs(x))
        self.long_term_memory.add_memory(x, importance=importance.numpy())
        
        # Calculate confidence for inference memory
        confidence = tf.nn.sigmoid(tf.reduce_mean(x)).numpy()
        self.inference_memory.add_inference(x, confidence=confidence)