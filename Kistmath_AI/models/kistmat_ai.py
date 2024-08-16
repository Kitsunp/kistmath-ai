import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow import keras
from keras import ops
from tensorflow.keras.utils import register_keras_serializable
from Kistmath_AI.models.external_memory import ExternalMemory, FormulativeMemory, ConceptualMemory, ShortTermMemory, LongTermMemory, InferenceMemory
from Kistmath_AI.config.settings import VOCAB_SIZE, MAX_LENGTH
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
@register_keras_serializable(package='Custom', name=None)
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        super(Kistmat_AI, self).__init__(name=name, **kwargs)
        
        # Define layers
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=64)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(512))
        self.dropout = keras.layers.Dropout(0.5)
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)
        
        self.memory = ExternalMemory(key_size=64, value_size=128)
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
        self.final_output = keras.layers.Dense(output_shape, activation='linear')

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
            
            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped)
            context = tf.squeeze(context, axis=1)
            
            query = self.memory_query(x)
            memory_output = self.memory.query(query)
            
            x = tf.concat([context, memory_output], axis=-1)
        
        x = self.reasoning_layer(x)
        
        if training:
            x = self.dropout(x)
        x = self.batch_norm(x)
        
        x = self.output_layers[current_stage](x)
        
        if training and current_stage != 'university':
            self.memory.update(query, x)
        
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

    def add_long_term_memory(self, data):
        self.long_term_memory.add_memory(data)

    def get_long_term_memory(self):
        return self.long_term_memory.get_memory()

    def add_inference(self, inference):
        self.inference_memory.add_inference(inference)

    def get_inferences(self):
        return self.inference_memory.get_inferences()