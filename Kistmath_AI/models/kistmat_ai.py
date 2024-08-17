import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable

from Kistmath_AI.models.external_memory import ExternalMemory, FormulativeMemory, ConceptualMemory, ShortTermMemory, LongTermMemory, InferenceMemory
from Kistmath_AI.config.settings import VOCAB_SIZE, MAX_LENGTH
from Kistmath_AI.models.symbolic_reasoning import SymbolicReasoning

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
        self.external_memory = ExternalMemory(memory_size=100, key_size=64, value_size=128)
        self.formulative_memory = FormulativeMemory(max_formulas=1000)
        self.conceptual_memory = ConceptualMemory()
        self.short_term_memory = ShortTermMemory(capacity=100)
        self.long_term_memory = LongTermMemory(capacity=10000)
        self.inference_memory = InferenceMemory(capacity=500, embedding_size=64)

    def _init_output_layers(self):
        stages = ['elementary1', 'elementary2', 'elementary3', 'junior_high1', 'junior_high2', 
                  'high_school1', 'high_school2', 'high_school3', 'university']
        self.output_layers = {stage: keras.layers.Dense(128, activation='linear') for stage in stages}
        self.final_output = keras.layers.Dense(self.output_shape, activation='sigmoid')

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
        memory_outputs = [
            self.external_memory.query(query),
            self.query_formulative_memory(query),
            self.query_conceptual_memory(query),
            self.short_term_memory.query_recent_memories(query),
            self.long_term_memory.query_important_memories(query),
            self.inference_memory.query_confident_inferences(query)
        ]
        memory_outputs = [tf.expand_dims(m, axis=1) if len(m.shape) == 2 else m for m in memory_outputs]
        combined_memory = tf.concat(memory_outputs, axis=1)
        return tf.reduce_mean(combined_memory, axis=1)

    def _apply_reasoning(self, x, training):
        x = self.reasoning_layer(x)
        if training:
            self.update_memories(x, x)
        return self.batch_norm(x)

    def _generate_output(self, x, current_stage, training):
        x = self.output_layers[current_stage](x)
        if training and current_stage != 'university':
            self.external_memory.update(self.memory_query(x), x)
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

    def add_formula(self, formula: tf.Tensor, terms: list) -> None:
        self.formulative_memory.add_formula(formula, terms)

    def query_similar_formulas(self, query: tf.Tensor, query_terms: list, top_k: int = 5) -> tf.Tensor:
        return self.formulative_memory.query_similar_formulas(query, query_terms, top_k)

    def get_formula_terms(self, index: int) -> list:
        return self.formulative_memory.get_formula_terms(index)

    def add_concept(self, key: tf.Tensor, concept: tf.Tensor) -> None:
        self.conceptual_memory.add_concept(key, concept)

    def get_concept(self, key_embedding: tf.Tensor) -> tf.Tensor:
        return self.conceptual_memory.get_concept(key_embedding)

    def query_similar_concepts(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        return self.conceptual_memory.query_similar_concepts(query, top_k)

    def add_short_term_memory(self, data: tf.Tensor) -> None:
        self.short_term_memory.add_memory(data)

    def get_short_term_memory(self) -> list:
        return self.short_term_memory.get_memory()

    def query_recent_memories(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        return self.short_term_memory.query_recent_memories(query, top_k)

    def add_long_term_memory(self, data: tf.Tensor, importance: float = 1.0) -> None:
        self.long_term_memory.add_memory(data, importance)

    def get_long_term_memory(self) -> list:
        return self.long_term_memory.get_memory()

    def query_important_memories(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        return self.long_term_memory.query_important_memories(query, top_k)

    def add_inference(self, inference: tf.Tensor, confidence: float = 0.8) -> None:
        self.inference_memory.add_inference(inference, confidence)

    def get_inferences(self) -> list:
        return self.inference_memory.get_inferences()

    def query_confident_inferences(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        return self.inference_memory.query_confident_inferences(query, top_k)

    def query_formulative_memory(self, query):
        # Convert query tensor to a fixed-size embedding
        query_embedding = tf.reduce_mean(query, axis=-1)
        
        # Adjust the reshape operation based on the actual shape of query_embedding
        query_shape = tf.shape(query_embedding)
        query_embedding = tf.reshape(query_embedding, [query_shape[0], -1])
        
        # Check if formula_embeddings is empty
        if tf.shape(self.formulative_memory.formula_embeddings)[0] == 0:
            # Return a placeholder tensor of the correct shape if empty
            return tf.zeros([query_shape[0], 64], dtype=tf.float32)

        # Extract relevant terms from the query
        relevant_terms = self._extract_relevant_terms(query)
        
        # Query the formulative memory
        similar_formulas = self.formulative_memory.query_similar_formulas(query_embedding, relevant_terms)
        
        return similar_formulas

    def _extract_relevant_terms(self, query):
        # This is a placeholder for more sophisticated term extraction
        current_stage = self.get_learning_stage()
        
        if current_stage in ['elementary1', 'elementary2', 'elementary3']:
            basic_math_terms = ['add', 'subtract', 'multiply', 'divide', 'number', 'equal']
            return [term for term in basic_math_terms if term in tf.strings.lower(query)]
        
        elif current_stage in ['junior_high1', 'junior_high2']:
            algebra_terms = ['variable', 'equation', 'function', 'graph', 'slope', 'intercept']
            return [term for term in algebra_terms if term in tf.strings.lower(query)]
        
        elif current_stage in ['high_school1', 'high_school2', 'high_school3']:
            advanced_math_terms = ['derivative', 'integral', 'limit', 'vector', 'matrix', 'probability']
            return [term for term in advanced_math_terms if term in tf.strings.lower(query)]
        
        else:  # university
            all_math_terms = ['theorem', 'proof', 'algorithm', 'optimization', 'analysis', 'topology']
            return [term for term in all_math_terms if term in tf.strings.lower(query)]


    def query_conceptual_memory(self, query):
        similar_concepts = self.conceptual_memory.query_similar_concepts(query)
        return similar_concepts

    def update_memories(self, key, x):
        self.add_concept(key, x)
        self.add_formula(key, ["term1", "term2", "term3"])  # Using dummy terms
        self.add_short_term_memory(x)
        
        importance = tf.reduce_mean(tf.abs(x))
        self.add_long_term_memory(x, importance=importance)
        
        confidence = tf.nn.sigmoid(tf.reduce_mean(x))
        self.add_inference(x, confidence=confidence)

    def get_memory_state(self):
        return self.external_memory.get_memory_state()