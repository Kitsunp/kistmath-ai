import tensorflow as tf

class ExternalMemory:
    def __init__(self, memory_size=100, key_size=64, value_size=128):
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.keys = tf.Variable(tf.random.normal([memory_size, key_size], dtype=tf.float32))
        self.values = tf.Variable(tf.zeros([memory_size, value_size], dtype=tf.float32))
        self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))

    @tf.function
    def query(self, query_key):
        query_key = tf.cast(query_key, tf.float32)
        similarities = tf.matmul(query_key, self.keys, transpose_b=True)
        weights = tf.nn.sigmoid(similarities)
        return tf.matmul(weights, self.values)

    @tf.function
    def update(self, key, value):
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)
        key = tf.reshape(key, [-1, self.key_size])
        value = tf.reshape(value, [-1, self.value_size])
        
        index = tf.argmin(self.usage)
        
        self.keys[index].assign(key[0])
        self.values[index].assign(value[0])
        self.usage[index].assign(1.0)
        
        # Decay usage
        self.usage.assign(self.usage * 0.99)

class FormulativeMemory:
    def __init__(self):
        self.formulas = []

    def add_formula(self, formula):
        self.formulas.append(formula)

    def get_formulas(self):
        return self.formulas

class ConceptualMemory:
    def __init__(self):
        self.concepts = {}

    def add_concept(self, key, concept):
        self.concepts[key] = concept

    def get_concept(self, key):
        return self.concepts.get(key, None)

class ShortTermMemory:
    def __init__(self):
        self.memory = []

    def add_memory(self, data):
        self.memory.append(data)

    def get_memory(self):
        return self.memory

class LongTermMemory:
    def __init__(self):
        self.memory = []

    def add_memory(self, data):
        self.memory.append(data)

    def get_memory(self):
        return self.memory

class InferenceMemory:
    def __init__(self):
        self.inferences = []

    def add_inference(self, inference):
        self.inferences.append(inference)

    def get_inferences(self):
        return self.inferences