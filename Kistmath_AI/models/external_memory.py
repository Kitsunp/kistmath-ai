import tensorflow as tf
from typing import List, Tuple, Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, k):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.children.insert(0, root)
            self._split_child(temp, 0)
            self._insert_non_full(temp, k)
        else:
            self._insert_non_full(root, k)

    def _split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = BTreeNode(y.leaf)
        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t: (2 * t) - 1]
        y.keys = y.keys[0: t - 1]
        if not y.leaf:
            z.children = y.children[t: 2 * t]
            y.children = y.children[0: t]

    def _insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            while i >= 0 and k[0] < x.keys[i][0]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k[0] < x.keys[i][0]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self._split_child(x, i)
                if k[0] > x.keys[i][0]:
                    i += 1
            self._insert_non_full(x.children[i], k)

    def search(self, k, x=None):
        if x is not None:
            i = 0
            while i < len(x.keys) and k > x.keys[i][0]:
                i += 1
            if i < len(x.keys) and k == x.keys[i][0]:
                return x.keys[i][1]
            elif x.leaf:
                return None
            else:
                return self.search(k, x.children[i])
        else:
            return self.search(k, self.root)

class ExternalMemory:
    def __init__(self, memory_size: int = 100, key_size: int = 64, value_size: int = 128):
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.btree = BTree(t=5)  # Adjust t based on your needs
        self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))

    @tf.function
    def query(self, query_key: tf.Tensor) -> tf.Tensor:
        query_key = tf.ensure_shape(query_key, [None, self.key_size])
        results = []
        for key in query_key:
            key_sum = tf.reduce_sum(key).numpy()
            value = self.btree.search(key_sum)
            results.append(value if value is not None else tf.zeros([self.value_size], dtype=tf.float32))
        
        retrieved_values = tf.stack(results)
        
        # Update usage based on retrieval
        self.usage.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(retrieved_values, 0), tf.float32), axis=-1))
        
        return retrieved_values

    @tf.function
    def update(self, key: tf.Tensor, value: tf.Tensor) -> None:
        key = tf.ensure_shape(key, [None, self.key_size])
        value = tf.ensure_shape(value, [None, self.value_size])
        
        for i in range(tf.shape(key)[0]):
            key_sum = tf.reduce_sum(key[i]).numpy()
            self.btree.insert((key_sum, value[i].numpy()))
        
        # Decay usage and normalize
        self.usage.assign(self.usage * 0.99)
        self.usage.assign(self.usage / tf.reduce_sum(self.usage))

    def get_memory_state(self) -> Tuple[Any, tf.Tensor]:
        return self.btree, self.usage

class FormulativeMemory:
    def __init__(self, max_formulas: int = 1000):
        self.max_formulas = max_formulas
        self.formula_embeddings = tf.Variable(tf.zeros([0, 64], dtype=tf.float32))
        self.inverted_index = {}
        self.formula_terms = []

    def add_formula(self, formula: tf.Tensor, terms: List[str]) -> None:
        if tf.shape(self.formula_embeddings)[0] >= self.max_formulas:
            self.formula_embeddings = tf.Variable(self.formula_embeddings[1:])
            removed_terms = self.formula_terms.pop(0)
            for term in removed_terms:
                self.inverted_index[term].pop(0)
        
        formula = tf.reshape(formula, [-1, 64])
        embedding = tf.reduce_mean(formula, axis=0, keepdims=True)
        self.formula_embeddings = tf.concat([self.formula_embeddings, embedding], axis=0)
        
        formula_index = tf.shape(self.formula_embeddings)[0] - 1
        self.formula_terms.append(terms)
        for term in terms:
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append(formula_index)

    def query_similar_formulas(self, query: tf.Tensor, query_terms: List[str], top_k: int = 5) -> Tuple[tf.Tensor, List[int]]:
        if tf.shape(self.formula_embeddings)[0] == 0:
            return tf.zeros([0, 64], dtype=tf.float32), []
        
        # Use inverted index to get candidate formulas
        candidate_indices = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_indices.update(self.inverted_index[term])
        
        if not candidate_indices:
            return tf.zeros([0, 64], dtype=tf.float32), []
        
        candidate_embeddings = tf.gather(self.formula_embeddings, list(candidate_indices))
        
        similarities = tf.matmul(tf.reshape(query, [1, -1]), candidate_embeddings, transpose_b=True)
        top_k = min(top_k, tf.shape(candidate_embeddings)[0])
        _, top_indices = tf.nn.top_k(similarities[0], k=top_k)
        
        result_indices = [list(candidate_indices)[i] for i in top_indices.numpy()]
        return tf.gather(candidate_embeddings, top_indices), result_indices

    def get_formula_terms(self, index: int) -> List[str]:
        return self.formula_terms[index]

class ConceptualMemory:
    def __init__(self):
        self.concepts = {}
        self.concept_embeddings = tf.Variable(tf.zeros([0, 64], dtype=tf.float32))
        self.nn_index = None

    def add_concept(self, key: tf.Tensor, concept: tf.Tensor) -> None:
        key = tf.reshape(key, [-1, 64])
        key_embedding = tf.reduce_mean(key, axis=0, keepdims=True)
        self.concept_embeddings = tf.concat([self.concept_embeddings, key_embedding], axis=0)
        self.concepts[tf.reduce_sum(key_embedding).numpy()] = concept
        
        # Rebuild the nearest neighbors index
        self._build_nn_index()

    def _build_nn_index(self):
        if tf.shape(self.concept_embeddings)[0] > 0:
            self.nn_index = NearestNeighbors(n_neighbors=min(5, tf.shape(self.concept_embeddings)[0]), 
                                             algorithm='ball_tree').fit(self.concept_embeddings.numpy())

    def query_similar_concepts(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        query_embedding = tf.reduce_mean(query, axis=-1)
        query_embedding = tf.reshape(query_embedding, [1, 64])
        
        if self.nn_index is None:
            return tf.zeros([0, 64], dtype=tf.float32)
        
        distances, indices = self.nn_index.kneighbors(query_embedding.numpy(), 
                                                      n_neighbors=min(top_k, tf.shape(self.concept_embeddings)[0]))
        
        return tf.gather(self.concept_embeddings, indices[0])

    def get_concept(self, key_embedding: tf.Tensor) -> tf.Tensor:
        key_sum = tf.reduce_sum(key_embedding).numpy()
        return self.concepts.get(key_sum, None)

class ShortTermMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory = []

    def add_memory(self, data: tf.Tensor) -> None:
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(data)

    def get_memory(self) -> List[tf.Tensor]:
        return self.memory

    def query_recent_memories(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) for mem in self.memory])
        _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
        return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])

class LongTermMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.importance_scores = []

    def add_memory(self, data: tf.Tensor, importance: float = 1.0) -> None:
        if len(self.memory) >= self.capacity:
            min_importance_index = np.argmin(self.importance_scores)
            if importance > self.importance_scores[min_importance_index]:
                self.memory[min_importance_index] = data
                self.importance_scores[min_importance_index] = importance
        else:
            self.memory.append(data)
            self.importance_scores.append(importance)
        
        # Ordenar las memorias por importancia
        sorted_indices = np.argsort(self.importance_scores)[::-1]
        self.memory = [self.memory[i] for i in sorted_indices]
        self.importance_scores = [self.importance_scores[i] for i in sorted_indices]

    def get_memory(self) -> List[tf.Tensor]:
        return self.memory

    def query_important_memories(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) * imp for mem, imp in zip(self.memory, self.importance_scores)])
        _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
        return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])

class InferenceMemory:
    def __init__(self, capacity: int = 500, embedding_size: int = 64):
        self.capacity = capacity
        self.embedding_size = embedding_size
        self.inferences = []
        self.confidence_scores = []
        self.index = AnnoyIndex(embedding_size, 'angular')
        self.current_index = 0

    def add_inference(self, inference: tf.Tensor, confidence: float) -> None:
        if len(self.inferences) >= self.capacity:
            min_confidence_index = np.argmin(self.confidence_scores)
            if confidence > self.confidence_scores[min_confidence_index]:
                self.inferences[min_confidence_index] = inference
                self.confidence_scores[min_confidence_index] = confidence
                self.index.delete_item(min_confidence_index)
                self.index.add_item(min_confidence_index, inference.numpy())
        else:
            self.inferences.append(inference)
            self.confidence_scores.append(confidence)
            self.index.add_item(self.current_index, inference.numpy())
            self.current_index += 1
        
        if self.current_index % 100 == 0:  # Rebuild index periodically
            self.index.build(10)  # 10 trees - adjust based on your needs

    def get_inferences(self) -> List[tf.Tensor]:
        return self.inferences

    def query_confident_inferences(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        if len(self.inferences) == 0:
            return tf.zeros([0, self.embedding_size], dtype=tf.float32)
        
        query_vector = tf.reshape(query, [-1]).numpy()
        nearest_indices = self.index.get_nns_by_vector(query_vector, top_k, include_distances=True)
        
        top_indices, distances = nearest_indices
        confidences = [self.confidence_scores[i] for i in top_indices]
        
        # Combine distance and confidence scores
        combined_scores = [1 / (d + 1e-5) * c for d, c in zip(distances, confidences)]
        
        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        result_indices = [top_indices[i] for i in sorted_indices]
        
        return tf.stack([tf.reshape(self.inferences[i], [-1]) for i in result_indices])