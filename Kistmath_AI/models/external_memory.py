import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
from typing import List, Tuple, Any, Dict, Optional
import pickle
import logging
from abc import ABC, abstractmethod

tf.data.experimental.enable_debug_mode()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryComponent(ABC):
    @abstractmethod
    def add(self, data: Any, metadata: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def query(self, query: Any, top_k: int = 5) -> List[Any]:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @abstractmethod
    def get_shareable_data(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        pass

class BTreeNode:
    def __init__(self, leaf: bool = False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree(MemoryComponent):
    def __init__(self, t: int):
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, k: Tuple[float, Any]) -> None:
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.children.insert(0, root)
            self._split_child(temp, 0)
            self._insert_non_full(temp, k)
        else:
            self._insert_non_full(root, k)

    def _split_child(self, x: BTreeNode, i: int) -> None:
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

    def _insert_non_full(self, x: BTreeNode, k: Tuple[float, Any]) -> None:
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

    def search(self, k: tf.Tensor, x: Optional[BTreeNode] = None) -> tf.Tensor:
        if x is None:
            x = self.root
        i = 0
        while i < len(x.keys) and k[0] > x.keys[i][0]:
            i += 1
        if i < len(x.keys) and tf.equal(k[0], x.keys[i][0]):
            return x.keys[i][1]
        elif x.leaf:
            return None
        else:
            return self.search(k[0], x.children[i])

    def add(self, data: Tuple[float, Any], metadata: Optional[Dict] = None) -> None:
        self.insert(data)

    def query(self, query: float, top_k: int = 1) -> List[Any]:
        result = self.search(query)
        return [result] if result is not None else []

    def save(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'rb') as f:
            loaded_tree = pickle.load(f)
            self.__dict__.update(loaded_tree.__dict__)

    def get_shareable_data(self) -> Dict[str, Any]:
        return {
            'root': self._serialize_node(self.root),
            't': self.t
        }

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        self.t = shared_data['t']
        self.root = self._deserialize_node(shared_data['root'])

    def _serialize_node(self, node: BTreeNode) -> Dict[str, Any]:
        return {
            'leaf': node.leaf,
            'keys': node.keys,
            'children': [self._serialize_node(child) for child in node.children] if not node.leaf else []
        }

    def _deserialize_node(self, data: Dict[str, Any]) -> BTreeNode:
        node = BTreeNode(data['leaf'])
        node.keys = data['keys']
        node.children = [self._deserialize_node(child) for child in data['children']] if not data['leaf'] else []
        return node
    
class ExternalMemory(MemoryComponent):
    def __init__(self, memory_size: int = 100, key_size: int = 64, value_size: int = 128):
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.btree = BTree(t=5)
        self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))
        self.memory_embeddings = tf.Variable(tf.zeros([memory_size, value_size], dtype=tf.float32))  # Add this line

    def add(self, data: Tuple[tf.Tensor, tf.Tensor], metadata: Optional[Dict] = None) -> None:
        key, value = data
        key = tf.ensure_shape(key, [None, self.key_size])
        value = tf.ensure_shape(value, [None, self.value_size])
        
        for i in range(tf.shape(key)[0]):
            key_sum = tf.reduce_sum(key[i])
            self.btree.insert((key_sum.numpy(), value[i].numpy()))
        
        # Update memory_embeddings
        current_size = tf.shape(self.memory_embeddings)[0]
        new_size = tf.minimum(self.memory_size, current_size + tf.shape(value)[0])
        self.memory_embeddings = tf.Variable(tf.concat([self.memory_embeddings[:new_size - tf.shape(value)[0]], value[:new_size - current_size]], axis=0))
        
        self._update_usage()
    def query(self, query_embedding: tf.Tensor, query_terms: Optional[tf.Tensor] = None, top_k: int = 5) -> Tuple[tf.Tensor, tf.Tensor]:
        if tf.equal(tf.shape(self.memory_embeddings)[0], 0):  # Change this line
            return tf.zeros([0, self.value_size], dtype=tf.float32), tf.constant([], dtype=tf.int32)
        
        if query_terms is None:
            # Si no se proporcionan query_terms, usar todos los índices disponibles
            candidate_indices = tf.range(tf.shape(self.formula_embeddings)[0])
        else:
            query_terms = tf.strings.lower(tf.strings.strip(query_terms))
            
            def check_term(term):
                return tf.reduce_any(tf.equal(tf.constant(list(self.inverted_index.keys())), term))
            
            mask = tf.map_fn(check_term, query_terms, dtype=tf.bool)
            valid_terms = tf.boolean_mask(query_terms, mask)
            
            def get_indices(term):
                return tf.constant(self.inverted_index[term.numpy().decode('utf-8')], dtype=tf.int32)
            
            candidate_indices = tf.map_fn(get_indices, valid_terms, dtype=tf.int32)
            candidate_indices = tf.unique(tf.reshape(candidate_indices, [-1])).y
        
        if tf.equal(tf.size(candidate_indices), 0):
            return tf.zeros([0, self.embedding_size], dtype=tf.float32), tf.constant([], dtype=tf.int32)
        
        candidate_embeddings = tf.gather(self.formula_embeddings, candidate_indices)
        
        similarities = tf.matmul(tf.reshape(query_embedding, [1, -1]), candidate_embeddings, transpose_b=True)
        top_k = tf.minimum(top_k, tf.shape(candidate_embeddings)[0])
        _, top_indices = tf.nn.top_k(similarities[0], k=top_k)
        
        result_indices = tf.gather(candidate_indices, top_indices)
        return tf.gather(candidate_embeddings, top_indices), result_indices
    def get_formula_terms(self, index: int) -> List[str]:
        try:
            return self.formula_terms[index]
        except IndexError as e:
            logger.error(f"Error en get_formula_terms: Índice {index} fuera de rango")
            return []
        except Exception as e:
            logger.error(f"Error inesperado en get_formula_terms: {e}")
            return []
            
    def _update_usage(self, retrieved_values: Optional[tf.Tensor] = None) -> None:
        if retrieved_values is not None:
            batch_size = tf.shape(retrieved_values)[0]
            usage_update = tf.reduce_sum(tf.cast(tf.not_equal(retrieved_values, 0), tf.float32), axis=-1)
            
            # Asegurarse de que usage_update tenga la misma forma que self.usage
            usage_update = tf.reduce_mean(usage_update)
            usage_update = tf.repeat(usage_update, self.memory_size)
            
            self.usage.assign_add(usage_update)
        
        # Decay usage and normalize
        self.usage.assign(self.usage * 0.99)
        total_usage = tf.reduce_sum(self.usage)
        if total_usage > 0:
            self.usage.assign(self.usage / total_usage)

    def save(self, filepath: str) -> None:
        state = {
            'btree': self.btree,
            'usage': self.usage.numpy()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.btree = state['btree']
        self.usage = tf.Variable(state['usage'])
    def get_shareable_data(self) -> Dict[str, Any]:
        return {
            'memory_embeddings': self.memory_embeddings.numpy()
        }

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        if 'memory_embeddings' in shared_data:
            self.memory_embeddings = tf.Variable(shared_data['memory_embeddings'])

class FormulativeMemory(MemoryComponent):
    def __init__(self, max_formulas: int = 1000, embedding_size: int = 64):
        self.max_formulas = max_formulas
        self.embedding_size = embedding_size
        self.formula_embeddings = tf.Variable(tf.zeros([0, embedding_size], dtype=tf.float32))
        self.inverted_index = {}
        self.formula_terms = []
        self.nn_index = None

    def add(self, data: Tuple[tf.Tensor, List[str]], metadata: Optional[Dict] = None) -> None:
        formula, terms = data
        if tf.shape(self.formula_embeddings)[0] >= self.max_formulas:
            self._remove_oldest_formula()
        
        formula = tf.reshape(formula, [-1, self.embedding_size])
        embedding = tf.reduce_mean(formula, axis=0, keepdims=True)
        self.formula_embeddings = tf.concat([self.formula_embeddings, embedding], axis=0)
        
        formula_index = tf.shape(self.formula_embeddings)[0] - 1
        self.formula_terms.append(terms)
        for term in terms:
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append(formula_index.numpy())
        
        self._update_nn_index()

    def _remove_oldest_formula(self) -> None:
        self.formula_embeddings = tf.Variable(self.formula_embeddings[1:])
        removed_terms = self.formula_terms.pop(0)
        for term in removed_terms:
            self.inverted_index[term].pop(0)
            if not self.inverted_index[term]:
                del self.inverted_index[term]

    def _update_nn_index(self) -> None:
        if tf.shape(self.formula_embeddings)[0] > 0:
            n_neighbors = min(5, tf.shape(self.formula_embeddings)[0])
            self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.formula_embeddings.numpy())

    def query(self, query: Tuple[tf.Tensor, List[str]], top_k: int = 5) -> Tuple[tf.Tensor, List[int]]:
        query_embedding, query_terms = query
        if tf.shape(self.formula_embeddings)[0] == 0:
            return tf.zeros([0, self.embedding_size], dtype=tf.float32), []
        
        candidate_indices = set()
        for term in query_terms:
            term_str = term.numpy().decode('utf-8') if isinstance(term, tf.Tensor) else term
            if term_str in self.inverted_index:
                candidate_indices.update(self.inverted_index[term_str])
        
        if not candidate_indices:
            return tf.zeros([0, self.embedding_size], dtype=tf.float32), []
        
        candidate_embeddings = tf.gather(self.formula_embeddings, list(candidate_indices))
        
        similarities = tf.matmul(tf.reshape(query_embedding, [1, -1]), candidate_embeddings, transpose_b=True)
        top_k = tf.minimum(top_k, tf.shape(candidate_embeddings)[0])
        _, top_indices = tf.nn.top_k(similarities[0], k=top_k)
        
        result_indices = [list(candidate_indices)[i] for i in top_indices.numpy()]
        return tf.gather(candidate_embeddings, top_indices), result_indices

    def get_formula_terms(self, index: int) -> List[str]:
        return self.formula_terms[index]

    def save(self, filepath: str) -> None:
        state = {
            'formula_embeddings': self.formula_embeddings.numpy(),
            'inverted_index': self.inverted_index,
            'formula_terms': self.formula_terms
        }
        with open(filepath, 'wb') as f:
            import pickle
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            import pickle
            state = pickle.load(f)
        self.formula_embeddings = tf.Variable(state['formula_embeddings'])
        self.inverted_index = state['inverted_index']
        self.formula_terms = state['formula_terms']
        self._update_nn_index()

    def get_shareable_data(self) -> Dict[str, Any]:
        return {
            'formula_embeddings': self.formula_embeddings.numpy(),
            'inverted_index': self.inverted_index,
            'formula_terms': self.formula_terms
        }

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        self.formula_embeddings = tf.Variable(shared_data['formula_embeddings'])
        self.inverted_index = shared_data['inverted_index']
        self.formula_terms = shared_data['formula_terms']
        self._update_nn_index()

class ConceptualMemory(MemoryComponent):
    def __init__(self, embedding_size: int = 64):
        self.embedding_size = embedding_size
        self.concepts = {}
        self.concept_embeddings = tf.Variable(tf.zeros([0, embedding_size], dtype=tf.float32))
        self.nn_index = None

    def add(self, data: Tuple[tf.Tensor, tf.Tensor], metadata: Optional[Dict] = None) -> None:
        key, concept = data
        key = tf.reshape(key, [-1, self.embedding_size])
        key_embedding = tf.reduce_mean(key, axis=0, keepdims=True)
        self.concept_embeddings = tf.concat([self.concept_embeddings, key_embedding], axis=0)
        self.concepts[tf.reduce_sum(key_embedding).numpy()] = concept
        
        self._update_nn_index()

    def _update_nn_index(self) -> None:
        if tf.shape(self.concept_embeddings)[0] > 0:
            n_neighbors = min(5, tf.shape(self.concept_embeddings)[0])
            self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.concept_embeddings.numpy())

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        query_embedding = tf.reduce_mean(query, axis=0)
        query_embedding = tf.reshape(query_embedding, [1, self.embedding_size])
        
        if self.nn_index is None:
            return tf.zeros([0, self.embedding_size], dtype=tf.float32)
        
        distances, indices = self.nn_index.kneighbors(query_embedding.numpy(), 
                                                      n_neighbors=min(top_k, tf.shape(self.concept_embeddings)[0]))
        
        return tf.gather(self.concept_embeddings, indices[0])

    def get_concept(self, key_embedding: tf.Tensor) -> tf.Tensor:
        key_sum = tf.reduce_sum(key_embedding).numpy()
        return self.concepts.get(key_sum, None)

    def save(self, filepath: str) -> None:
        state = {
            'concepts': self.concepts,
            'concept_embeddings': self.concept_embeddings.numpy()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.concepts = state['concepts']
        self.concept_embeddings = tf.Variable(state['concept_embeddings'])
        self._update_nn_index()

class ShortTermMemory(MemoryComponent):
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory = []

    def add(self, data: tf.Tensor, metadata: Optional[Dict] = None) -> None:
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(data)

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) for mem in self.memory])
        _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
        return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.memory = pickle.load(f)
        self.memory = self.memory[-self.capacity:]  # Ensure we don't exceed capacity

class LongTermMemory(MemoryComponent):
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.importance_scores = []
        self.time_added = []

    def add(self, data: tf.Tensor, metadata: Optional[Dict] = None) -> None:
        importance = metadata.get('importance', 1.0) if metadata else 1.0
        if len(self.memory) >= self.capacity:
            self._remove_least_important()
        self.memory.append(data)
        self.importance_scores.append(importance)
        self.time_added.append(tf.timestamp())
        self._sort_by_importance()

    def _remove_least_important(self) -> None:
        min_importance_index = np.argmin(self.importance_scores)
        self.memory.pop(min_importance_index)
        self.importance_scores.pop(min_importance_index)
        self.time_added.pop(min_importance_index)

    def _sort_by_importance(self) -> None:
        sorted_indices = np.argsort(self.importance_scores)[::-1]
        self.memory = [self.memory[i] for i in sorted_indices]
        self.importance_scores = [self.importance_scores[i] for i in sorted_indices]
        self.time_added = [self.time_added[i] for i in sorted_indices]

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) * imp for mem, imp in zip(self.memory, self.importance_scores)])
        _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
        return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])

    def update_importance(self, index: int, new_importance: float) -> None:
        if 0 <= index < len(self.importance_scores):
            self.importance_scores[index] = new_importance
            self._sort_by_importance()

    def save(self, filepath: str) -> None:
        state = {
            'memory': self.memory,
            'importance_scores': self.importance_scores,
            'time_added': self.time_added
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.memory = state['memory']
        self.importance_scores = state['importance_scores']
        self.time_added = state['time_added']
        self._sort_by_importance()

class InferenceMemory(MemoryComponent):
    def __init__(self, capacity: int = 500, embedding_size: int = 64):
        self.capacity = capacity
        self.embedding_size = embedding_size
        self.inferences = []
        self.confidence_scores = []
        self.index = AnnoyIndex(embedding_size, 'angular')
        self.current_index = 0

    def add(self, data: Tuple[tf.Tensor, float], metadata: Optional[Dict] = None) -> None:
        inference, confidence = data
        if len(self.inferences) >= self.capacity:
            self._remove_least_confident()
        self.inferences.append(inference)
        self.confidence_scores.append(confidence)
        self.index.add_item(self.current_index, inference.numpy())
        self.current_index += 1
        
        if self.current_index % 100 == 0:
            self._rebuild_index()

    def _remove_least_confident(self) -> None:
        min_confidence_index = np.argmin(self.confidence_scores)
        self.inferences.pop(min_confidence_index)
        self.confidence_scores.pop(min_confidence_index)
        self.index.unbuild()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.index = AnnoyIndex(self.embedding_size, 'angular')
        for i, inference in enumerate(self.inferences):
            self.index.add_item(i, inference.numpy())
        self.index.build(10)  # 10 trees - adjust based on your needs

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        if len(self.inferences) == 0:
            return tf.zeros([0, self.embedding_size], dtype=tf.float32)
        
        query_vector = tf.reshape(query, [-1]).numpy()
        nearest_indices = self.index.get_nns_by_vector(query_vector, top_k, include_distances=True)
        
        top_indices, distances = nearest_indices
        confidences = [self.confidence_scores[i] for i in top_indices]
        
        combined_scores = [1 / (d + 1e-5) * c for d, c in zip(distances, confidences)]
        sorted_indices = np.argsort(combined_scores)[::-1]
        result_indices = [top_indices[i] for i in sorted_indices]
        
        return tf.stack([tf.reshape(self.inferences[i], [-1]) for i in result_indices])

    def update_confidence(self, index: int, new_confidence: float) -> None:
        if 0 <= index < len(self.confidence_scores):
            self.confidence_scores[index] = new_confidence

    def save(self, filepath: str) -> None:
        state = {
            'inferences': self.inferences,
            'confidence_scores': self.confidence_scores,
            'current_index': self.current_index
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        self.index.save(filepath + '.ann')

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.inferences = state['inferences']
        self.confidence_scores = state['confidence_scores']
        self.current_index = state['current_index']
        self.index.load(filepath + '.ann')

class IntegratedMemorySystem:
    def __init__(self, config: Dict[str, Any]):
        self.external_memory = ExternalMemory(**config.get('external_memory', {}))
        self.formulative_memory = FormulativeMemory(**config.get('formulative_memory', {}))
        self.conceptual_memory = ConceptualMemory(**config.get('conceptual_memory', {}))
        self.short_term_memory = ShortTermMemory(**config.get('short_term_memory', {}))
        self.long_term_memory = LongTermMemory(**config.get('long_term_memory', {}))
        self.inference_memory = InferenceMemory(**config.get('inference_memory', {}))

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        
        if 'external_query' in input_data:
            query_embedding = input_data['external_query']
            query_terms = input_data.get('external_query_terms', None)
            results['external_memory'] = self.external_memory.query(query_embedding, query_terms)
        
        if 'formulative_query' in input_data:
            results['formulative_memory'] = self.formulative_memory.query(input_data['formulative_query'])
        
        if 'conceptual_query' in input_data:
            results['conceptual_memory'] = self.conceptual_memory.query(input_data['conceptual_query'])
        
        if 'short_term_query' in input_data:
            results['short_term_memory'] = self.short_term_memory.query(input_data['short_term_query'])
        
        if 'long_term_query' in input_data:
            results['long_term_memory'] = self.long_term_memory.query(input_data['long_term_query'])
        
        if 'inference_query' in input_data:
            results['inference_memory'] = self.inference_memory.query(input_data['inference_query'])
        
        return results
    def share_data(self) -> Dict[str, Dict[str, Any]]:
        shared_data = {}
        for component_name, component in self.__dict__.items():
            if isinstance(component, MemoryComponent):
                shared_data[component_name] = component.get_shareable_data()
        return shared_data

    def update_from_shared_data(self, shared_data: Dict[str, Dict[str, Any]]) -> None:
        for component_name, component_data in shared_data.items():
            if hasattr(self, component_name):
                getattr(self, component_name).update_from_shared_data(component_data)

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Share data between components before processing
        shared_data = self.share_data()
        self.update_from_shared_data(shared_data)
    def update_memories(self, update_data: Dict[str, Any]) -> None:
        if 'external_memory' in update_data:
            self.external_memory.add(**update_data['external_memory'])
        
        if 'formulative_memory' in update_data:
            self.formulative_memory.add(**update_data['formulative_memory'])
        
        if 'conceptual_memory' in update_data:
            self.conceptual_memory.add(**update_data['conceptual_memory'])
        
        if 'short_term_memory' in update_data:
            self.short_term_memory.add(**update_data['short_term_memory'])
        
        if 'long_term_memory' in update_data:
            self.long_term_memory.add(**update_data['long_term_memory'])
        
        if 'inference_memory' in update_data:
            self.inference_memory.add(**update_data['inference_memory'])

    def save_state(self, directory: str) -> None:
        self.external_memory.save(f"{directory}/external_memory.pkl")
        self.formulative_memory.save(f"{directory}/formulative_memory.pkl")
        self.conceptual_memory.save(f"{directory}/conceptual_memory.pkl")
        self.short_term_memory.save(f"{directory}/short_term_memory.pkl")
        self.long_term_memory.save(f"{directory}/long_term_memory.pkl")
        self.inference_memory.save(f"{directory}/inference_memory.pkl")

    def load_state(self, directory: str) -> None:
        self.external_memory.load(f"{directory}/external_memory.pkl")
        self.formulative_memory.load(f"{directory}/formulative_memory.pkl")
        self.conceptual_memory.load(f"{directory}/conceptual_memory.pkl")
        self.short_term_memory.load(f"{directory}/short_term_memory.pkl")
        self.long_term_memory.load(f"{directory}/long_term_memory.pkl")
        self.inference_memory.load(f"{directory}/inference_memory.pkl")
    
    