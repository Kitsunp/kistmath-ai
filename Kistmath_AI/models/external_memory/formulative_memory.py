import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryWarning, MemoryCritical
from sklearn.neighbors import NearestNeighbors
import logging
import pickle

logger = logging.getLogger(__name__)

class FormulativeMemory(MemoryComponent):
    def __init__(self, max_formulas: int = 1000, embedding_size: int = 64):
        if max_formulas <= 0 or embedding_size <= 0:
            raise MemoryError("Invalid FormulativeMemory parameters: sizes must be positive integers")
        self.max_formulas = max_formulas
        self.embedding_size = embedding_size
        self.formula_embeddings = tf.Variable(tf.zeros([0, embedding_size], dtype=tf.float32))
        self.inverted_index = {}
        self.formula_terms = []
        self.nn_index = None

    def add(self, data: Tuple[tf.Tensor, List[str]], metadata: Optional[Dict] = None) -> None:
        try:
            formula, terms = data
            if self.formula_embeddings.shape[0] >= self.max_formulas:
                self._remove_oldest_formula()

            formula = tf.reshape(formula, [-1, self.embedding_size])
            embedding = tf.reduce_mean(formula, axis=0, keepdims=True)
            self.formula_embeddings = tf.concat([self.formula_embeddings, embedding], axis=0)

            formula_index = self.formula_embeddings.shape[0] - 1
            self.formula_terms.append(terms)
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(formula_index)

            self._update_nn_index()
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input shape for FormulativeMemory add: {e}")
            raise MemoryError(f"Input shape mismatch in FormulativeMemory add: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in FormulativeMemory add: {e}")
            raise MemoryCritical(f"Critical error in FormulativeMemory add operation: {e}")

    def query(self, query: Tuple[tf.Tensor, List[str]], top_k: int = 5) -> Tuple[tf.Tensor, List[int]]:
        try:
            query_embedding, query_terms = query
            if self.formula_embeddings.shape[0] == 0:
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
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for FormulativeMemory query: {e}")
            raise MemoryError(f"Query operation failed in FormulativeMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in FormulativeMemory query: {e}")
            raise MemoryCritical(f"Critical error in FormulativeMemory query operation: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'formula_embeddings': self.formula_embeddings.numpy(),
                'inverted_index': self.inverted_index,
                'formula_terms': self.formula_terms
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from FormulativeMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from FormulativeMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.formula_embeddings = tf.Variable(shared_data['formula_embeddings'])
            self.inverted_index = shared_data['inverted_index']
            self.formula_terms = shared_data['formula_terms']
            self._update_nn_index()
        except KeyError as e:
            logger.error(f"Missing key in shared data for FormulativeMemory update: {e}")
            raise MemoryError(f"Invalid shared data format for FormulativeMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating FormulativeMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in FormulativeMemory update from shared data: {e}")

    def _remove_oldest_formula(self) -> None:
        try:
            self.formula_embeddings = tf.Variable(self.formula_embeddings[1:])
            removed_terms = self.formula_terms.pop(0)
            for term in removed_terms:
                self.inverted_index[term].pop(0)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
        except IndexError as e:
            logger.warning(f"Attempted to remove formula from empty FormulativeMemory: {e}")
            raise MemoryWarning(f"Cannot remove formula from empty FormulativeMemory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in FormulativeMemory _remove_oldest_formula: {e}")
            raise MemoryError(f"Failed to remove oldest formula in FormulativeMemory: {e}")

    def _update_nn_index(self) -> None:
        try:
            if self.formula_embeddings.shape[0] > 0:
                n_neighbors = min(5, self.formula_embeddings.shape[0])
                self.nn_index = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm='ball_tree').fit(self.formula_embeddings.numpy())
        except ValueError as e:
            logger.warning(f"Invalid input for NearestNeighbors in FormulativeMemory: {e}")
            raise MemoryWarning(f"Failed to update NN index in FormulativeMemory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating NN index in FormulativeMemory: {e}")
            raise MemoryError(f"Critical error updating NN index in FormulativeMemory: {e}")

    def save(self, filepath: str) -> None:
        try:
            state = {
                'formula_embeddings': self.formula_embeddings.numpy(),
                'inverted_index': self.inverted_index,
                'formula_terms': self.formula_terms
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except IOError as e:
            logger.error(f"IO error while saving FormulativeMemory: {e}")
            raise MemoryError(f"Failed to save FormulativeMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving FormulativeMemory: {e}")
            raise MemoryCritical(f"Critical error saving FormulativeMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.formula_embeddings = tf.Variable(state['formula_embeddings'])
            self.inverted_index = state['inverted_index']
            self.formula_terms = state['formula_terms']
            self._update_nn_index()
        except IOError as e:
            logger.error(f"IO error while loading FormulativeMemory: {e}")
            raise MemoryError(f"Failed to load FormulativeMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading FormulativeMemory: {e}")
            raise MemoryError(f"Corrupted FormulativeMemory data in {filepath}: {e}")
        except KeyError as e:
            logger.error(f"Missing key in loaded FormulativeMemory data: {e}")
            raise MemoryError(f"Incomplete FormulativeMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading FormulativeMemory: {e}")
            raise MemoryCritical(f"Critical error loading FormulativeMemory: {e}")