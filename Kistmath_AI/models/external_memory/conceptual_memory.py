import tensorflow as tf
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryCritical, MemoryWarning
from sklearn.neighbors import NearestNeighbors
import logging
import pickle
logger = logging.getLogger(__name__)

class ConceptualMemory(MemoryComponent):
    def __init__(self, embedding_size: int = 64):
        if embedding_size <= 0:
            raise MemoryError("Invalid ConceptualMemory parameter: embedding_size must be a positive integer")
        self.embedding_size = embedding_size
        self.concepts = {}
        self.concept_embeddings = tf.Variable(tf.zeros([0, embedding_size], dtype=tf.float32))
        self.nn_index = None

    def add(self, data: Tuple[tf.Tensor, tf.Tensor], metadata: Optional[Dict] = None) -> None:
        try:
            key, concept = data
            key = tf.reshape(key, [-1, self.embedding_size])
            key_embedding = tf.reduce_mean(key, axis=0, keepdims=True)
            self.concept_embeddings = tf.concat([self.concept_embeddings, key_embedding], axis=0)
            self.concepts[tf.reduce_sum(key_embedding).numpy()] = concept

            self._update_nn_index()
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input shape for ConceptualMemory add: {e}")
            raise MemoryError(f"Input shape mismatch in ConceptualMemory add: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in ConceptualMemory add: {e}")
            raise MemoryCritical(f"Critical error in ConceptualMemory add operation: {e}")

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        try:
            query_embedding = tf.reduce_mean(query, axis=0)
            query_embedding = tf.reshape(query_embedding, [1, self.embedding_size])

            if self.nn_index is None:
                return tf.zeros([0, self.embedding_size], dtype=tf.float32)

            distances, indices = self.nn_index.kneighbors(query_embedding.numpy(), 
                                                          n_neighbors=min(top_k, self.concept_embeddings.shape[0]))

            return tf.gather(self.concept_embeddings, indices[0])
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for ConceptualMemory query: {e}")
            raise MemoryError(f"Query operation failed in ConceptualMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in ConceptualMemory query: {e}")
            raise MemoryCritical(f"Critical error in ConceptualMemory query operation: {e}")

    def _update_nn_index(self) -> None:
        try:
            if self.concept_embeddings.shape[0] > 0:
                n_neighbors = min(5, self.concept_embeddings.shape[0])
                self.nn_index = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm='ball_tree').fit(self.concept_embeddings.numpy())
        except ValueError as e:
            logger.warning(f"Invalid input for NearestNeighbors in ConceptualMemory: {e}")
            raise MemoryWarning(f"Failed to update NN index in ConceptualMemory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating NN index in ConceptualMemory: {e}")
            raise MemoryError(f"Critical error updating NN index in ConceptualMemory: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'concepts': self.concepts,
                'concept_embeddings': self.concept_embeddings.numpy()
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from ConceptualMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from ConceptualMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.concepts = shared_data['concepts']
            self.concept_embeddings = tf.Variable(shared_data['concept_embeddings'])
            self._update_nn_index()
        except KeyError as e:
            logger.error(f"Missing key in shared data for ConceptualMemory update: {e}")
            raise MemoryError(f"Invalid shared data format for ConceptualMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating ConceptualMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in ConceptualMemory update from shared data: {e}")

    def save(self, filepath: str) -> None:
        try:
            state = {
                'concepts': self.concepts,
                'concept_embeddings': self.concept_embeddings.numpy()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except IOError as e:
            logger.error(f"IO error while saving ConceptualMemory: {e}")
            raise MemoryError(f"Failed to save ConceptualMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving ConceptualMemory: {e}")
            raise MemoryCritical(f"Critical error saving ConceptualMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.concepts = state['concepts']
            self.concept_embeddings = tf.Variable(state['concept_embeddings'])
            self._update_nn_index()
        except IOError as e:
            logger.error(f"IO error while loading ConceptualMemory: {e}")
            raise MemoryError(f"Failed to load ConceptualMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading ConceptualMemory: {e}")
            raise MemoryError(f"Corrupted ConceptualMemory data in {filepath}: {e}")
        except KeyError as e:
            logger.error(f"Missing key in loaded ConceptualMemory data: {e}")
            raise MemoryError(f"Incomplete ConceptualMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading ConceptualMemory: {e}")
            raise MemoryCritical(f"Critical error loading ConceptualMemory: {e}")