import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryWarning, MemoryCritical
from annoy import AnnoyIndex
import logging
import pickle
logger = logging.getLogger(__name__)

class InferenceMemory(MemoryComponent):
    def __init__(self, capacity: int = 500, embedding_size: int = 64):
        if capacity <= 0 or embedding_size <= 0:
            raise MemoryError("Invalid InferenceMemory parameters: capacity and embedding_size must be positive integers")
        self.capacity = capacity
        self.embedding_size = embedding_size
        self.inferences = []
        self.confidence_scores = []
        self.index = AnnoyIndex(embedding_size, 'angular')
        self.current_index = 0

    def add(self, data: Tuple[tf.Tensor, float], metadata: Optional[Dict] = None) -> None:
        try:
            inference, confidence = data
            if len(self.inferences) >= self.capacity:
                self._remove_least_confident()
            self.inferences.append(inference)
            self.confidence_scores.append(confidence)
            self.index.add_item(self.current_index, inference.numpy())
            self.current_index += 1

            if self.current_index % 100 == 0:
                self._rebuild_index()
        except Exception as e:
            logger.error(f"Unexpected error in InferenceMemory add: {e}")
            raise MemoryError(f"Failed to add data to InferenceMemory: {e}")

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        try:
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
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for InferenceMemory query: {e}")
            raise MemoryError(f"Query operation failed in InferenceMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in InferenceMemory query: {e}")
            raise MemoryCritical(f"Critical error in InferenceMemory query operation: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'inferences': self.inferences,
                'confidence_scores': self.confidence_scores,
                'current_index': self.current_index
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from InferenceMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from InferenceMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.inferences = shared_data['inferences']
            self.confidence_scores = shared_data['confidence_scores']
            self.current_index = shared_data['current_index']
            self._rebuild_index()
        except KeyError as e:
            logger.error(f"Missing key in shared data for InferenceMemory update: {e}")
            raise MemoryError(f"Invalid shared data format for InferenceMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating InferenceMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in InferenceMemory update from shared data: {e}")

    def _remove_least_confident(self) -> None:
        try:
            min_confidence_index = np.argmin(self.confidence_scores)
            self.inferences.pop(min_confidence_index)
            self.confidence_scores.pop(min_confidence_index)
            self.index.unbuild()
            self._rebuild_index()
        except Exception as e:
            logger.error(f"Error removing least confident item from InferenceMemory: {e}")
            raise MemoryError(f"Failed to remove least confident item from InferenceMemory: {e}")

    def _rebuild_index(self) -> None:
        try:
            self.index = AnnoyIndex(self.embedding_size, 'angular')
            for i, inference in enumerate(self.inferences):
                self.index.add_item(i, inference.numpy())
            self.index.build(10)  # 10 trees - adjust based on your needs
        except Exception as e:
            logger.error(f"Error rebuilding index in InferenceMemory: {e}")
            raise MemoryError(f"Failed to rebuild index in InferenceMemory: {e}")

    def save(self, filepath: str) -> None:
        try:
            state = {
                'inferences': self.inferences,
                'confidence_scores': self.confidence_scores,
                'current_index': self.current_index
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            self.index.save(filepath + '.ann')
        except IOError as e:
            logger.error(f"IO error while saving InferenceMemory: {e}")
            raise MemoryError(f"Failed to save InferenceMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving InferenceMemory: {e}")
            raise MemoryCritical(f"Critical error saving InferenceMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.inferences = state['inferences']
            self.confidence_scores = state['confidence_scores']
            self.current_index = state['current_index']
            self.index.load(filepath + '.ann')
        except IOError as e:
            logger.error(f"IO error while loading InferenceMemory: {e}")
            raise MemoryError(f"Failed to load InferenceMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading InferenceMemory: {e}")
            raise MemoryError(f"Corrupted InferenceMemory data in {filepath}: {e}")
        except KeyError as e:
            logger.error(f"Missing key in loaded InferenceMemory data: {e}")
            raise MemoryError(f"Incomplete InferenceMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading InferenceMemory: {e}")
            raise MemoryCritical(f"Critical error loading InferenceMemory: {e}")