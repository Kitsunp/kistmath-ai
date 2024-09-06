import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryCritical
import logging
import pickle
logger = logging.getLogger(__name__)

class LongTermMemory(MemoryComponent):
    def __init__(self, capacity: int = 10000):
        if capacity <= 0:
            raise MemoryError("Invalid LongTermMemory parameter: capacity must be a positive integer")
        self.capacity = capacity
        self.memory = []
        self.importance_scores = []
        self.time_added = []

    def add(self, data: tf.Tensor, metadata: Optional[Dict] = None) -> None:
        try:
            importance = metadata.get('importance', 1.0) if metadata else 1.0
            if len(self.memory) >= self.capacity:
                self._remove_least_important()
            self.memory.append(data)
            self.importance_scores.append(importance)
            self.time_added.append(tf.timestamp())
            self._sort_by_importance()
        except Exception as e:
            logger.error(f"Unexpected error in LongTermMemory add: {e}")
            raise MemoryError(f"Failed to add data to LongTermMemory: {e}")

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        try:
            similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) * imp for mem, imp in zip(self.memory, self.importance_scores)])
            _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
            return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for LongTermMemory query: {e}")
            raise MemoryError(f"Query operation failed in LongTermMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in LongTermMemory query: {e}")
            raise MemoryCritical(f"Critical error in LongTermMemory query operation: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'memory': self.memory,
                'importance_scores': self.importance_scores,
                'time_added': self.time_added
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from LongTermMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from LongTermMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.memory = shared_data['memory']
            self.importance_scores = shared_data['importance_scores']
            self.time_added = shared_data['time_added']
            self._sort_by_importance()
        except KeyError as e:
            logger.error(f"Missing key in shared data for LongTermMemory update: {e}")
            raise MemoryError(f"Invalid shared data format for LongTermMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating LongTermMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in LongTermMemory update from shared data: {e}")

    def _remove_least_important(self) -> None:
        try:
            min_importance_index = np.argmin(self.importance_scores)
            self.memory.pop(min_importance_index)
            self.importance_scores.pop(min_importance_index)
            self.time_added.pop(min_importance_index)
        except Exception as e:
            logger.error(f"Error removing least important item from LongTermMemory: {e}")
            raise MemoryError(f"Failed to remove least important item from LongTermMemory: {e}")

    def _sort_by_importance(self) -> None:
        try:
            sorted_indices = np.argsort(self.importance_scores)[::-1]
            self.memory = [self.memory[i] for i in sorted_indices]
            self.importance_scores = [self.importance_scores[i] for i in sorted_indices]
            self.time_added = [self.time_added[i] for i in sorted_indices]
        except Exception as e:
            logger.error(f"Error sorting LongTermMemory by importance: {e}")
            raise MemoryError(f"Failed to sort LongTermMemory by importance: {e}")

    def save(self, filepath: str) -> None:
        try:
            state = {
                'memory': self.memory,
                'importance_scores': self.importance_scores,
                'time_added': self.time_added
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except IOError as e:
            logger.error(f"IO error while saving LongTermMemory: {e}")
            raise MemoryError(f"Failed to save LongTermMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving LongTermMemory: {e}")
            raise MemoryCritical(f"Critical error saving LongTermMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.memory = state['memory']
            self.importance_scores = state['importance_scores']
            self.time_added = state['time_added']
            self._sort_by_importance()
        except IOError as e:
            logger.error(f"IO error while loading LongTermMemory: {e}")
            raise MemoryError(f"Failed to load LongTermMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading LongTermMemory: {e}")
            raise MemoryError(f"Corrupted LongTermMemory data in {filepath}: {e}")
        except KeyError as e:
            logger.error(f"Missing key in loaded LongTermMemory data: {e}")
            raise MemoryError(f"Incomplete LongTermMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading LongTermMemory: {e}")
            raise MemoryCritical(f"Critical error loading LongTermMemory: {e}")