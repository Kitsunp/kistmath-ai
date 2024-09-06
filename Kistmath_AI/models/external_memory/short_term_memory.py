import tensorflow as tf
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryCritical
import logging
import pickle
logger = logging.getLogger(__name__)

class ShortTermMemory(MemoryComponent):
    def __init__(self, capacity: int = 100):
        if capacity <= 0:
            raise MemoryError("Invalid ShortTermMemory parameter: capacity must be a positive integer")
        self.capacity = capacity
        self.memory = []

    def add(self, data: tf.Tensor, metadata: Optional[Dict] = None) -> None:
        try:
            if len(self.memory) >= self.capacity:
                self.memory.pop(0)
            self.memory.append(data)
        except Exception as e:
            logger.error(f"Unexpected error in ShortTermMemory add: {e}")
            raise MemoryError(f"Failed to add data to ShortTermMemory: {e}")

    def query(self, query: tf.Tensor, top_k: int = 5) -> tf.Tensor:
        try:
            similarities = tf.stack([tf.reduce_sum(query * tf.reshape(mem, [1, -1])) for mem in self.memory])
            _, top_indices = tf.nn.top_k(similarities, k=min(top_k, len(self.memory)))
            return tf.stack([tf.reshape(self.memory[i], [-1]) for i in top_indices.numpy()])
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for ShortTermMemory query: {e}")
            raise MemoryError(f"Query operation failed in ShortTermMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in ShortTermMemory query: {e}")
            raise MemoryCritical(f"Critical error in ShortTermMemory query operation: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {'memory': self.memory}
        except Exception as e:
            logger.error(f"Error getting shareable data from ShortTermMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from ShortTermMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.memory = shared_data['memory'][-self.capacity:]
        except KeyError as e:
            logger.error(f"Missing key in shared data for ShortTermMemory update: {e}")
            raise MemoryError(f"Invalid shared data format for ShortTermMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating ShortTermMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in ShortTermMemory update from shared data: {e}")

    def save(self, filepath: str) -> None:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.memory, f)
        except IOError as e:
            logger.error(f"IO error while saving ShortTermMemory: {e}")
            raise MemoryError(f"Failed to save ShortTermMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving ShortTermMemory: {e}")
            raise MemoryCritical(f"Critical error saving ShortTermMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
            self.memory = self.memory[-self.capacity:]
        except IOError as e:
            logger.error(f"IO error while loading ShortTermMemory: {e}")
            raise MemoryError(f"Failed to load ShortTermMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading ShortTermMemory: {e}")
            raise MemoryError(f"Corrupted ShortTermMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading ShortTermMemory: {e}")
            raise MemoryCritical(f"Critical error loading ShortTermMemory: {e}")