import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryCritical, MemoryWarning
from .btree import BTree
import logging
import pickle

logger = logging.getLogger(__name__)

class ExternalMemory(MemoryComponent):
    def __init__(self, memory_size: int = 100, key_size: int = 64, value_size: int = 128):
        if memory_size <= 0 or key_size <= 0 or value_size <= 0:
            raise MemoryError("Invalid memory parameters: sizes must be positive integers")
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.btree = BTree(t=5)
        self.usage = tf.Variable(tf.zeros([memory_size], dtype=tf.float32))
        self.memory_embeddings = tf.Variable(tf.zeros([memory_size, value_size], dtype=tf.float32))

    def add(self, data: Tuple[tf.Tensor, tf.Tensor], metadata: Optional[Dict] = None) -> None:
        try:
            key, value = data
            key = tf.ensure_shape(key, [None, self.key_size])
            value = tf.ensure_shape(value, [None, self.value_size])

            for i in range(tf.shape(key)[0]):
                key_sum = tf.reduce_sum(key[i])
                self.btree.insert((key_sum.numpy(), value[i].numpy()))

            current_size = tf.shape(self.memory_embeddings)[0]
            new_size = tf.minimum(self.memory_size, current_size + tf.shape(value)[0])
            self.memory_embeddings = tf.Variable(tf.concat([self.memory_embeddings[:new_size - tf.shape(value)[0]], value[:new_size - current_size]], axis=0))

            self._update_usage()
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input shape for ExternalMemory add: {e}")
            raise MemoryError(f"Input shape mismatch in ExternalMemory add: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in ExternalMemory add: {e}")
            raise MemoryCritical(f"Critical error in ExternalMemory add operation: {e}")

    def query(self, query_embedding: tf.Tensor, query_terms: Optional[tf.Tensor] = None, top_k: int = 5) -> Tuple[tf.Tensor, tf.Tensor]:
        try:
            if tf.equal(tf.shape(self.memory_embeddings)[0], 0):
                return tf.zeros([0, self.value_size], dtype=tf.float32), tf.constant([], dtype=tf.int32)

            similarities = tf.matmul(tf.reshape(query_embedding, [1, -1]), self.memory_embeddings, transpose_b=True)
            _, top_indices = tf.nn.top_k(similarities[0], k=min(top_k, tf.shape(self.memory_embeddings)[0]))

            return tf.gather(self.memory_embeddings, top_indices), top_indices
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid input for ExternalMemory query: {e}")
            raise MemoryError(f"Query operation failed in ExternalMemory: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in ExternalMemory query: {e}")
            raise MemoryCritical(f"Critical error in ExternalMemory query operation: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'memory_embeddings': self.memory_embeddings.numpy(),
                'usage': self.usage.numpy()
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from ExternalMemory: {e}")
            raise MemoryError(f"Failed to get shareable data from ExternalMemory: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            if 'memory_embeddings' in shared_data:
                self.memory_embeddings = tf.Variable(shared_data['memory_embeddings'])
            if 'usage' in shared_data:
                self.usage = tf.Variable(shared_data['usage'])
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid shared data format for ExternalMemory update: {e}")
            raise MemoryError(f"Failed to update ExternalMemory from shared data: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error updating ExternalMemory from shared data: {e}")
            raise MemoryCritical(f"Critical error in ExternalMemory update from shared data: {e}")

    def _update_usage(self, retrieved_values: Optional[tf.Tensor] = None) -> None:
        try:
            if retrieved_values is not None:
                usage_update = tf.reduce_sum(tf.cast(tf.not_equal(retrieved_values, 0), tf.float32), axis=-1)
                usage_update = tf.reduce_mean(usage_update)
                usage_update = tf.repeat(usage_update, self.memory_size)
                self.usage.assign_add(usage_update)

            self.usage.assign(self.usage * 0.99)
            total_usage = tf.reduce_sum(self.usage)
            if total_usage > 0:
                self.usage.assign(self.usage / total_usage)
        except tf.errors.InvalidArgumentError as e:
            logger.warning(f"Invalid usage update in ExternalMemory: {e}")
            raise MemoryWarning(f"Usage update warning in ExternalMemory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in ExternalMemory usage update: {e}")
            raise MemoryError(f"Failed to update usage in ExternalMemory: {e}")

    def save(self, filepath: str) -> None:
        try:
            state = {
                'btree': self.btree,
                'usage': self.usage.numpy(),
                'memory_embeddings': self.memory_embeddings.numpy()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except IOError as e:
            logger.error(f"IO error while saving ExternalMemory: {e}")
            raise MemoryError(f"Failed to save ExternalMemory to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving ExternalMemory: {e}")
            raise MemoryCritical(f"Critical error saving ExternalMemory: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.btree = state['btree']
            self.usage = tf.Variable(state['usage'])
            self.memory_embeddings = tf.Variable(state['memory_embeddings'])
        except IOError as e:
            logger.error(f"IO error while loading ExternalMemory: {e}")
            raise MemoryError(f"Failed to load ExternalMemory from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading ExternalMemory: {e}")
            raise MemoryError(f"Corrupted ExternalMemory data in {filepath}: {e}")
        except KeyError as e:
            logger.error(f"Missing key in loaded ExternalMemory data: {e}")
            raise MemoryError(f"Incomplete ExternalMemory data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading ExternalMemory: {e}")
            raise MemoryCritical(f"Critical error loading ExternalMemory: {e}")