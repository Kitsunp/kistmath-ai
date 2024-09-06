import tensorflow as tf
from .base import MemoryComponent, MemoryError, MemoryCritical
import logging
from typing import List, Tuple, Any, Dict, Optional
import pickle
from collections import OrderedDict
import time
import sys
import psutil

logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: float) -> Optional[Any]:
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: float, value: Any) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

class BTreeNode:
    __slots__ = ['leaf', 'keys', 'children']
    def __init__(self, leaf: bool = False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree(MemoryComponent):
    def __init__(self, t: int):
        if t < 2:
            raise MemoryError("BTree degree must be at least 2")
        self.root = BTreeNode(True)
        self.t = t
        self.cache = LRUCache(100000)  # Initial cache size
        self.cache_ttl = 3600  # Time-to-live for cache entries (1 hour)
        self.modified_ranges = []
        self.last_cache_resize = time.time()
        self.cache_resize_interval = 300  # Resize cache every 5 minutes

    def _update_cache(self, key: float, value: Any) -> None:
        expiration = time.time() + self.cache_ttl
        self.cache.put(key, (value, expiration))
        self._resize_cache_if_needed()

    def _resize_cache_if_needed(self) -> None:
        current_time = time.time()
        if current_time - self.last_cache_resize > self.cache_resize_interval:
            available_memory = psutil.virtual_memory().available
            max_cache_size = int(available_memory * 0.1)  # Use up to 10% of available memory
            current_cache_size = sum(sys.getsizeof(item) for item in self.cache.cache.items())
            
            if current_cache_size > max_cache_size:
                new_capacity = max(1000, self.cache.capacity // 2)  # Ensure a minimum capacity
            elif current_cache_size < max_cache_size // 2:
                new_capacity = min(1000000, self.cache.capacity * 2)  # Set a maximum capacity
            else:
                return  # No need to resize
            
            new_cache = LRUCache(new_capacity)
            for key, value in self.cache.cache.items():
                new_cache.put(key, value)
            self.cache = new_cache
            self.last_cache_resize = current_time

    def insert(self, k: Tuple[float, Any]) -> None:
        if not isinstance(k, tuple) or len(k) != 2:
            raise MemoryError("Invalid key format for BTree insertion")
        key, value = k
        key = float(key)  # Ensure the key is a float
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.children.insert(0, root)
            self._split_child(temp, 0)
            self._insert_non_full(temp, (key, value))
        else:
            self._insert_non_full(root, (key, value))
        
        self._update_cache(key, value)
        self._invalidate_cache_range(key - 0.1, key + 0.1)  # Invalidate nearby keys

    def _invalidate_cache_range(self, start: float, end: float) -> None:
        self.modified_ranges.append((start, end))
        if len(self.modified_ranges) > 100:  # Limit the number of ranges to prevent excessive memory usage
            self._merge_modified_ranges()

    def _merge_modified_ranges(self) -> None:
        if not self.modified_ranges:
            return
        self.modified_ranges.sort(key=lambda x: x[0])
        merged = [self.modified_ranges[0]]
        for start, end in self.modified_ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.modified_ranges = merged

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
        key, value = k
        if x.leaf:
            index = self._find_insertion_index(x, key)
            x.keys.insert(index, (key, value))
        else:
            index = self._find_insertion_index(x, key)
            if index < len(x.keys) and self._is_close(key, x.keys[index][0]):
                x.keys[index] = (key, value)  # Update existing key
                return
            if len(x.children[index].keys) == (2 * self.t) - 1:
                self._split_child(x, index)
                if key > x.keys[index][0]:
                    index += 1
            self._insert_non_full(x.children[index], (key, value))

    def _find_insertion_index(self, node: BTreeNode, key: float) -> int:
        left, right = 0, len(node.keys)
        while left < right:
            mid = (left + right) // 2
            if self._is_close(key, node.keys[mid][0]):
                return mid
            elif key < node.keys[mid][0]:
                right = mid
            else:
                left = mid + 1
        return left

    def search(self, k: tf.Tensor, x: BTreeNode = None) -> Optional[tf.Tensor]:
        k_value = float(k.numpy()) if isinstance(k, tf.Tensor) else float(k)
        
        # Check cache first
        cached_value = self.cache.get(k_value)
        if cached_value is not None:
            value, expiration = cached_value
            if time.time() < expiration:
                return tf.constant(value)
            else:
                self.cache.cache.pop(k_value)

        # Check if the key is in a modified range
        for start, end in self.modified_ranges:
            if start <= k_value <= end:
                break
        else:
            if x is None:
                x = self.root
            
            # Binary search
            left, right = 0, len(x.keys) - 1
            while left <= right:
                mid = (left + right) // 2
                if self._is_close(k_value, x.keys[mid][0]):
                    if x.keys[mid][1] is not None:
                        self._update_cache(k_value, x.keys[mid][1])
                        return tf.constant(x.keys[mid][1])
                    else:
                        return None
                elif k_value < x.keys[mid][0]:
                    right = mid - 1
                else:
                    left = mid + 1
            
            if x.leaf:
                return None
            else:
                return self.search(k, x.children[left])

        return None

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def delete(self, k: tf.Tensor) -> None:
        k_value = float(k.numpy()) if isinstance(k, tf.Tensor) else float(k)
        self._delete_key(self.root, k_value)
        
        # Si la raíz tiene 0 claves y al menos un hijo, actualiza la raíz
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]
        
        # Eliminar de la caché
        self.cache.cache.pop(k_value, None)

    def _delete_key(self, x: BTreeNode, k: float) -> None:
        index = self._find_key_index(x, k)
        
        # Caso 1: La clave está en este nodo
        if index < len(x.keys) and self._is_close(k, x.keys[index][0]):
            if x.leaf:
                self._delete_from_leaf(x, index)
            else:
                self._delete_from_non_leaf(x, index)
        # Caso 2: La clave no está en este nodo
        else:
            if x.leaf:
                return  # La clave no existe en el árbol
            
            # Determina si la clave está en el último hijo
            flag = index == len(x.keys)
            
            # Si el hijo tiene menos de t claves, lo rellenamos
            if len(x.children[index].keys) < self.t:
                self._fill(x, index)
            
            # Si el último hijo se ha fusionado, debemos ir al hijo anterior
            if flag and index > len(x.keys):
                self._delete_key(x.children[index - 1], k)
            else:
                self._delete_key(x.children[index], k)

    def _delete_from_leaf(self, x: BTreeNode, index: int) -> None:
        x.keys.pop(index)

    def _delete_from_non_leaf(self, x: BTreeNode, index: int) -> None:
        k = x.keys[index]
        
        # Caso 3a: Si el hijo que precede a k tiene al menos t claves
        if len(x.children[index].keys) >= self.t:
            pred = self._get_pred(x, index)
            x.keys[index] = pred
            self._delete_key(x.children[index], pred[0])
        
        # Caso 3b: Si el hijo que sucede a k tiene al menos t claves
        elif len(x.children[index + 1].keys) >= self.t:
            succ = self._get_succ(x, index)
            x.keys[index] = succ
            self._delete_key(x.children[index + 1], succ[0])
        
        # Caso 3c: Si ambos hijos tienen menos de t claves
        else:
            self._merge(x, index)
            self._delete_key(x.children[index], k[0])

    def _get_pred(self, x: BTreeNode, index: int) -> Tuple[float, Any]:
        current = x.children[index]
        while not current.leaf:
            current = current.children[-1]
        return current.keys[-1]

    def _get_succ(self, x: BTreeNode, index: int) -> Tuple[float, Any]:
        current = x.children[index + 1]
        while not current.leaf:
            current = current.children[0]
        return current.keys[0]

    def _fill(self, x: BTreeNode, index: int) -> None:
        # Caso 1: Si el hermano izquierdo tiene más de t-1 claves
        if index != 0 and len(x.children[index - 1].keys) >= self.t:
            self._borrow_from_prev(x, index)
        
        # Caso 2: Si el hermano derecho tiene más de t-1 claves
        elif index != len(x.children) - 1 and len(x.children[index + 1].keys) >= self.t:
            self._borrow_from_next(x, index)
        
        # Caso 3: Fusionar con un hermano
        else:
            if index != len(x.children) - 1:
                self._merge(x, index)
            else:
                self._merge(x, index - 1)

    def _borrow_from_prev(self, x: BTreeNode, index: int) -> None:
        child = x.children[index]
        sibling = x.children[index - 1]
        
        # Mover una clave del nodo x al final de child
        child.keys.insert(0, x.keys[index - 1])
        
        # Mover la última clave del hermano a x
        x.keys[index - 1] = sibling.keys.pop()
        
        # Si el hermano no es una hoja, mover también el último hijo
        if not sibling.leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, x: BTreeNode, index: int) -> None:
        child = x.children[index]
        sibling = x.children[index + 1]
        
        # Mover una clave del nodo x al principio de child
        child.keys.append(x.keys[index])
        
        # Mover la primera clave del hermano a x
        x.keys[index] = sibling.keys.pop(0)
        
        # Si el hermano no es una hoja, mover también el primer hijo
        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, x: BTreeNode, index: int) -> None:
        child = x.children[index]
        sibling = x.children[index + 1]
        
        # Mover una clave de x al final de child
        child.keys.append(x.keys.pop(index))
        
        # Mover todas las claves de sibling a child
        child.keys.extend(sibling.keys)
        
        # Si no son hojas, mover también los hijos
        if not child.leaf:
            child.children.extend(sibling.children)
        
        # Eliminar sibling de la lista de hijos de x
        x.children.pop(index + 1)

    def _find_key_index(self, node: BTreeNode, key: float) -> int:
        left, right = 0, len(node.keys)
        while left < right:
            mid = (left + right) // 2
            if self._is_close(key, node.keys[mid][0]):
                return mid
            elif key < node.keys[mid][0]:
                right = mid
            else:
                left = mid + 1
        return left

    def add(self, data: Tuple[float, Any], metadata: Optional[Dict] = None) -> None:
        try:
            self.insert(data)
        except MemoryError as e:
            logger.error(f"Failed to add data to BTree: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error while adding data to BTree: {e}")
            raise MemoryCritical(f"Critical error in BTree add operation: {e}")

    def query(self, query: float, top_k: int = 1) -> List[Any]:
        try:
            result = self.search(tf.constant(query))
            return [result.numpy()] if result is not None else []
        except Exception as e:
            logger.error(f"Error querying BTree: {e}")
            raise MemoryError(f"Failed to query BTree: {e}")

    def get_shareable_data(self) -> Dict[str, Any]:
        try:
            return {
                'root': self._serialize_node(self.root),
                't': self.t,
                'cache': list(self.cache.cache.items()),
                'modified_ranges': self.modified_ranges
            }
        except Exception as e:
            logger.error(f"Error getting shareable data from BTree: {e}")
            raise MemoryError(f"Failed to get shareable data from BTree: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        try:
            self.t = shared_data['t']
            self.root = self._deserialize_node(shared_data['root'])
            self.cache = LRUCache(100000)
            for key, value in shared_data['cache']:
                self.cache.put(key, value)
            self.modified_ranges = shared_data['modified_ranges']
        except KeyError as e:
            logger.error(f"Missing key in shared data for BTree update: {e}")
            raise MemoryError(f"Invalid shared data format for BTree: {e}")
        except Exception as e:
            logger.critical(f"Failed to update BTree from shared data: {e}")
            raise MemoryCritical(f"Critical error updating BTree from shared data: {e}")

    def save(self, filepath: str) -> None:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except IOError as e:
            logger.error(f"IO error while saving BTree: {e}")
            raise MemoryError(f"Failed to save BTree to {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while saving BTree: {e}")
            raise MemoryCritical(f"Critical error saving BTree: {e}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                loaded_tree = pickle.load(f)
                self.__dict__.update(loaded_tree.__dict__)
        except IOError as e:
            logger.error(f"IO error while loading BTree: {e}")
            raise MemoryError(f"Failed to load BTree from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error while loading BTree: {e}")
            raise MemoryError(f"Corrupted BTree data in {filepath}: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error while loading BTree: {e}")
            raise MemoryCritical(f"Critical error loading BTree: {e}")

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