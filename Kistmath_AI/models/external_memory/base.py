import tensorflow as tf
from typing import List, Tuple, Any, Dict, Optional
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryException(Exception):
    """Base exception class for memory-related errors."""
    pass

class MemoryWarning(MemoryException):
    """Exception class for non-critical memory warnings."""
    pass

class MemoryError(MemoryException):
    """Exception class for memory errors that may affect functionality."""
    pass

class MemoryCritical(MemoryException):
    """Exception class for critical memory errors that prevent normal operation."""
    pass

class MemoryComponent(ABC):
    @abstractmethod 
    def add(self, data: Any, metadata: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def query(self, query: Any, top_k: int = 5) -> Any:
        pass

    @abstractmethod
    def get_shareable_data(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_from_shared_data(self, shared_data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass