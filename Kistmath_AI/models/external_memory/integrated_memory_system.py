import tensorflow as tf
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from .base import MemoryComponent, MemoryError, MemoryCritical
from .external_memory import ExternalMemory
from .formulative_memory import FormulativeMemory
from .conceptual_memory import ConceptualMemory
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .inference_memory import InferenceMemory
import logging

logger = logging.getLogger(__name__)

class IntegratedMemorySystem:
    def __init__(self, config: Dict[str, Any]):
        try:
            self.external_memory = ExternalMemory(**config.get('external_memory', {}))
            self.formulative_memory = FormulativeMemory(**config.get('formulative_memory', {}))
            self.conceptual_memory = ConceptualMemory(**config.get('conceptual_memory', {}))
            self.short_term_memory = ShortTermMemory(**config.get('short_term_memory', {}))
            self.long_term_memory = LongTermMemory(**config.get('long_term_memory', {}))
            self.inference_memory = InferenceMemory(**config.get('inference_memory', {}))
        except MemoryError as e:
            logger.error(f"Error initializing memory components: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error initializing IntegratedMemorySystem: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem initialization: {e}")

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Share data between components before processing
            shared_data = self.share_data()
            self.update_from_shared_data(shared_data)

            results = {}

            if 'external_query' in input_data:
                query_embedding, query_terms = input_data['external_query']
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

            # Combine and process results
            combined_results = self.combine_results(results)

            return combined_results
        except MemoryError as e:
            logger.error(f"Error processing input in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem process_input: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem process_input: {e}")

    def share_data(self) -> Dict[str, Dict[str, Any]]:
        try:
            shared_data = {}
            for component_name, component in self.__dict__.items():
                if isinstance(component, MemoryComponent):
                    shared_data[component_name] = component.get_shareable_data()
            return shared_data
        except MemoryError as e:
            logger.error(f"Error sharing data in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem share_data: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem share_data: {e}")

    def update_from_shared_data(self, shared_data: Dict[str, Dict[str, Any]]) -> None:
        try:
            for component_name, component_data in shared_data.items():
                if hasattr(self, component_name):
                    getattr(self, component_name).update_from_shared_data(component_data)
        except MemoryError as e:
            logger.error(f"Error updating from shared data in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem update_from_shared_data: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem update_from_shared_data: {e}")

    def combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        try:
            combined_results = {}
            
            # Combine embeddings from different memory components
            all_embeddings = []
            for memory_type, result in results.items():
                if isinstance(result, tuple) and len(result) == 2:
                    embeddings, _ = result
                    all_embeddings.append(embeddings)
                elif isinstance(result, tf.Tensor):
                    all_embeddings.append(result)
            
            if all_embeddings:
                combined_embeddings = tf.concat(all_embeddings, axis=0)
                combined_results['combined_embeddings'] = combined_embeddings

            # Combine other relevant information
            combined_results['memory_outputs'] = results

            return combined_results
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Invalid argument error in combine_results: {e}")
            raise MemoryError(f"Failed to combine results in IntegratedMemorySystem: {e}")
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem combine_results: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem combine_results: {e}")

    def update_memories(self, update_data: Dict[str, Any]) -> None:
        try:
            for memory_type, data in update_data.items():
                if hasattr(self, memory_type):
                    getattr(self, memory_type).add(**data)

            # After updating individual memories, share the updated data
            shared_data = self.share_data()
            self.update_from_shared_data(shared_data)
        except MemoryError as e:
            logger.error(f"Error updating memories in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem update_memories: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem update_memories: {e}")

    def save_state(self, directory: str) -> None:
        try:
            for component_name, component in self.__dict__.items():
                if isinstance(component, MemoryComponent):
                    component.save(f"{directory}/{component_name}.pkl")
        except MemoryError as e:
            logger.error(f"Error saving state in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem save_state: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem save_state: {e}")

    def load_state(self, directory: str) -> None:
        try:
            for component_name, component in self.__dict__.items():
                if isinstance(component, MemoryComponent):
                    component.load(f"{directory}/{component_name}.pkl")
        except MemoryError as e:
            logger.error(f"Error loading state in IntegratedMemorySystem: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in IntegratedMemorySystem load_state: {e}")
            raise MemoryCritical(f"Critical error in IntegratedMemorySystem load_state: {e}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        try:
            stats = {}
            for component_name, component in self.__dict__.items():
                if isinstance(component, MemoryComponent):
                    if isinstance(component, ExternalMemory):
                        stats[component_name] = {
                            'size': tf.shape(component.memory_embeddings)[0].numpy(),
                            'capacity': component.memory_size,
                            'usage': tf.reduce_mean(component.usage).numpy()
                        }
                    elif isinstance(component, FormulativeMemory):
                        stats[component_name] = {
                            'size': tf.shape(component.formula_embeddings)[0].numpy(),
                            'capacity': component.max_formulas,
                            'unique_terms': len(component.inverted_index)
                        }
                    elif isinstance(component, ConceptualMemory):
                        stats[component_name] = {
                            'size': tf.shape(component.concept_embeddings)[0].numpy(),
                            'unique_concepts': len(component.concepts)
                        }
                    elif isinstance(component, ShortTermMemory):
                        stats[component_name] = {
                            'size': len(component.memory),
                            'capacity': component.capacity
                        }
                    elif isinstance(component, LongTermMemory):
                        stats[component_name] = {
                            'size': len(component.memory),
                            'capacity': component.capacity,
                            'avg_importance': np.mean(component.importance_scores) if component.importance_scores else 0
                        }
                    elif isinstance(component, InferenceMemory):
                        stats[component_name] = {
                            'size': len(component.inferences),
                            'capacity': component.capacity,
                            'avg_confidence': np.mean(component.confidence_scores) if component.confidence_scores else 0
                        }
            return stats
        except Exception as e:
            logger.error(f"Error getting memory statistics in IntegratedMemorySystem: {e}")
            raise MemoryError(f"Failed to get memory statistics in IntegratedMemorySystem: {e}")