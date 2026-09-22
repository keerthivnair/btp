from typing import List, Tuple
import numpy as np
from core.types import NDArrays

class AggregationStrategy:
    """Minimal interface for aggregation strategies."""
    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        """
        Aggregates a list of client updates.
        Each update is a tuple of (num_examples, parameters).
        """
        raise NotImplementedError

class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging strategy implementation."""
    
    def aggregate(self, updates: List[Tuple[int, NDArrays]]) -> NDArrays:
        if not updates:
            raise ValueError("No updates provided for aggregation.")
            
        total_examples = sum(num_examples for num_examples, _ in updates)
        if total_examples == 0:
            raise ValueError("Total number of examples is 0.")
            
        # Create a list of zero arrays with the same shape as the parameters
        # We assume all updates have the same shape
        first_params = updates[0][1]
        aggregated_params = [np.zeros_like(p) for p in first_params]
        
        for num_examples, parameters in updates:
            weight = num_examples / total_examples
            for i, p in enumerate(parameters):
                aggregated_params[i] += p * weight
                
        return aggregated_params
