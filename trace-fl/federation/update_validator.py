from typing import List, Optional
import numpy as np
from core.types import NDArrays

class UpdateValidator:
    """Validates client updates before aggregation."""
    def __init__(self, expected_shapes: Optional[List[tuple]] = None):
        self.expected_shapes = expected_shapes
        
    def validate(self, client_id: str, num_examples: int, parameters: NDArrays) -> bool:
        """
        Returns True if the update is valid, False otherwise.
        """
        if num_examples <= 0:
            # Cannot aggregate zero examples
            return False
            
        if not parameters:
            # Empty parameters
            return False
            
        for p in parameters:
            if not isinstance(p, np.ndarray):
                return False
                
            # Check for NaNs and Infs
            if not np.isfinite(p).all():
                return False
                
        if self.expected_shapes is not None:
            if len(parameters) != len(self.expected_shapes):
                return False
            for p, expected_shape in zip(parameters, self.expected_shapes):
                if p.shape != expected_shape:
                    return False
                    
        return True
