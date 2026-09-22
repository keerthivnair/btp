from typing import List, Dict, Any
import numpy as np
from core.types import NDArrays

class CommunicationTracker:
    """Tracks estimated payload bytes for Federated Learning simulation."""
    
    def __init__(self):
        self.client_uplink: Dict[str, int] = {}
        self.client_downlink: Dict[str, int] = {}
        self.round_uplink: Dict[int, int] = {}
        self.round_downlink: Dict[int, int] = {}
        
    def _estimate_ndarrays_bytes(self, ndarrays: NDArrays) -> int:
        """Estimates the serialized size of the given NumPy arrays."""
        return sum(arr.nbytes for arr in ndarrays)
        
    def record_downlink(self, round_num: int, client_id: str, ndarrays: NDArrays) -> int:
        """Record parameters sent from server to client."""
        estimated_bytes = self._estimate_ndarrays_bytes(ndarrays)
        
        self.client_downlink[client_id] = self.client_downlink.get(client_id, 0) + estimated_bytes
        self.round_downlink[round_num] = self.round_downlink.get(round_num, 0) + estimated_bytes
        
        return estimated_bytes
        
    def record_uplink(self, round_num: int, client_id: str, ndarrays: NDArrays) -> int:
        """Record parameters sent from client to server."""
        estimated_bytes = self._estimate_ndarrays_bytes(ndarrays)
        
        self.client_uplink[client_id] = self.client_uplink.get(client_id, 0) + estimated_bytes
        self.round_uplink[round_num] = self.round_uplink.get(round_num, 0) + estimated_bytes
        
        return estimated_bytes

    def get_client_totals(self, client_id: str) -> Dict[str, int]:
        return {
            "uplink_bytes": self.client_uplink.get(client_id, 0),
            "downlink_bytes": self.client_downlink.get(client_id, 0),
            "total_bytes": self.client_uplink.get(client_id, 0) + self.client_downlink.get(client_id, 0)
        }
        
    def get_round_totals(self, round_num: int) -> Dict[str, int]:
        return {
            "uplink_bytes": self.round_uplink.get(round_num, 0),
            "downlink_bytes": self.round_downlink.get(round_num, 0),
            "total_bytes": self.round_uplink.get(round_num, 0) + self.round_downlink.get(round_num, 0)
        }
