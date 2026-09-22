import random
from typing import List

class ClientSelector:
    """Interface for client selection strategies."""
    def select(self, available_clients: List[str], num_clients: int) -> List[str]:
        raise NotImplementedError

class RandomClientSelector(ClientSelector):
    """Selects clients randomly."""
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
    def select(self, available_clients: List[str], num_clients: int) -> List[str]:
        if num_clients > len(available_clients):
            raise ValueError(f"Cannot select {num_clients} clients from {len(available_clients)} available.")
        return self.rng.sample(available_clients, num_clients)
