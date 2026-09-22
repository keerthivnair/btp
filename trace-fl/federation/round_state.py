from typing import List, Dict, Any, Optional

class RoundState:
    """Maintains state for the current federation round."""
    def __init__(self):
        self.round_number: int = 0
        self.selected_clients: List[str] = []
        self.successful_clients: List[str] = []
        self.failed_clients: List[str] = []
        
        # Performance metrics
        self.global_loss: float = 0.0
        self.global_accuracy: float = 0.0
        
        # Timing
        self.round_start_time: float = 0.0
        self.aggregation_time: float = 0.0
        self.round_duration: float = 0.0

    def start_round(self, round_number: int, selected_clients: List[str]):
        self.round_number = round_number
        self.selected_clients = selected_clients
        self.successful_clients = []
        self.failed_clients = []
        self.global_loss = 0.0
        self.global_accuracy = 0.0
        self.aggregation_time = 0.0
        self.round_duration = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "selected_clients": self.selected_clients,
            "successful_clients": self.successful_clients,
            "failed_clients": self.failed_clients,
            "global_loss": self.global_loss,
            "global_accuracy": self.global_accuracy,
            "aggregation_time": self.aggregation_time,
            "round_duration": self.round_duration
        }
