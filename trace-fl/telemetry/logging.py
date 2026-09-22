import logging
import json
from typing import Dict, Any, List

def setup_logger(name: str = "TRACE-FL") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()

def log_round(
    round_num: int, 
    selected_clients: List[str], 
    successful_clients: List[str],
    failed_clients: List[str],
    global_loss: float,
    global_accuracy: float,
    uplink_bytes: int,
    downlink_bytes: int,
    aggregation_time: float,
    round_time: float
):
    """Log structured round telemetry."""
    log_data = {
        "event": "round_complete",
        "round": round_num,
        "selected_clients": len(selected_clients),
        "successful_clients": len(successful_clients),
        "failed_clients": len(failed_clients),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        "total_payload_bytes": uplink_bytes + downlink_bytes,
        "aggregation_time": aggregation_time,
        "round_time": round_time
    }
    logger.info(json.dumps(log_data))

def log_client(
    client_id: str,
    partition_id: int,
    num_examples: int,
    local_epochs: int,
    training_time: float,
    local_loss: float,
    local_accuracy: float,
    uplink_bytes: int,
    downlink_bytes: int
):
    """Log structured client telemetry."""
    log_data = {
        "event": "client_complete",
        "client_id": client_id,
        "partition_id": partition_id,
        "num_examples": num_examples,
        "local_epochs": local_epochs,
        "training_time": training_time,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes
    }
    logger.info(json.dumps(log_data))

def log_client_evaluate(
    client_id: str,
    partition_id: int,
    num_examples: int,
    loss: float,
    accuracy: float
):
    """Log structured client evaluation telemetry."""
    log_data = {
        "event": "client_evaluate",
        "client_id": client_id,
        "partition_id": partition_id,
        "num_examples": num_examples,
        "global_model_loss_on_local_data": loss,
        "global_model_accuracy_on_local_data": accuracy
    }
    logger.info(json.dumps(log_data))
