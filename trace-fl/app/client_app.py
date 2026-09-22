import yaml
import torch
import time
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from core.model import create_model, set_parameters, get_parameters
from core.parameters import flower_parameters_to_ndarrays, ndarrays_to_flower_parameters
from data.dataset import FederatedDatasetProvider
from training.trainer import train, test
from telemetry.communication import CommunicationTracker
from telemetry.logging import log_client  

class TRACEClient(NumPyClient):
    def __init__(self, client_id: str, partition_id: int, config: dict):
        self.client_id = client_id
        self.partition_id = partition_id
        self.config = config
        
        self.model = create_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load data partition
        self.dataset_provider = FederatedDatasetProvider(
            num_clients=config.get("num_clients", 20),
            batch_size=config.get("local_training", {}).get("batch_size", 32),
            seed=config.get("seed", 42)
        )
        self.partition = self.dataset_provider.get_partition(self.partition_id)
        self.tracker = CommunicationTracker()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        start_time = time.time()
        round_num = config.get("round", 1)
        
        # Track downlink
        downlink_bytes = self.tracker.record_downlink(round_num, self.client_id, parameters)
        
        # Update model
        set_parameters(self.model, parameters)
        
        # Train
        epochs = self.config.get("local_training", {}).get("epochs", 1)
        lr = self.config.get("local_training", {}).get("learning_rate", 0.01)
        
        metrics = train(
            model=self.model,
            train_loader=self.partition.train_loader,
            epochs=epochs,
            learning_rate=lr,
            device=self.device
        )
        
        # Get updated parameters
        updated_parameters = get_parameters(self.model)
        
        # Track uplink
        uplink_bytes = self.tracker.record_uplink(round_num, self.client_id, updated_parameters)
        
        training_time = time.time() - start_time
        
        # Log client execution
        log_client(
            client_id=self.client_id,
            partition_id=self.partition_id,
            num_examples=self.partition.num_examples,
            local_epochs=epochs,
            training_time=training_time,
            local_loss=metrics["loss"],
            local_accuracy=metrics["accuracy"],
            uplink_bytes=uplink_bytes,
            downlink_bytes=downlink_bytes
        )
        
        return updated_parameters, self.partition.num_examples, {
            "loss": float(metrics["loss"]), 
            "accuracy": float(metrics["accuracy"]),
            "client_id": self.client_id
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(
            model=self.model,
            test_loader=self.partition.test_loader,
            device=self.device
        )
        
        from telemetry.logging import log_client_evaluate
        log_client_evaluate(
            client_id=self.client_id,
            partition_id=self.partition_id,
            num_examples=self.partition.num_examples,
            loss=float(loss),
            accuracy=float(accuracy)
        )
        
        return float(loss), self.partition.num_examples, {"accuracy": float(accuracy)}

def client_fn(context: Context):
    """Flower ClientApp entry point."""
    # Load default configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Merge with context config if provided
    # ...
    
    # In Simulation Runtime, node_id is a random 64-bit integer.
    # We use a file-based counter to deterministically assign unique partition IDs.
    if "partition-id" in context.node_config:
        partition_id = int(context.node_config["partition-id"])
    else:
        import os, fcntl
        counter_file = "/tmp/flwr_client_counter.txt"
        with open(counter_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            val = f.read().strip()
            count = int(val) if val else 0
            f.seek(0)
            f.truncate()
            f.write(str(count + 1))
            fcntl.flock(f, fcntl.LOCK_UN)
        partition_id = count % config.get("num_clients", 20)
        
    client_id = f"client_{partition_id}"
    
    return TRACEClient(client_id, partition_id, config).to_client()

app = ClientApp(client_fn=client_fn)
