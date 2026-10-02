import yaml
import torch
import time

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from core.model import create_model, set_parameters, get_parameters
from data.dataset import FederatedDatasetProvider
from training.trainer import train, evaluate
from telemetry.communication import CommunicationTracker
from telemetry.logging import log_client, log_client_evaluate


class TRACEClient(NumPyClient):
    def __init__(self, client_id: str, partition_id: int, config: dict):
        self.client_id = client_id
        self.partition_id = partition_id
        self.config = config

        self.model = create_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------------
        # Dataset configuration
        # ------------------------------------------------------------------
        dataset_config = config.get("dataset", {})

        dataset_name = dataset_config.get("name", "mnist")
        partition_type = dataset_config.get("partition_type", "iid")
        alpha = dataset_config.get("alpha", 0.5)

        # ------------------------------------------------------------------
        # Load client data partition
        # ------------------------------------------------------------------
        self.dataset_provider = FederatedDatasetProvider(
            num_clients=config.get("num_clients", 20),
            batch_size=config.get("local_training", {}).get("batch_size", 32),
            seed=config.get("seed", 42),
            dataset_name=dataset_name,
            partition_type=partition_type,
            alpha=alpha,
        )

        self.partition = self.dataset_provider.get_partition(
            self.partition_id
        )

        self.tracker = CommunicationTracker()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        start_time = time.time()
        round_num = int(config.get("round", 1))

        # Track downlink
        downlink_bytes = self.tracker.record_downlink(
            round_num,
            self.client_id,
            parameters,
        )

        # Update model with global parameters
        set_parameters(self.model, parameters)

        # Local training configuration
        epochs = self.config.get("local_training", {}).get("epochs", 1)
        learning_rate = self.config.get(
            "local_training", {}
        ).get("learning_rate", 0.01)

        # Train locally
        metrics = train(
            model=self.model,
            train_loader=self.partition.train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
        )

        # Get updated parameters
        updated_parameters = get_parameters(self.model)

        # Track uplink
        uplink_bytes = self.tracker.record_uplink(
            round_num,
            self.client_id,
            updated_parameters,
        )

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
            downlink_bytes=downlink_bytes,
        )

        return (
            updated_parameters,
            self.partition.num_examples,
            {
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "client_id": self.client_id,
            },
        )

    def evaluate(self, parameters, config):
        # Update model
        set_parameters(self.model, parameters)

        # Evaluate on client's local test partition
        loss, accuracy = evaluate(
            model=self.model,
            test_loader=self.partition.test_loader,
            device=self.device,
        )

        # Log evaluation
        log_client_evaluate(
            client_id=self.client_id,
            partition_id=self.partition_id,
            num_examples=self.partition.num_examples,
            loss=float(loss),
            accuracy=float(accuracy),
        )

        return (
            float(loss),
            self.partition.num_examples,
            {"accuracy": float(accuracy), "client_id": self.client_id},
        )


def client_fn(context: Context):
    """Flower ClientApp entry point."""

    # ----------------------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------------------
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------------------
    # Partition ID
    #
    # Prefer an explicitly provided partition-id from Flower.
    # The fallback uses the Windows-compatible process ID instead of fcntl.
    # ----------------------------------------------------------------------
    if "partition-id" in context.node_config:
        partition_id = int(context.node_config["partition-id"])
    else:
        partition_id = (
            int(context.node_id)
            % config.get("num_clients", 20)
        )

    client_id = f"client_{partition_id}"

    return TRACEClient(
        client_id=client_id,
        partition_id=partition_id,
        config=config,
    ).to_client()


app = ClientApp(client_fn=client_fn)