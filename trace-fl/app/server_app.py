import yaml
import time
import torch
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

from core.model import create_model, get_parameters, set_parameters
from core.parameters import flower_parameters_to_ndarrays, ndarrays_to_flower_parameters
from federation.strategy import FedAvgStrategy
from federation.client_selector import RandomClientSelector
from federation.update_validator import UpdateValidator
from federation.round_state import RoundState
from telemetry.communication import CommunicationTracker
from telemetry.logging import log_round, logger
from training.trainer import test
from data.dataset import FederatedDatasetProvider

class TRACEStrategy(Strategy):
    def __init__(self, config: dict):
        self.config = config
        self.aggregation_strategy = FedAvgStrategy()
        
        # In simulation, clients are sequential indices.
        seed = self.config.get("seed", 42)
        self.client_selector = RandomClientSelector(seed=seed)
        
        # Update validator expects parameter shapes matching the global model
        self.global_model = create_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        initial_params = get_parameters(self.global_model)
        expected_shapes = [p.shape for p in initial_params]
        self.update_validator = UpdateValidator(expected_shapes=expected_shapes)
        
        self.round_state = RoundState()
        self.tracker = CommunicationTracker()
        
        self.min_available_clients = self.config.get("federation", {}).get("min_available_clients", 20)
        self.fraction_train = self.config.get("federation", {}).get("fraction_train", 1.0)
        
        # Test dataset for global evaluation
        self.dataset_provider = FederatedDatasetProvider(
            num_clients=self.config.get("num_clients", 20),
            batch_size=self.config.get("local_training", {}).get("batch_size", 32),
            seed=seed
        )
        self.test_loader = self.dataset_provider.get_global_test_loader()

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ndarrays = get_parameters(self.global_model)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.round_state.start_round(server_round, [])
        self.round_state.round_start_time = time.time()
        
        # Sample clients
        sample_size = int(client_manager.num_available() * self.fraction_train)
        sample_size = max(sample_size, 1)
        
        # Use our custom selector conceptually
        available_clients_dict = client_manager.all()
        # Ensure we have enough clients
        if len(available_clients_dict) < self.min_available_clients:
            return []
            
        # Instead of directly using our selector, we use client_manager.sample 
        # which delegates to random sampling. To strictly follow the interface requirement, 
        # we can pass client IDs to our selector:
        cids = list(available_clients_dict.keys())
        selected_cids = self.client_selector.select(cids, sample_size)
        clients = [available_clients_dict[cid] for cid in selected_cids]
        
        self.round_state.selected_clients = selected_cids
        
        # Track downlink communication (estimated on client side, but we also track server side)
        ndarrays = parameters_to_ndarrays(parameters)
        for cid in selected_cids:
            self.tracker.record_downlink(server_round, cid, ndarrays)
            
        fit_ins = FitIns(parameters, {"round": server_round})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        agg_start_time = time.time()
        
        valid_updates = []
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            
            # Explicitly log the client's local metrics to bypass Ray's log deduplication
            metrics = fit_res.metrics
            cid_str = metrics.get('client_id', client.cid)
            logger.info(
                f"Client {cid_str} local training complete -> "
                f"Loss: {float(metrics.get('loss', 0.0)):.4f} | "
                f"Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}"
            )
            
            # Track uplink
            self.tracker.record_uplink(server_round, client.cid, ndarrays)
            
            # Validate update
            is_valid = self.update_validator.validate(
                client_id=client.cid, 
                num_examples=fit_res.num_examples, 
                parameters=ndarrays
            )
            
            if is_valid:
                valid_updates.append((fit_res.num_examples, ndarrays))
                self.round_state.successful_clients.append(client.cid)
            else:
                self.round_state.failed_clients.append(client.cid)
                
        if not valid_updates:
            return None, {}
            
        # Aggregate
        aggregated_ndarrays = self.aggregation_strategy.aggregate(valid_updates)
        
        self.round_state.aggregation_time = time.time() - agg_start_time
        
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation. Disabled per user request."""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
            
        total_examples = sum(res.num_examples for _, res in results)
        
        # Weighted average of loss
        agg_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        
        # Weighted average of accuracy
        agg_acc = sum(res.metrics.get("accuracy", 0.0) * res.num_examples for _, res in results) / total_examples
        
        return agg_loss, {"accuracy": agg_acc}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(self.global_model, ndarrays)
        
        loss, accuracy = test(self.global_model, self.test_loader, self.device)
        
        self.round_state.global_loss = loss
        self.round_state.global_accuracy = accuracy
        self.round_state.round_duration = time.time() - self.round_state.round_start_time
        
        round_totals = self.tracker.get_round_totals(server_round)
        
        # Log round telemetry
        log_round(
            round_num=server_round,
            selected_clients=self.round_state.selected_clients,
            successful_clients=self.round_state.successful_clients,
            failed_clients=self.round_state.failed_clients,
            global_loss=loss,
            global_accuracy=accuracy,
            uplink_bytes=round_totals["uplink_bytes"],
            downlink_bytes=round_totals["downlink_bytes"],
            aggregation_time=self.round_state.aggregation_time,
            round_time=self.round_state.round_duration
        )
        
        return loss, {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    """Flower ServerApp entry point."""
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    num_rounds = config.get("num_rounds", 20)
    
    strategy = TRACEStrategy(config)
    config_obj = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config_obj)

app = ServerApp(server_fn=server_fn)
