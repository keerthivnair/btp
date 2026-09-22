import torch
from core.model import create_model, get_parameters
from data.dataset import FederatedDatasetProvider
from training.trainer import train, test

def test_training_step():
    provider = FederatedDatasetProvider(num_clients=2, batch_size=32, seed=42)
    partition = provider.get_partition(0)
    model = create_model()
    
    initial_params = get_parameters(model)
    
    device = torch.device("cpu")
    
    metrics = train(
        model=model,
        train_loader=partition.train_loader,
        epochs=1,
        learning_rate=0.01,
        device=device
    )
    
    assert "loss" in metrics
    assert "accuracy" in metrics
    
    trained_params = get_parameters(model)
    
    # Check that weights changed
    changed = False
    for p1, p2 in zip(initial_params, trained_params):
        if not (p1 == p2).all():
            changed = True
            break
            
    assert changed
