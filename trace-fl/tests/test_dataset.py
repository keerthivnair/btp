import pytest
from data.dataset import FederatedDatasetProvider

def test_dataset_partitioning():
    provider = FederatedDatasetProvider(num_clients=2, batch_size=32, seed=42)
    
    partition_0 = provider.get_partition(0)
    partition_1 = provider.get_partition(1)
    
    assert partition_0.partition_id == 0
    assert partition_1.partition_id == 1
    assert partition_0.num_examples > 0
    assert partition_1.num_examples > 0

def test_invalid_partition():
    provider = FederatedDatasetProvider(num_clients=2, batch_size=32, seed=42)
    with pytest.raises(ValueError):
        provider.get_partition(2)
