import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from data.partition import LocalPartition

class FederatedDatasetProvider:
    """Provides isolated dataset partitions for federated clients."""
    def __init__(self, num_clients: int, batch_size: int, seed: int = 42):
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.seed = seed
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load the MNIST dataset
        # We assume the dataset is stored in a common data/ directory
        self.train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            './data', train=False, download=True, transform=transform
        )
        
        # Partition the dataset
        self._partition_datasets()

    def _partition_datasets(self):
        """Deterministically splits the dataset into partitions."""
        # Use a fixed generator for reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        
        # Split train dataset
        partition_size = len(self.train_dataset) // self.num_clients
        lengths = [partition_size] * self.num_clients
        # Distribute remainder to the last partition
        lengths[-1] += len(self.train_dataset) - sum(lengths)
        
        self.train_partitions = random_split(self.train_dataset, lengths, generator=generator)
        
        # For evaluation, we can either split the test set or evaluate globally. 
        # Here we also split the test set for local evaluation if needed.
        test_partition_size = len(self.test_dataset) // self.num_clients
        test_lengths = [test_partition_size] * self.num_clients
        test_lengths[-1] += len(self.test_dataset) - sum(test_lengths)
        
        self.test_partitions = random_split(self.test_dataset, test_lengths, generator=generator)

    def get_partition(self, partition_id: int) -> LocalPartition:
        """Returns the isolated partition for a given client."""
        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(f"Invalid partition_id {partition_id}. Must be between 0 and {self.num_clients - 1}.")
            
        train_loader = DataLoader(
            self.train_partitions[partition_id], 
            batch_size=self.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_partitions[partition_id], 
            batch_size=self.batch_size, 
            shuffle=False
        )
        num_examples = len(self.train_partitions[partition_id])
        
        return LocalPartition(
            partition_id=partition_id,
            train_loader=train_loader,
            test_loader=test_loader,
            num_examples=num_examples
        )

    def get_global_test_loader(self) -> DataLoader:
        """Returns the full test dataset for global evaluation on the server."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
