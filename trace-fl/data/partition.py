from torch.utils.data import DataLoader

class LocalPartition:
    """Represents a local dataset partition for a specific client."""
    def __init__(self, partition_id: int, train_loader: DataLoader, test_loader: DataLoader, num_examples: int):
        self.partition_id = partition_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_examples = num_examples
