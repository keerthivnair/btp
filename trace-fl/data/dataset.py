"""
Dataset provider for federated learning experiments.

This module supports:
1. MNIST and Fashion-MNIST datasets.
2. IID dataset partitioning.
3. Dirichlet-based Non-IID partitioning.
4. Local client DataLoaders.
5. A global test DataLoader for server-side evaluation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from data.partition import LocalPartition, create_dirichlet_partitions


class FederatedDatasetProvider:
    """
    Provides dataset partitions for federated clients.

    Args:
        num_clients:
            Number of federated clients.

        batch_size:
            Batch size used by each client's DataLoader.

        seed:
            Random seed used for reproducible partitioning.

        dataset_name:
            Dataset to use:
                "mnist"
                "fashion_mnist"

        partition_type:
            Partitioning strategy:
                "iid"       -> equal-size random partitions
                "dirichlet" -> Non-IID class distribution

        alpha:
            Dirichlet concentration parameter.

            Smaller values generally produce stronger
            class imbalance across clients.

            Example:
                alpha=1.0
                alpha=0.5
                alpha=0.1
    """

    SUPPORTED_DATASETS = {"mnist", "fashion_mnist"}
    SUPPORTED_PARTITIONS = {"iid", "dirichlet"}

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        seed: int = 42,
        dataset_name: str = "mnist",
        partition_type: str = "iid",
        alpha: float = 0.5,
    ):
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_name = dataset_name.lower()
        self.partition_type = partition_type.lower()
        self.alpha = alpha

        # Validate configuration

        if num_clients <= 0:
            raise ValueError("num_clients must be greater than 0.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from {sorted(self.SUPPORTED_DATASETS)}."
            )

        if self.partition_type not in self.SUPPORTED_PARTITIONS:
            raise ValueError(
                f"Unsupported partition_type '{partition_type}'. "
                f"Choose from {sorted(self.SUPPORTED_PARTITIONS)}."
            )

        if self.partition_type == "dirichlet" and alpha <= 0:
            raise ValueError("alpha must be greater than 0.")

        # Dataset-specific preprocessing

        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,),
                    (0.3081,),
                ),
            ])

            dataset_class = datasets.MNIST

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.2860,),
                    (0.3530,),
                ),
            ])

            dataset_class = datasets.FashionMNIST

        # Load training dataset

        self.train_dataset = dataset_class(
            "./data",
            train=True,
            download=True,
            transform=transform,
        )

        # Load test dataset

        self.test_dataset = dataset_class(
            "./data",
            train=False,
            download=True,
            transform=transform,
        )

        # Dataset metadata

        self.num_classes = len(self.train_dataset.classes)
        self.class_names = self.train_dataset.classes

        # Create client partitions

        self._partition_datasets()


    # Partition creation


    def _partition_datasets(self):
        """
        Create training and local-test partitions.

        Training data uses the selected partitioning strategy.

        Local test data remains IID and is split deterministically
        between clients.

        The complete test dataset is kept unchanged for
        server-side global evaluation.
        """

        if self.partition_type == "iid":
            self._create_iid_train_partitions()
        else:
            self._create_dirichlet_train_partitions()

        # Local test partitions remain IID.
        self._create_test_partitions()

    def _create_iid_train_partitions(self):
        """Create equal-sized IID training partitions."""

        generator = torch.Generator().manual_seed(self.seed)

        partition_size = len(self.train_dataset) // self.num_clients

        lengths = [partition_size] * self.num_clients

        # Give any remaining samples to the final client.
        lengths[-1] += len(self.train_dataset) - sum(lengths)

        self.train_partitions = random_split(
            self.train_dataset,
            lengths,
            generator=generator,
        )

    def _create_dirichlet_train_partitions(self):
        """
        Create Non-IID training partitions using Dirichlet sampling.
        """

        # Both MNIST and Fashion-MNIST expose their labels
        # through the targets attribute.
        labels = np.asarray(self.train_dataset.targets)

        client_indices = create_dirichlet_partitions(
            labels=labels,
            num_clients=self.num_clients,
            alpha=self.alpha,
            seed=self.seed,
        )

        # Convert lists of indices into PyTorch Subset objects.
        self.train_partitions = [
            Subset(
                self.train_dataset,
                client_indices[client_id],
            )
            for client_id in range(self.num_clients)
        ]

    def _create_test_partitions(self):
        """Create deterministic IID local test partitions."""

        generator = torch.Generator().manual_seed(self.seed)

        test_partition_size = len(self.test_dataset) // self.num_clients

        test_lengths = [test_partition_size] * self.num_clients

        # Give any remaining samples to the final client.
        test_lengths[-1] += len(self.test_dataset) - sum(test_lengths)

        self.test_partitions = random_split(
            self.test_dataset,
            test_lengths,
            generator=generator,
        )

    # Client access

    def get_partition(self, partition_id: int) -> LocalPartition:
        """
        Return the dataset partition belonging to one client.

        Args:
            partition_id:
                ID of the federated client.

        Returns:
            LocalPartition containing:
                - training DataLoader
                - local test DataLoader
                - number of training examples
        """

        if partition_id < 0 or partition_id >= self.num_clients:
            raise ValueError(
                f"Invalid partition_id {partition_id}. "
                f"Must be between 0 and {self.num_clients - 1}."
            )

        # Separate generator for reproducible DataLoader shuffling.
        train_generator = torch.Generator().manual_seed(
            self.seed + partition_id
        )

        train_loader = DataLoader(
            self.train_partitions[partition_id],
            batch_size=self.batch_size,
            shuffle=True,
            generator=train_generator,
        )

        test_loader = DataLoader(
            self.test_partitions[partition_id],
            batch_size=self.batch_size,
            shuffle=False,
        )

        num_examples = len(self.train_partitions[partition_id])

        return LocalPartition(
            partition_id=partition_id,
            train_loader=train_loader,
            test_loader=test_loader,
            num_examples=num_examples,
        )

    # Server-side evaluation

    def get_global_test_loader(self) -> DataLoader:
        """
        Return the complete test dataset.

        This loader is used by the server for global evaluation.
        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )