"""
Dataset partitioning utilities for federated learning.

This module contains:
1. LocalPartition - container for a client's DataLoaders.
2. create_dirichlet_partitions - creates Non-IID client partitions
   using Dirichlet sampling.

A smaller Dirichlet alpha produces a more heterogeneous distribution
of classes across clients.
"""

from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader


def create_dirichlet_partitions(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Split dataset indices among clients using Dirichlet sampling.

    The partitioning is performed independently for each class. For each
    class, a Dirichlet distribution determines how that class's samples
    are distributed across the clients.

    Example:

        alpha = 1.0
            -> relatively balanced class distributions

        alpha = 0.5
            -> moderate class imbalance

        alpha = 0.1
            -> strong class imbalance

    Args:
        labels:
            One label for every sample in the training dataset.

        num_clients:
            Number of federated clients.

        alpha:
            Dirichlet concentration parameter. Must be greater than 0.
            Smaller values produce stronger Non-IID behavior.

        seed:
            Random seed used to make the partition reproducible.

    Returns:
        Dictionary mapping each client ID to the list of dataset indices
        assigned to that client.

        Example:
            {
                0: [12, 45, 78, ...],
                1: [3, 19, 56, ...],
                ...
            }

    Raises:
        ValueError:
            If num_clients or alpha is invalid.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")

    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")

    rng = np.random.default_rng(seed)

    client_indices: Dict[int, List[int]] = {
        client_id: []
        for client_id in range(num_clients)
    }

    # Process each class independently so that the class distribution
    # across clients is controlled by the Dirichlet distribution.
    for class_label in np.unique(labels):

        class_indices = np.flatnonzero(labels == class_label)

        # Shuffle samples of this class before assigning them.
        rng.shuffle(class_indices)

        # Sample the proportion of this class that each client receives.
        class_proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        # Convert proportions into cumulative split positions.
        split_points = (
            np.cumsum(class_proportions) * len(class_indices)
        ).astype(int)[:-1]

        client_class_indices = np.split(
            class_indices,
            split_points,
        )

        # Add this class's samples to each client's partition.
        for client_id, indices in enumerate(client_class_indices):
            client_indices[client_id].extend(indices.tolist())

    # Shuffle each client's complete partition so that samples from
    # different classes are mixed.
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


class LocalPartition:
    """
    Stores the data loaders and metadata for one federated client.

    This class is kept compatible with the existing Flower ClientApp.
    """

    def __init__(
        self,
        partition_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_examples: int,
    ):
        self.partition_id = partition_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_examples = num_examples