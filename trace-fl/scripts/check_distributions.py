import numpy as np

from data.dataset import FederatedDatasetProvider


def get_class_distribution(dataset, indices):
    """Count how many samples of each class are assigned to a client."""
    labels = np.asarray(dataset.targets)

    client_labels = labels[indices]

    classes, counts = np.unique(
        client_labels,
        return_counts=True,
    )

    return dict(zip(classes.tolist(), counts.tolist()))


def print_distributions(provider, name):
    """Print class distribution for every client."""

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for client_id, partition in enumerate(provider.train_partitions):

        indices = partition.indices

        distribution = get_class_distribution(
            provider.train_dataset,
            indices,
        )

        print(
            f"Client {client_id:2d} | "
            f"Samples: {len(indices):5d} | "
            f"Classes: {distribution}"
        )


def main():

    num_clients = 20
    batch_size = 32
    seed = 42

    # ------------------------------------------------------------------
    # 1. IID
    # ------------------------------------------------------------------
    iid_provider = FederatedDatasetProvider(
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
        partition_type="iid",
    )

    print_distributions(
        iid_provider,
        "IID PARTITION",
    )

    # ------------------------------------------------------------------
    # 2. Dirichlet alpha = 1.0
    # ------------------------------------------------------------------
    alpha_1_provider = FederatedDatasetProvider(
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
        partition_type="dirichlet",
        alpha=1.0,
    )

    print_distributions(
        alpha_1_provider,
        "DIRICHLET PARTITION - ALPHA = 1.0",
    )

    # ------------------------------------------------------------------
    # 3. Dirichlet alpha = 0.5
    # ------------------------------------------------------------------
    alpha_05_provider = FederatedDatasetProvider(
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
        partition_type="dirichlet",
        alpha=0.5,
    )

    print_distributions(
        alpha_05_provider,
        "DIRICHLET PARTITION - ALPHA = 0.5",
    )

    # ------------------------------------------------------------------
    # 4. Dirichlet alpha = 0.1
    # ------------------------------------------------------------------
    alpha_01_provider = FederatedDatasetProvider(
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
        partition_type="dirichlet",
        alpha=0.1,
    )

    print_distributions(
        alpha_01_provider,
        "DIRICHLET PARTITION - ALPHA = 0.1",
    )


if __name__ == "__main__":
    main()