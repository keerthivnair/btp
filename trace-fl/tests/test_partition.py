import numpy as np

from data.partition import create_dirichlet_partitions


def test_dirichlet_partition_covers_all_samples():
    """Every dataset sample should belong to exactly one client."""

    labels = np.array([0, 1, 2, 3, 4] * 100)

    partitions = create_dirichlet_partitions(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    # Check that all clients exist.
    assert set(partitions.keys()) == set(range(20))

    # Combine all client indices.
    all_indices = [
        index
        for client_indices in partitions.values()
        for index in client_indices
    ]

    # Every sample should appear exactly once.
    assert len(all_indices) == len(labels)
    assert len(set(all_indices)) == len(labels)

    # All indices must be valid.
    assert set(all_indices) == set(range(len(labels)))


def test_dirichlet_partition_is_reproducible():
    """The same seed should produce the same partition."""

    labels = np.array([0, 1, 2, 3, 4] * 100)

    partition_a = create_dirichlet_partitions(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    partition_b = create_dirichlet_partitions(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    assert partition_a == partition_b


def test_dirichlet_partition_changes_with_seed():
    """Different seeds should normally produce different partitions."""

    labels = np.array([0, 1, 2, 3, 4] * 100)

    partition_a = create_dirichlet_partitions(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    partition_b = create_dirichlet_partitions(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=123,
    )

    assert partition_a != partition_b


def test_dirichlet_partition_rejects_invalid_parameters():
    """Invalid client count and alpha should raise ValueError."""

    labels = np.array([0, 1, 2, 3])

    try:
        create_dirichlet_partitions(
            labels,
            num_clients=0,
            alpha=0.5,
        )
        assert False
    except ValueError:
        pass

    try:
        create_dirichlet_partitions(
            labels,
            num_clients=2,
            alpha=0,
        )
        assert False
    except ValueError:
        pass