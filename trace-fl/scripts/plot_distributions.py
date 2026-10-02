
"""
Visualize federated dataset distributions.

Supports:
    - MNIST
    - Fashion-MNIST

For each dataset, generates:
    1. Client-vs-class heatmaps.
    2. 100% stacked bar charts.
    3. Client dataset-size comparison.

Partitioning strategies:
    - IID
    - Dirichlet alpha=1.0
    - Dirichlet alpha=0.5
    - Dirichlet alpha=0.1

Output structure:

artifacts/
└── distributions/
    ├── mnist/
    └── fashion_mnist/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.dataset import FederatedDatasetProvider


# ============================================================================
# Configuration
# ============================================================================

NUM_CLIENTS = 20
BATCH_SIZE = 32
SEED = 42

BASE_OUTPUT_DIR = Path("artifacts/distributions")

DATASETS = [
    "mnist",
    "fashion_mnist",
]


# ============================================================================
# Dataset provider
# ============================================================================

def create_provider(
    dataset_name,
    partition_type,
    alpha=None,
):
    """Create a dataset provider for one experiment."""

    kwargs = {
        "num_clients": NUM_CLIENTS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "dataset_name": dataset_name,
        "partition_type": partition_type,
    }

    if alpha is not None:
        kwargs["alpha"] = alpha

    return FederatedDatasetProvider(**kwargs)


# ============================================================================
# Distribution matrix
# ============================================================================

def get_distribution_matrix(provider):
    """
    Build a client-vs-class distribution matrix.

    Rows:
        Client IDs.

    Columns:
        Dataset classes.

    Values:
        Number of samples of each class assigned
        to each client.
    """

    labels = np.asarray(
        provider.train_dataset.targets
    )

    matrix = np.zeros(
        (
            provider.num_clients,
            provider.num_classes,
        ),
        dtype=int,
    )

    for client_id, partition in enumerate(
        provider.train_partitions
    ):
        indices = np.asarray(
            partition.indices
        )

        client_labels = labels[indices]

        classes, counts = np.unique(
            client_labels,
            return_counts=True,
        )

        matrix[
            client_id,
            classes,
        ] = counts

    return matrix


# ============================================================================
# Heatmap
# ============================================================================

def plot_heatmap(
    matrix,
    class_names,
    dataset_name,
    title,
    output_path,
):
    """Create and save a client-vs-class heatmap."""

    plt.figure(
        figsize=(14, 8)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Number of Samples"
    )

    plt.xlabel("Class")
    plt.ylabel("Client ID")

    plt.title(
        f"{dataset_name} — {title}"
    )

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(NUM_CLIENTS)
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# 100% stacked bar chart
# ============================================================================

def plot_stacked_distribution(
    matrix,
    class_names,
    dataset_name,
    title,
    output_path,
):
    """
    Create a 100% stacked bar chart showing
    class composition for every client.
    """

    client_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    percentages = np.divide(
        matrix,
        client_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=client_totals != 0,
    ) * 100

    client_ids = np.arange(
        matrix.shape[0]
    )

    plt.figure(
        figsize=(16, 8)
    )

    bottom = np.zeros(
        matrix.shape[0]
    )

    for class_id, class_name in enumerate(
        class_names
    ):
        values = percentages[
            :,
            class_id,
        ]

        plt.bar(
            client_ids,
            values,
            bottom=bottom,
            label=class_name,
        )

        bottom += values

    plt.xlabel("Client ID")

    plt.ylabel(
        "Class Distribution (%)"
    )

    plt.title(
        f"{dataset_name} — {title}"
    )

    plt.xticks(
        client_ids
    )

    plt.ylim(
        0,
        100,
    )

    plt.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Client-size comparison
# ============================================================================

def plot_client_sizes(
    results,
    dataset_name,
    output_path,
):
    """Compare training samples assigned to each client."""

    plt.figure(
        figsize=(14, 7)
    )

    client_ids = np.arange(
        NUM_CLIENTS
    )

    for name, matrix in results.items():

        client_sizes = matrix.sum(
            axis=1
        )

        plt.plot(
            client_ids,
            client_sizes,
            marker="o",
            label=name,
        )

    plt.xlabel("Client ID")

    plt.ylabel(
        "Number of Training Samples"
    )

    plt.title(
        f"Training Samples per Client — {dataset_name}"
    )

    plt.xticks(
        client_ids
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Generate plots for one dataset
# ============================================================================

def generate_dataset_plots(
    dataset_name,
):
    """Generate all plots for one dataset."""

    output_dir = (
        BASE_OUTPUT_DIR / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(f"Generating plots for: {dataset_name}")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Get dataset metadata
    # ------------------------------------------------------------------------

    metadata_provider = create_provider(
        dataset_name=dataset_name,
        partition_type="iid",
    )

    class_names = metadata_provider.class_names

    print(f"Dataset     : {dataset_name}")
    print(f"Clients     : {NUM_CLIENTS}")
    print(f"Classes     : {len(class_names)}")
    print(f"Class names : {class_names}")

    # ------------------------------------------------------------------------
    # Partitioning experiments
    # ------------------------------------------------------------------------

    experiments = {
        "IID": {
            "partition_type": "iid",
            "alpha": None,
            "heatmap": "iid_distribution.png",
            "stacked": "iid_stacked_distribution.png",
        },

        "Dirichlet α=1.0": {
            "partition_type": "dirichlet",
            "alpha": 1.0,
            "heatmap": "dirichlet_alpha_1.0.png",
            "stacked": "dirichlet_alpha_1.0_stacked.png",
        },

        "Dirichlet α=0.5": {
            "partition_type": "dirichlet",
            "alpha": 0.5,
            "heatmap": "dirichlet_alpha_0.5.png",
            "stacked": "dirichlet_alpha_0.5_stacked.png",
        },

        "Dirichlet α=0.1": {
            "partition_type": "dirichlet",
            "alpha": 0.1,
            "heatmap": "dirichlet_alpha_0.1.png",
            "stacked": "dirichlet_alpha_0.1_stacked.png",
        },
    }

    results = {}

    # ------------------------------------------------------------------------
    # Generate each experiment
    # ------------------------------------------------------------------------

    for name, settings in experiments.items():

        print()
        print(f"Generating: {name}")

        provider = create_provider(
            dataset_name=dataset_name,
            partition_type=settings[
                "partition_type"
            ],
            alpha=settings["alpha"],
        )

        matrix = get_distribution_matrix(
            provider
        )

        results[name] = matrix

        # --------------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------------

        plot_heatmap(
            matrix=matrix,
            class_names=class_names,
            dataset_name=dataset_name,
            title=f"Client Class Distribution — {name}",
            output_path=(
                output_dir
                / settings["heatmap"]
            ),
        )

        # --------------------------------------------------------------
        # 100% stacked bar chart
        # --------------------------------------------------------------

        plot_stacked_distribution(
            matrix=matrix,
            class_names=class_names,
            dataset_name=dataset_name,
            title=f"Client Class Composition — {name}",
            output_path=(
                output_dir
                / settings["stacked"]
            ),
        )

    # ------------------------------------------------------------------------
    # Client-size comparison
    # ------------------------------------------------------------------------

    plot_client_sizes(
        results=results,
        dataset_name=dataset_name,
        output_path=(
            output_dir
            / "client_sizes_comparison.png"
        ),
    )

    print()
    print(f"Completed: {dataset_name}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate plots for MNIST and Fashion-MNIST."""

    print()
    print("=" * 70)
    print("FEDERATED DATASET DISTRIBUTION VISUALIZATION")
    print("=" * 70)

    for dataset_name in DATASETS:
        generate_dataset_plots(
            dataset_name
        )

    print()
    print("=" * 70)
    print("ALL DATASET PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()