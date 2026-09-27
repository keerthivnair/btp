#!/bin/sh
# Flower (>=1.32) defaults to 2 simulated SuperNodes, so match it to num_clients before running.
set -eu
cd "$(dirname "$0")/.."

# Otherwise the local SuperLink re-resolves dependencies per run, ignoring uv.lock.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

NUM_CLIENTS=$(python -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['num_clients'])")

flwr federation simulation-config --num-supernodes "$NUM_CLIENTS"
exec flwr run . --stream "$@"
