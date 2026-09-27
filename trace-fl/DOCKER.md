# Running TRACE-FL in Docker

TRACE-FL runs as a **single container** that holds the whole Flower simulation: the server and every simulated client. The goal is for every team member to get the same Python, the same library versions and the same results, whether they use macOS (Intel or Apple Silicon), Linux or Windows.

## Quickstart

You need Docker Desktop (macOS or Windows) or Docker Engine with the Compose plugin (Linux). Nothing else has to be installed on the host: no Python, no `uv`, no PyTorch.

```bash
cd trace-fl

docker compose build                          # first build: ~5 min (downloads PyTorch + MNIST)
docker compose run --rm trace-fl              # run the FL simulation
docker compose run --rm trace-fl pytest       # run the test suite
docker compose run --rm trace-fl pytest tests/test_model.py   # a single test file
docker compose run --rm trace-fl bash         # shell inside the container
```

- **Changing the experiment:** edit `config/config.yaml` (clients, rounds, epochs, batch size, learning rate, seed) and run again. No rebuild is needed, because `config/` is mounted into the container.
- **Changing Python code or dependencies:** rebuild with `docker compose build`.

## What is inside the container

```
┌──────────────────────── trace-fl container ────────────────────────┐
│  scripts/run_simulation.sh                                         │
│    1. flwr federation simulation-config --num-supernodes <N>       │
│    2. flwr run . --stream                                          │
│                                                                    │
│  flwr run ──starts──► local SuperLink (background, simulation mode)│
│                          │                                         │
│                          ▼                                         │
│                   Simulation runtime (Ray)                         │
│                   ServerApp ── TRACEStrategy (FedAvg)              │
│                      │                                             │
│          ┌───────────┼───────────┐                                 │
│      ClientApp 0  ClientApp 1 … ClientApp N-1                      │
│      partition 0  partition 1   partition N-1                      │
│                                                                    │
│  /opt/venv      locked Python 3.11.14 environment (from uv.lock)   │
│  /app           project source                                     │
│  /app/data/MNIST  dataset, downloaded when the image is built      │
└────────────────────────────────────────────────────────────────────┘
```

Everything runs in one container because Phase 1 uses Flower **simulation**: clients are virtual and scheduled by Ray, not separate machines. The container is a reproducible runtime, not a network of nodes.

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | `python:3.11.14-slim-bookworm` + `uv` 0.12.19; installs `uv.lock` with `uv sync --frozen`; downloads MNIST at build time |
| `docker-compose.yml` | Sets `shm_size` for Ray; mounts `config/` (read-only) and `experiments/` |
| `.dockerignore` | Keeps the host's `.venv`, caches and downloaded data out of the image |
| `.gitattributes` | Forces LF line endings so shell scripts work when the repo is cloned on Windows |
| `scripts/run_simulation.sh` | Sets the simulated client count to `num_clients` and starts the run; also works outside Docker |

## Cross-platform problems this setup fixes

These issues were found while containerizing the project. Each one could make the project fail or silently behave differently from one machine to another.

### 1. Python and dependency drift
`pyproject.toml` pins Python to exactly `3.11.14`, but nothing enforced it. For example, one team member's Mac had a Homebrew Python 3.14 `.venv` with no packages installed and no `uv`. The image uses exactly 3.11.14 and installs dependencies with `uv sync --frozen`, so everyone gets the versions in `uv.lock`.

### 2. Flower re-installed dependencies on every run, ignoring `uv.lock`
In Flower 1.37, `flwr run .` starts a local SuperLink that by default creates a **fresh environment per run** and resolves dependencies again from `pyproject.toml`. It does not use `uv.lock`. One observed run got `numpy 2.4.6` even though the project pins `numpy<2`. It also needs internet access and adds about 20 s to every run.

Fix: `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` is set in the image and exported by `scripts/run_simulation.sh`. The run then uses the locked environment.

### 3. Flower's default of 2 simulated clients
Since Flower 1.32, a local simulation starts **2** SuperNodes unless told otherwise, regardless of `num_clients` in `config/config.yaml`:
- If `num_clients` is larger than 2, only partitions 0 and 1 train and the rest of the data is never used, with no error.
- If `num_clients` is smaller than the number of SuperNodes, a client gets a `partition-id` that doesn't exist and crashes.

Fix: `scripts/run_simulation.sh` reads `num_clients` and runs `flwr federation simulation-config --num-supernodes <num_clients>` before every run.

### 4. `.gitignore` excluded the `data/` Python package
The rule `data/` was meant for the MNIST download, but it also matched the `data/` source package. Existing files were only tracked because they were added before the rule existed, but:
- any new file in `data/` would have been silently ignored by git;
- `flwr run` builds the app bundle (FAB) using `.gitignore` rules, so the `data` package could be left out of the bundle.

Fix: the rule is now `/data/MNIST/`.

### 5. Dataset download races
Each client and the server call `datasets.MNIST(..., download=True)`. On a fresh machine several simulated clients can start downloading into the same folder at the same moment. The image downloads MNIST once at build time, so runs never touch the network.

### 6. Ray shared memory
Ray keeps its object store in `/dev/shm` and wants more than 30% of RAM there. Docker's default is 64 MB, so Ray falls back to slower disk storage. The compose file sets `shm_size: 3gb`. If you give Docker Desktop more than about 10 GB of RAM, raise this value too.

### 7. Unix-only partition fallback (`fcntl`, `/tmp`)
`app/client_app.py` has a fallback that assigns partitions with `fcntl.flock` on `/tmp/flwr_client_counter.txt`. That only works on Unix-like systems. Flower's simulation always passes `partition-id` in `node_config`, so the fallback isn't used inside the container, and a Linux container never hits the Windows problem anyway.

### 8. CPU architecture
`uv.lock` contains CPU PyTorch wheels for both `linux/arm64` (Apple Silicon) and `linux/amd64` (Intel/AMD). The image builds natively on either, with no emulation. Only the `arm64` build has been tested so far.

### 9. No GPU in the container
Docker on macOS cannot use the Apple GPU (MPS), and the lockfile pins CPU-only PyTorch. The code picks `cuda:0` only if it's available, so it runs on CPU everywhere. That's fine for MNIST-sized experiments.

## Running without Docker

The same fixes apply to local runs. With `uv` installed:

```bash
cd trace-fl
uv sync --extra dev
uv run sh scripts/run_simulation.sh   # use this instead of bare `flwr run .` (see problems 2 and 3)
uv run pytest
```

## Resource notes

- 5 clients × 3 rounds takes about 1 minute on a 10-core Apple Silicon Mac with Docker's default 8 GB of RAM.
- By default Flower gives each simulated client 2 CPUs, so the number of clients that train at the same time is roughly the Docker CPU count divided by 2.
- For 20-client experiments, give Docker Desktop more CPUs and memory (Settings → Resources).

## Next step: multi-container deployment (Option B)

This setup is **Option A**: one container running the simulation. **Option B** would run Flower's deployment mode with Docker Compose on the same machine:
- one SuperLink container (server);
- N SuperNode containers (clients) communicating over real gRPC.

That gives real network traffic, lets you kill a client or inject an attacker container, and prepares for later phases (authentication, trust). This image can be reused for B. The remaining work is:

1. A compose file with one `flower-superlink` service and N `flower-supernode` services.
2. Passing each SuperNode its partition explicitly: `--node-config "partition-id=<i> num-partitions=<N>"`. The `/tmp` counter fallback cannot work across containers, because each container has its own `/tmp` and every client would end up on partition 0.
3. Sharing the MNIST data with each container, either from the image (already done) or a shared volume.
4. Budgeting RAM: each SuperNode loads its own PyTorch (about 0.5 GB or more), so start with 3–5 clients.
