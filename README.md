# TRACE-FL — Phase 1: FL Foundation + Attack Benchmark

## 1. Goal

Phase 1 establishes the **basic federated-learning experiment platform** that every later TRACE-FL component will use.

By the end of Phase 1, the team should be able to:

```text
Create clients
    ↓
Give every client different local data
    ↓
Train locally
    ↓
Send model updates
    ↓
Aggregate with FedAvg
    ↓
Repeat for multiple rounds
    ↓
Inject attacks into selected clients
    ↓
Measure whether malicious updates look different
```

The purpose is **not to build the final TRACE-FL defense yet**.

The purpose is to create a reliable experimental foundation.

---

# 2. Technology Choice

We will use:

* Python
* PyTorch
* Flower
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

For Phase 1, we will use **Flower Simulation**.

We will NOT initially deploy separate physical servers or cloud machines.

A single machine will simulate the federated system using multiple virtual clients.

```text
                 ONE MACHINE

              Flower ServerApp
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Client 1    Client 2    Client 3
        ↓           ↓           ↓
      Data 1      Data 2      Data 3

              ... up to 20 clients
```

---

# 3. What "Server" Means

The **Flower ServerApp** is simply our server-side federated-learning logic.

It is responsible for:

* starting federated rounds
* selecting clients
* sending the current global model
* receiving client updates
* aggregating updates using FedAvg
* producing the next global model

We are NOT writing a custom socket server.

Flower provides the federation runtime.

Our code mainly defines what the server should do.

---

# 4. What "Client" Means

A **Flower ClientApp** represents one participating FL client.

Each client owns its own local dataset.

For example:

```text
Client 1 → local dataset D1
Client 2 → local dataset D2
Client 3 → local dataset D3
...
Client 20 → local dataset D20
```

Each client:

1. receives the current global model
2. trains locally
3. produces an updated model
4. sends the update back

The local dataset never needs to be combined into one centralized training dataset.

---

# 5. Phase 1 Architecture

```text
                    PHASE 1
              FLOWER SIMULATION

                 ServerApp
                    │
             Global Model
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
 ClientApp 1   ClientApp 2   ClientApp N
      │             │             │
     D1            D2            DN
      │             │             │
 Local Training  Local Training  Local Training
      │             │             │
      └─────────────┼─────────────┘
                    ↓
             Client Updates
                    │
                    ▼
               ServerApp
                    │
                  FedAvg
                    │
                    ▼
              Global Model
                    │
                    ▼
                Next Round
```

---

# 6. Team Split

## Member 1 — Flower + FL Infrastructure

### Main responsibility

Build the federated-learning skeleton.

### Implement

* Flower project
* `ServerApp`
* `ClientApp`
* FedAvg
* round lifecycle
* model parameter exchange
* client registration/identification
* basic logging
* communication byte counter

### Important clarification

Member 1 is NOT building a FastAPI backend yet.

The objective is:

```text
Flower ServerApp
        +
Flower ClientApps
        +
working FL rounds
```

### Deliverable

A working command that can run:

```text
20 clients
→ local training
→ FedAvg
→ 20 rounds
→ final global model
```

---

# 7. Member 2 — Dataset + Non-IID Pipeline

### Main responsibility

Create realistic federated client datasets.

### Initial datasets

Start with:

```text
MNIST
Fashion-MNIST
```

### Partitioning

Implement:

### IID

Every client gets approximately the same class distribution.

```text
Client 1 → balanced
Client 2 → balanced
Client 3 → balanced
```

### Non-IID

Use Dirichlet partitioning:

```text
α = 1.0
α = 0.5
α = 0.1
```

Smaller α means stronger heterogeneity.

Example:

```text
Client 1 → mostly class 1
Client 2 → mostly class 7
Client 3 → classes 2, 5
Client 4 → classes 3, 8
```

### Deliverables

* dataset loaders
* partition generator
* reproducible random seeds
* client dataloaders
* class-distribution plots

---

# 8. Member 3 — Attack Engine

### Main responsibility

Create malicious clients without modifying the core FL system.

The attack module should behave like a plug-in.

```text
Normal Client
      ↓
Local Training
      ↓
Update

Attacker Client
      ↓
Local Training
      ↓
Attack
      ↓
Malicious Update
```

### Initial attacks

Implement:

```text
1. Label Flipping
2. Gaussian Noise
3. Sign Flip
4. Scaling / Boosting
```

### Attack configuration

Every attack should be configurable.

Example:

```yaml
attack:
  enabled: true
  type: sign_flip
  intensity: 1.0
  start_round: 5
  end_round: 15
```

Also support:

```text
continuous attacker
intermittent attacker
```

Example:

```text
Round 1 → honest
Round 2 → honest
Round 3 → attack
Round 4 → honest
Round 5 → attack
```

### Deliverable

A reusable:

```text
AttackEngine
```

that can be attached to any client.

---

# 9. Member 4 — Detection Baseline

### Main responsibility

Study the model updates produced by clients.

For every client update, record:

```text
client_id
round
update norm
cosine similarity
CKA similarity
```

Initially, CKA is only being studied as a **research signal**.

We are NOT yet implementing the final TRACE-FL trust mechanism.

### Baseline aggregators

Implement/reproduce:

```text
FedAvg
Krum
Median
Trimmed Mean
```

The goal is to establish how existing aggregation behaves under attacks.

### Deliverables

Initial plots:

```text
Update norm:
honest vs malicious

Cosine similarity:
honest vs malicious

CKA:
honest vs malicious
```

---

# 10. Complete Phase 1 Flow

```text
                 DATASET
                    │
                    ▼
             Partition Dataset
                    │
          ┌─────────┴─────────┐
          │                   │
        IID                 Non-IID
          │                   │
          └─────────┬─────────┘
                    ▼
             Flower Clients
                    │
          ┌─────────┴─────────┐
          │                   │
       Honest              Attacker
       Client                Client
          │                   │
          │              AttackEngine
          │                   │
          └─────────┬─────────┘
                    ▼
              Local Training
                    │
                    ▼
             Client Updates
                    │
                    ▼
               ServerApp
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        Norm      Cosine      CKA
          │         │         │
          └─────────┼─────────┘
                    ▼
            Baseline Analysis
                    │
                    ▼
          FedAvg / Krum / Median
                    │
                    ▼
             Experimental Results
```

---

# 11. Phase 1 Experiments

Use initially:

```text
20 clients
20 communication rounds
```

Then vary:

```text
IID
Dirichlet α = 1.0
Dirichlet α = 0.5
Dirichlet α = 0.1
```

### Attack percentage

Start with:

```text
10% malicious
20% malicious
```

Later expand.

---

# 12. Metrics

Record:

### Model performance

```text
Global accuracy
Training loss
Round-to-round convergence
```

### Attack impact

```text
Accuracy degradation
Global model degradation
```

### Update behavior

```text
Update norm
Cosine similarity
CKA similarity
```

### Communication

```text
Model bytes downloaded
Model bytes uploaded
Total bytes per round
Total bytes across experiment
```

---

# 13. Phase 1 Main Research Question

The most important experiment is:

> **Under non-IID data, can representation-level similarity such as CKA distinguish malicious updates from legitimate client heterogeneity?**

We specifically want to avoid assuming:

```text
different update = malicious
```

because non-IID clients naturally produce different updates.

Instead we want to observe:

```text
                   CLIENT UPDATES

              ┌────────────────────┐
              │                    │
         Honest non-IID        Malicious
              │                    │
              ▼                    ▼
        Natural drift      Poisoned drift
              │                    │
              └────────┬───────────┘
                       ▼
                 CKA / Norm /
                 Cosine Analysis
                       │
                       ▼
             Is there separation?
```

If CKA gives useful separation, it becomes the foundation for the next phase.

If it does not, we investigate why and add complementary signals.

---

# 14. What We Are NOT Building in Phase 1

Do NOT implement these yet:

```text
✗ Temporal trust
✗ Authentication challenge-response
✗ PKI
✗ Sybil detection
✗ Collusion graph
✗ Adaptive attacker
✗ Feature freezing
✗ Virtual aggregation
✗ Communication scheduler
✗ TRACE-FL final trust engine
✗ TRACE-FL-NIDS
✗ Production cloud deployment
```

These belong to later phases.

---

# 15. Phase 1 Definition of Done

Phase 1 is complete when the team can run:

```text
20 Flower clients
        ↓
MNIST / Fashion-MNIST
        ↓
IID / non-IID
        ↓
FedAvg
        ↓
attack selected clients
        ↓
collect client updates
        ↓
compute norm / cosine / CKA
        ↓
compare honest vs malicious behavior
        ↓
run baseline aggregators
        ↓
produce reproducible plots
```

And all four members' modules integrate into the same pipeline.

---

# 16. Repository Structure

```text
trace-fl/
│
├── client/
│   ├── client_app.py
│   ├── trainer.py
│   └── model.py
│
├── server/
│   ├── server_app.py
│   └── strategies.py
│
├── datasets/
│   ├── mnist.py
│   ├── fashion_mnist.py
│   └── partition.py
│
├── attacks/
│   ├── label_flip.py
│   ├── gaussian.py
│   ├── sign_flip.py
│   ├── scaling.py
│   └── attack_engine.py
│
├── detection/
│   ├── norms.py
│   ├── cosine.py
│   └── cka.py
│
├── aggregation/
│   ├── fedavg.py
│   ├── krum.py
│   ├── median.py
│   └── trimmed_mean.py
│
├── experiments/
│   ├── configs/
│   └── results/
│
├── plots/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

---

# 17. What comes immediately after Phase 1

Once Phase 1 produces reliable evidence:

```text
Phase 1
FL + attacks + CKA evidence
        ↓
Phase 2
Temporal Trust
        ↓
Phase 3
Identity / Authentication
        ↓
Phase 4
Trust-aware Aggregation
        ↓
Phase 5
Communication Optimization
        ↓
Phase 6
Adaptive / Collusion Attacks
        ↓
Phase 7
NIDS Extension
```

That way you're **not overengineering before you know your central detection hypothesis works**.
