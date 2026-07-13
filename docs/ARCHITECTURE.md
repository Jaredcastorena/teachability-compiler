# Proposed Architecture

## 1. System overview

```text
Raw corpus
    │
    ├── semantic metadata / domains
    ├── candidate chunks
    └── initial effect clusters
             │
             ▼
    Action Generator ─────────────┐
             │                    │
             ▼                    │
     Hierarchical MCTS            │
       │             │            │
       │ uses        │ proposes   │
       ▼             ▼            │
Transition Simulator      Curriculum actions
       ▲                         │
       │ corrected by            │
       │                         ▼
Oracle Branch Runner ◄── Target learner checkpoint
       │
       ├── probe deltas
       ├── update sketches
       ├── activation sketches
       ├── forgetting signatures
       └── next compressed state
```

## 2. Components

### 2.1 Corpus index

Stores each candidate unit with:

- immutable source identifier;
- token count;
- semantic/domain metadata;
- deduplication signature;
- current transition-effect embedding;
- exposure count and recency;
- provenance and trust fields;
- factual uniqueness estimate.

The transition-effect embedding is checkpoint-relative and must be versioned by learner state region.

### 2.2 State encoder

`StateEncoder` maps a target checkpoint plus training metadata into `LearningState`.

Initial fields:

- probe loss vector;
- probe logit sketch;
- projected recent update direction;
- optimizer moment norms/sketches;
- domain exposure histogram;
- history embedding;
- target architecture descriptor;
- uncertainty mask for missing fields.

### 2.3 Oracle branch runner

Given state \(x_t\) and action \(a\):

1. create temporary branch state;
2. execute a small number of real training steps;
3. evaluate fixed and rotating probes;
4. compute compressed transition targets;
5. discard branch weights after recording results.

Possible implementations:

- full state copy for very small models;
- LoRA/adapters for cheap local branches;
- functional optimizers with copy-on-write tensors;
- distributed branch workers sharing frozen base weights.

### 2.4 Transition simulator

Inputs:

```text
LearningState + CurriculumAction + architecture descriptor
```

Outputs:

```text
mean state delta
transition covariance / uncertainty
probe loss deltas
forgetting risk
novelty estimate
compute estimate
optional successor-state value features
```

Start with an ensemble of small MLPs. Only move to transformers or graph models after demonstrating that the state/action representation contains useful signal.

### 2.5 Action generator

Produces a manageable candidate set for each node:

- highest predicted immediate gain;
- highest novelty;
- strongest prerequisite candidates;
- strongest bridge candidates;
- replay candidates for recently damaged anchors;
- uncertainty-driven exploratory actions;
- random diversity actions.

### 2.6 Hierarchical MCTS

Search levels:

1. select domain/effect family;
2. select cluster or cluster mixture;
3. select recipe and duration;
4. optionally materialize concrete chunks.

Each node tracks:

- state belief;
- visits;
- value moments/quantiles;
- remaining budget;
- curriculum prefix hash;
- simulator version;
- real-verification status.

### 2.7 Adversary / robustness sampler

For each rollout, sample:

- simulator ensemble member;
- target seed descriptor;
- batch realization;
- probe weights;
- transition noise;
- optional conflicting-data injection.

Back up mean, variance, and lower-tail value.

### 2.8 Real execution controller

After planning a tranche:

1. execute it on the real target learner;
2. compare observed and simulated transitions;
3. log model error;
4. update simulator training data;
5. re-encode current state;
6. replan.

This is receding-horizon model-predictive curriculum control, not one static plan generated before training.

## 3. Data contracts

### LearningState

- `probe_losses: vector[float]`
- `probe_logits_sketch: vector[float]`
- `update_sketch: vector[float]`
- `optimizer_sketch: vector[float]`
- `activation_sketch: vector[float]`
- `exposure_histogram: vector[float]`
- `history_embedding: vector[float]`
- `architecture_embedding: vector[float]`
- `step: int`
- `tokens_seen: int`

### CurriculumAction

- `cluster_ids: tuple[str, ...]`
- `mixture_weights: tuple[float, ...]`
- `optimizer_steps: int`
- `token_budget: int`
- `learning_rate_scale: float`
- `replay_policy: str | None`
- `materialization_seed: int`

### TransitionObservation

- `state_before`
- `action`
- `state_after`
- `parameter_delta_sketch`
- `probe_delta`
- `activation_delta_sketch`
- `compute_cost`
- `seed_metadata`

## 4. Storage

A simple first version can use:

- Parquet for corpus and transition records;
- SQLite or DuckDB for indexes and experiment metadata;
- PyTorch checkpoints for simulator ensembles;
- JSON/YAML for experiment configurations.

Every simulated transition must record the simulator version. Every oracle observation must record target commit, configuration hash, checkpoint hash, and random seed.

## 5. Safety rails

- Never silently overwrite raw corpus metadata.
- Preserve source provenance through all clustering and materialization.
- Separate trusted correction data from ordinary corpus data.
- Treat negative interference as ambiguous until evaluated against trusted probes.
- Keep hidden evaluation probes inaccessible to the planner.
- Require real-model validation before executing long simulator-designed tranches.
