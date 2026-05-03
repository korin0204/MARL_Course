# Project Structure and Config Reference

This document describes the repository layout, module responsibilities, and the JSON config knobs used by the training and evaluation scripts.

## Top-level layout

- pyproject.toml: Packaging, dependencies, and pytest configuration.
- README.md: Quick start commands and entry points.
- configs/: JSON configs for training/evaluation. CLI flags override config values.
- docs/: Assignments, grading guide, and teaching notes.
- notebooks/: Colab notebooks for student-facing tutorials.
- outputs/: Training artifacts (policy.pt, policy.py, metrics.csv, effective_config.json, gifs).
- scripts/: CLI entry points for training, evaluation, visualization, and smoke tests.
- src/: Python package (marl_course) with environments, algorithms, evaluation, and visualization.
- submissions/: Example layout and instructions for student submissions.
- tests/: Lightweight regression tests for envs and evaluators.
- wandb/: Optional Weights & Biases run data (when enabled).

## Package layout (src/marl_course)

- common/: Small shared utilities.
  - api.py: Minimal PettingZoo-style Parallel API protocol.
  - config.py: JSON config loader and CLI override helper.
  - grid.py: Grid utilities (paths, movement, distance).
  - rewards.py: Event schema and reward hook base classes.
  - submission.py: Loader for student submissions (policy.py + policy.pt + metadata.json).
- envs/: Game environments.
  - bomber_arena/: Competitive 4-player Bomber environment and baseline policies.
  - coop_kitchen/: Cooperative 4-player kitchen environment, layouts, and baseline policies.
- algos/: Educational baselines and helpers.
  - bomber_qlearning.py: Tabular Q-learning baseline for Bomber.
  - bomber_torch.py: Torch DQN and actor-critic models plus loaders.
  - coop_bandit.py: Simple strategy-bandit baseline for Coop.
  - logging.py: CSV logger with optional W&B mirroring.
  - artifacts.py: GIF recording helpers.
  - reward_examples.py: Example shaped rewards (not used for grading).
  - student_templates.py: Minimal baseline policies for teaching.
- evaluation/: Teacher-facing evaluation loops for Bomber and Coop.
- visualization/: ASCII, GIF, and pygame rendering helpers.

## Script summary (scripts/)

- train_bomber.py: Tabular Q-learning training for Bomber Arena.
- train_bomber_dqn.py: DQN training for Bomber Arena (torch).
- train_bomber_actor_critic.py: Actor-critic training for Bomber Arena (torch).
- train_coop_mappo.py: MAPPO-lite style strategy-bandit training for Coop Kitchen.
- train_coop_qmix.py: QMIX-lite style strategy-bandit training for Coop Kitchen.
- train_coop_zero_shot_league.py: Strategy-bandit training over generated layouts.
- evaluate_bomber_tournament.py: Round-robin tournament evaluation for Bomber submissions.
- evaluate_coop_submissions.py: Cooperative evaluation for fixed layouts.
- evaluate_coop_zero_shot.py: Cooperative evaluation across seen/unseen/heldout layouts.
- visualize_episode.py: Quick local viewer for Bomber or Coop.
- smoke_check.py: Minimal end-to-end sanity check.

## Config reference (configs/*.json)

All scripts accept --config and use the following rule for each key:

1) CLI flag value
2) JSON config value
3) Script default

Each training run writes outputs/.../effective_config.json with the final resolved values.

### Common training keys

- student_id: Submission name written into metadata.json.
- out: Output directory path for artifacts.
- episodes: Number of training episodes.
- seed: Random seed.
- wandb: Enable Weights & Biases logging.
- wandb_project: W&B project name.
- gif_steps: Max steps to render for post_training_episode.gif.
- gif_fps: GIF frame rate.
- gif_tile_size: Pixel size per tile in GIF output.

### Bomber Q-learning (train_bomber.json)

- alpha: Q-learning step size.
- gamma: Discount factor.
- epsilon_start / epsilon_end: Epsilon-greedy schedule for exploration.

### Bomber DQN (train_bomber_dqn.json)

- gamma: Discount factor.
- lr: Learning rate for Adam.
- batch_size: Replay batch size.
- replay_size: Replay buffer capacity.
- warmup_steps: Steps before training starts.
- train_every: Gradient update frequency in steps.
- target_update: Target network sync interval.
- epsilon_start / epsilon_end: Epsilon-greedy schedule.
- hidden_dim: MLP hidden size.
- device: Torch device string (e.g., cpu, cuda).

### Bomber Actor-Critic (train_bomber_actor_critic.json)

- gamma: Discount factor.
- lr: Learning rate for Adam.
- value_coef: Critic loss weight.
- entropy_coef: Entropy bonus weight.
- epsilon_start / epsilon_end: Exploration smoothing.
- hidden_dim: MLP hidden size.
- device: Torch device string (e.g., cpu, cuda).

### Coop MAPPO-lite (train_coop_mappo.json)

- alpha: Bandit update step size.
- epsilon_start / epsilon_end: Strategy exploration schedule.

### Coop QMIX-lite (train_coop_qmix.json)

- alpha: Bandit update step size.
- epsilon_start / epsilon_end: Strategy exploration schedule.

### Coop Zero-Shot League (train_coop_zero_shot.json)

- alpha: Bandit update step size.
- epsilon_start / epsilon_end: Strategy exploration schedule.
- families: List of layout families used for generated layouts.

### Evaluation configs (evaluate_*.json)

- submissions: Path to submissions folder (optional, defaults to built-in baselines).
- episodes: Episodes per matchup/layout.
- seed: Random seed.
- live: Enable pygame viewer.
- live_ascii: Enable ASCII live viewer.
- live_sleep: Delay between frames for live viewers.
- live_tile_size: Tile size for pygame viewer.
