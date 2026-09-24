# Failure Debugging Sandbox

> Failure-Driven Robot Learning & Evaluation

## What this project does
A simulated pick-and-place robot arm (MuJoCo) that:
- Runs controlled failure experiments (low friction, pose error, action delay, etc.)
- Records structured telemetry for every episode
- Automatically finds **where** and **when** the failure first occurred
- Tests causal hypotheses via counterfactual re-runs

## Quick Start (Reproducible)
```bash
git clone <repo-url>
cd obliviq_failure_sandbox
conda create -n obliviq python=3.11 -y
conda activate obliviq
pip install -r requirements.txt
python verify_setup.py           # Check everything works
python experiments/run_sweep.py  # Run all experiments
```

## Project Structure
```
configs/          - YAML configs (nominal baseline + 6 perturbations)
simulation/       - MuJoCo scene + scripted controller
telemetry/        - Logger and data schema
evaluation/       - Task phases, first-divergence, metrics
experiments/      - Automated episode runner
data/episodes/    - Raw CSV telemetry per episode
plots/            - Generated figures
report/           - Written failure analysis
```

## Reproducibility
Every experiment records:
- Random seed (fixed per scenario)
- Git commit hash
- Simulator version (MuJoCo 3.x)
- Config file used
- Episode IDs

## Environment
- Python 3.11
- MuJoCo 3.x
- See `requirements.txt` for pinned versions
