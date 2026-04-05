# DPP-DRL for NOMA AoI Optimization

This repository contains a Deep Reinforcement Learning workflow for minimizing Age of Information (AoI) in a NOMA system with battery-constrained users.

## Project Structure

- `main.py`: Main training/evaluation script using `MemoryDNN`.
- `memory.py`: Replay-memory-based DNN model for action generation.
- `bisection.py`: Objective and state-transition evaluation for each action.
- `DPP_exhaustive.py`: Exhaustive baseline for small-user settings.
- `data/`: MATLAB `.mat` input files (for example `data_10.mat`, `data_3.mat`).

## Environment

Recommended Python version: 3.8+

Core dependencies used by this project:

- `numpy`
- `scipy`
- `torch`
- `matplotlib`

You can install dependencies from the provided environment snapshot:

```bash
pip install -r requirements.txt
```

## Data Requirements

The scripts expect MATLAB files under `./data/` with keys:

- `input_h`
- `input_g`

Expected files for default runs:

- `./data/data_10.mat` (used by `main.py`)
- `./data/data_3.mat` (used by `DPP_exhaustive.py`)

## Run

Train and evaluate the DPP-DRL workflow:

```bash
python main.py
```

Run exhaustive baseline:

```bash
python DPP_exhaustive.py
```

## Notes

- `main.py` uses large loop counts by default (`n = 30000`, repeated runs), so training can take a long time.
- `memory.py` prints the local PyTorch version on import to match current project behavior.
