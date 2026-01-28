# MetaKAN reproduction workspace

This folder contains small scripts to reproduce MetaKAN baselines and parameter sweeps for:
- image classification
- function fitting
- PDE solving

## File layout
- `metakan_repro/run_image_experiments.py`
- `metakan_repro/plot_results.py`
- `metakan_repro/run_function_fitting.py`
- `metakan_repro/plot_function_fitting.py`
- `metakan_repro/run_pde_experiments.py`
- `metakan_repro/plot_pde.py`
- `metakan_repro/out/` (run logs + plots)

## Image classification
### Compared
- MLP (train.py)
- KAN (train.py)
- MetaKAN (train_meta.py)
- MetaKAN hidden_dim sweep (train_meta.py)

### Datasets available (MetaKAN/image_classification)
- MNIST
- EMNIST-Letters
- EMNIST-Balanced
- FMNIST
- KMNIST
- Cifar10
- Cifar100
- SVHN

### Models available (MetaKAN/image_classification)
- MLP
- KAN
- MetaKAN
- FastKAN
- WavKAN
- MetaFastKAN
- MetaWavKAN

### Usage
Dry-run:
```bash
python metakan_repro/run_image_experiments.py --dataset MNIST
```

Run:
```bash
python metakan_repro/run_image_experiments.py --dataset MNIST --run
```

Plot:
```bash
python metakan_repro/plot_results.py
```

Outputs (in `metakan_repro/out/`):
- `runs.jsonl`
- `baseline_compare.png`
- `metakan_hidden_dim_sweep.png`

## Function fitting
### Compared
- MLP (train.py)
- KAN (train.py)
- HyperKAN (train_hyper.py)
- HyperKAN hidden_dim sweep (train_hyper.py)

### Datasets available (MetaKAN/function_fitting)
- MNIST
- Feynman symbolic datasets (e.g., `I.6.20b`, `I.12.4`, `II.11.3`, `III.7.38`, ...)

### Models available (MetaKAN/function_fitting)
- MLP
- KAN
- HyperKAN

### Usage
Dry-run:
```bash
python metakan_repro/run_function_fitting.py --dataset I.6.20b
```

Run:
```bash
python metakan_repro/run_function_fitting.py --dataset I.6.20b --run
```

Plot:
```bash
python metakan_repro/plot_function_fitting.py
```

Outputs (in `metakan_repro/out/`):
- `runs_function_fitting.jsonl`
- `ff_baseline_compare.png`
- `ff_hyperkan_hidden_dim_sweep.png`

## PDE solving
### Compared
- MLP
- KAN
- MetaKAN
- MetaKAN hidden_dim sweep

### Datasets available (MetaKAN/solving_pde)
- Poisson
- Allen_Cahn

### Usage
Dry-run:
```bash
python metakan_repro/run_pde_experiments.py --dataset Poisson
```

Run:
```bash
python metakan_repro/run_pde_experiments.py --dataset Poisson --run
```

Plot:
```bash
python metakan_repro/plot_pde.py --dataset Poisson --runs metakan_repro/out/runs_pde_Poisson.jsonl
```

Outputs (in `metakan_repro/out/`):
- `runs_pde_Poisson.jsonl` or `runs_pde_Allen_Cahn.jsonl`
- `pde_<dataset>_baseline.png`
- `pde_<dataset>_metakan_hidden_dim_sweep.png`

## Notes
- All runners default to dry-run and only execute when `--run` is passed.
- These scripts call MetaKAN training scripts and parse their outputs:
  - image classification: `MetaKAN/image_classification/results/results.csv`
  - function fitting: `MetaKAN/results/results.csv`
  - PDE: `MetaKAN/solving_pde/saved_loss_l2/*.xlsx`
- Sweeps track hidden_dim in the run logs so plotting stays consistent.
- Use `--help` on any script to see tunable parameters.
