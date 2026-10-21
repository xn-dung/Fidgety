# Fidgety - PyTorch

PyTorch port of the supplied experiment scripts for *Deep learning empowered
sensor fusion boosts infant movement classification*.

- [Kaggle notebook](train_kaggle.ipynb): run a fast published-model baseline or
  the full 40-architecture protocol, calculate early/late fusion and export one CSV.
- Default fast baseline: 9 folds x published architecture #1 x 1 run x 4 network families = 36 runs.
- Optional paper mode: 9 folds x 10 architectures x 20 restarts x 4 network families.
- Use smoke mode for a short pipeline check.
- [Protocol, differences and setup](README_PYTORCH.md).

The main table compares published architecture #1 for each modality. All ten
architectures are also reported individually. The notebook does not repeat
hyperparameter search or guarantee the original TensorFlow numerical results.

Upload data to Kaggle separately; this repository contains source only.
