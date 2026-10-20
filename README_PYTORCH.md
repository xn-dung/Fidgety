# PyTorch version

All 16 experiment scripts use PyTorch. Data paths, preprocessing, folds,
run counts, layer configurations, dropout, and output text formats retain
the original experiment structure.

Install dependencies with `python -m pip install -r requirements.txt`.
Run each script from this directory as before. CUDA is selected automatically
when available; otherwise training and inference use CPU.

Training writes `.pt` checkpoints containing layer configuration and a
`state_dict`. Evaluation searches the original results directories for these
new checkpoints. Existing Keras `.hf5` files cannot be loaded by this version;
train the PyTorch models and place their checkpoints in the corresponding
results directories before evaluation.

`torch_utils.py` contains the shared training, evaluation, early stopping,
and checkpoint routines. The scripts construct their networks using
`torch.nn.Sequential`. The original channels-last flatten order is retained.
Adam uses the original learning rate (0.001) and epsilon (1e-7); convolution
weights use orthogonal initialization and dense weights use Glorot uniform.
Batch normalization retains epsilon 0.001 and moving-average momentum 0.99
(expressed as 0.01 in PyTorch). The final training batch, if it contains a
single sample, uses stored statistics for dense batch normalization.
PyTorch and Keras have different batch-variance updates and random generators,
so runs are not numerically identical.

`torch_tuning.py` replaces Keras Tuner with Optuna. Fixed grids remain
exhaustive. Bayesian searches use Optuna TPE instead of Keras Tuner's Gaussian
process, so search trajectories differ. Studies resume from SQLite files in
the existing tuning output directories; Keras Tuner histories are not reused.

As in the author's training scripts, reported training/validation accuracy
thresholds raw logits at 0.5. Evaluation scripts apply sigmoid and round, as
in the original code. Checkpoints from best-model training restore the best
validation-loss epoch before saving.


## Kaggle: run all four models and export one CSV

Import `train_kaggle.ipynb` into Kaggle, attach the author's extracted data
folder and enable GPU and Internet. This notebook contains code cells only.
Its default data directory is `/kaggle/input/datasets/xndung/fidgety/data`.
Run all cells to train VID, IMU, MAT and their early fusion, sequentially.
The smoke configuration uses fold 1, one run and at most five epochs per
model. For the original training schedule set `FOLDS = list(range(1, 10))`,
`RUNS = 20`, `EPOCHS = 1000` in the first cell. This does not rerun tuning.

Each experiment gets a new timestamped directory under
`/kaggle/working/outputs`. One `results.csv` contains a row per model/fold/run;
it is updated after every completed run. Separate model/fold directories
contain logs, the original result text files and best-model checkpoints.
The last notebook cell displays all rows and mean metrics per model, with a
link to the CSV. The experiment configuration records the source commit and
seed. GPU operations are not guaranteed to be bitwise deterministic.

CSV test accuracy, sensitivity, specificity, balanced accuracy and confusion
counts use sigmoid probabilities, matching the author's evaluation scripts.
Columns named `val_accuracy_logits` and `test_accuracy_logits` preserve the
original training metric's threshold of 0.5 on raw logits. Summary means
weight each fold/run equally. Checkpoint selection uses validation loss.

Optional environment variables (defaults preserve the original scripts):

- `FIDGETY_DATA_ROOT`: directory containing `data_*_9fold` and random indices.
- `FIDGETY_DATA_ZIP`: read the same relative file paths directly from a ZIP
  instead; ZIP takes precedence if both data variables are set.
- `FIDGETY_FOLDS`: comma-separated fold numbers; default is all nine folds.
- `FIDGETY_RUNS`: best-model training repetitions; default `20`.
- `FIDGETY_EPOCHS`: maximum epochs; default is the script's original value.
- `FIDGETY_RESULTS_CSV`: optional CSV destination; append one row per run.
- `FIDGETY_MODEL`: model name written to CSV.
- `FIDGETY_SEED`: optional seed for Python, NumPy and PyTorch.

Without data environment variables, scripts use the original relative paths.
The notebook unsets the ZIP variable to read your Kaggle folder directly.
Arrays still occupy host RAM. Interrupted runs are not resumed; a new
notebook experiment starts fresh. Save or download Kaggle outputs before
ending the session. Evaluation scripts still expect checkpoints in their
original results directory layout; this notebook computes test metrics
directly during training and does not run those scripts.

Archives, datasets, checkpoints and generated results are excluded from Git.
