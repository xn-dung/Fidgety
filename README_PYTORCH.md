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


## Kaggle: published-architecture retraining and paper comparison

`train_kaggle.ipynb` contains only executable code cells. Add the author's
data directory at `/kaggle/input/datasets/xndung/fidgety/data`, enable GPU
and Internet, and run the cells in order. Default `MODE = "paper"` trains:

- MAT, IMU, VID and ALL 1-Net (early fusion).
- All ten published architectures for each network family.
- Nine subject-separated folds, twenty random restarts per architecture/fold.
- Batch size four, validation split 1/8, patience ten, maximum 1000 epochs.

This is **360 training jobs / 7200 random restarts**. Use `MODE = "smoke"`
to test one architecture, one fold, two restarts and five epochs before
launching the full schedule. Smoke/partial tables are labeled accordingly.

`paper_architectures.json` transcribes all forty configurations directly
from Supplementary Tables 2 and 3. The notebook retrains those fixed published
architectures; it does **not** rerun Bayesian optimization or grid search.
The separate legacy tuning scripts use Optuna TPE and are not the original
KerasTuner Gaussian-process search.

The original four training scripts still perform preprocessing and model
construction. Optional parameter overrides select the published architecture.
For each architecture/fold, the lowest-validation-loss random restart is
selected, with its best-validation epoch restored. Only this checkpoint's
probabilities enter reported metrics. Restarts are never averaged.

Before training, the runner checks sample identity/order and labels across
modalities, random-index permutations, disjoint train/test infant IDs and
non-overlapping test-fold infant IDs. Full paper mode requires 32 train and
four test infants per fold. It uses the supplied folds without a new random
split. Original normalization and Fortran reshape behavior are retained.

Late fusion averages probabilities from selected independent MAT/IMU/VID
networks for MAT+IMU, MAT+VID, IMU+VID and ALL 3-Nets. It does not average
logits or concatenate features; the latter is the separate ALL 1-Net model.

One `results.csv` includes these `row_type` values:

- `fold`: metrics for every architecture/fold and late-fusion condition.
- `architecture_summary`: Supplementary Tables 4/5-style summaries for each
  published architecture, using one selected restart per fold.
- `table2`: main comparison using **published architecture #1**, the model
  reported as best in each original architecture table, fixed before retraining.
  No new architecture is selected from the new test scores. These are therefore
  comparisons of published models, not a new best-of-ten test ranking.
- `table3`: exact paired two-sided Wilcoxon signed-rank comparisons of those
  main models across matching folds, with no multiple-comparison correction.

Sensitivity, specificity and balanced accuracy are reported with the author's
normal-approximation mean CI: `mean +/- 1.96 * sample_std / sqrt(n)`.
This matches `class_results/plot_results.m`; a Student t interval is not used.
Wilcoxon probabilities are calculated by exact sign enumeration, with zero
differences removed and tied absolute differences assigned average ranks.
Floating differences are rounded to 12 decimals before identifying ties.
The CSV retains all fold observations, so statistical choices can be audited.

`paper_reference.json` contains the authors' original aggregate results from
the small `class_results.zip` package, for comparison only. New metrics are
computed from your newly trained predictions. Table rows include reference
values and differences in percentage points. The notebook also plots
`paper_comparison.png`. New values are not replaced by reference numbers.

Results, logs, selected predictions and checkpoints are written under
`OUTPUT_DIR`. Completed architecture/fold jobs are skipped when resuming the
same configuration, source and dataset metadata. Interrupted jobs restart all
their random restarts; epoch/optimizer resume is not provided. You must save
or download Kaggle outputs to preserve them across sessions; copying the
whole experiment directory back to the same OUTPUT_DIR allows job resume.
A changed source commit/configuration requires a new experiment directory.

The scripts remain a PyTorch port, not a claim of bitwise reproduction of
TensorFlow 2.10. Batch-normalization running variance, single-sample dense
BatchNorm handling, RNGs and GPU operations may produce different results.
The notebook checks the published architecture and evaluation protocol; only
a full actual-data run can assess numerical agreement with the paper.

Sources:

- Article: https://doi.org/10.1038/s43856-024-00701-w
- Published architectures: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43856-024-00701-w/MediaObjects/43856_2024_701_MOESM2_ESM.pdf
- Reference fold results / MATLAB statistics: https://zenodo.org/records/14046527/files/class_results.zip?download=1

Without environment overrides, the scripts retain the original relative
filesystem paths and default architecture. `FIDGETY_DATA_ROOT` supports a
mounted directory; `FIDGETY_DATA_ZIP` supports streaming from ZIP and takes
precedence. The notebook uses a directory. Arrays still occupy system RAM.
Archives, datasets, checkpoints and generated results are excluded from Git.


## Multiple Kaggle GPUs and runtime

The runner starts one independent training subprocess per visible GPU by
default (`GPU_WORKERS = 0`). With two GPUs it dynamically assigns independent
architecture/fold jobs to both, pinning each subprocess through
`CUDA_VISIBLE_DEVICES`. Batch size four, early stopping and checkpoint
selection stay unchanged. One coordinator writes the shared CSV; job markers
are published atomically. On an error or interrupt it terminates active
training subprocesses. Set `GPU_WORKERS = 1` if host RAM limits parallel runs.
`CPU_THREADS_PER_WORKER = 2` limits CPU oversubscription. Dataset loading and
system RAM are shared, so two GPUs do not guarantee twice the throughput.

Use `MODE = "benchmark"` for four published architecture-1 jobs, one fold,
two restarts each, and the original 1000-epoch cap / patience ten. Once all
four network families have a completed timing sample, the runner prints an
approximate full-schedule runtime and remaining ETA from observed seconds
per restart. The full schedule has 1800 restarts per network family. These
are estimates: architectures, stopping epochs, GPU speed and I/O contention
vary. Smoke mode's five-epoch cap is explicitly excluded from extrapolating
to the full paper schedule. Benchmark tables are partial, not paper results.
