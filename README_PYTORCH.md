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


## Kaggle: train directly from data.zip

Import `train_kaggle.ipynb` into Kaggle, attach your Dataset containing
`data.zip`, enable GPU and Internet, then run the notebook cells in order.
The default smoke run uses VID, fold 1, one run and five epochs. Change the
configuration cell for a full experiment. Results and logs are saved under
`/kaggle/working/outputs/<experiment>/fold_<n>`.

No archive extraction is needed. `data_io.loadtxt` reads each requested text
file from `FIDGETY_DATA_ZIP` as a stream; without that environment variable,
it reads the original filesystem paths. Arrays still occupy host RAM.

Optional environment variables (defaults preserve the original scripts):

- `FIDGETY_DATA_ZIP`: path to the author's ZIP archive.
- `FIDGETY_FOLDS`: comma-separated fold numbers, default `1,2,3,4,5,6,7,8,9`.
- `FIDGETY_RUNS`: number of best-model training runs, default `20`.
- `FIDGETY_EPOCHS`: maximum epochs, default is the script's original value.

The notebook skips folds marked complete with the same configuration and
source commit. Interrupted folds restart from their first run; checkpoints
do not include optimizer state. Evaluation scripts still expect checkpoints
in their original results directory layout; the training notebook does not
run those evaluation scripts automatically.

The repository excludes archives, datasets, checkpoints and generated results
through `.gitignore`. Upload `data.zip` to Kaggle separately.
