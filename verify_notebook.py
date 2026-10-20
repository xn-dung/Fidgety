"""Exercise the code-only notebook on small synthetic filesystem data (CPU)."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

root = Path(__file__).resolve().parent
notebook = json.loads((root / 'train_kaggle.ipynb').read_text(encoding='utf-8'))
assert all(cell['cell_type'] == 'code' for cell in notebook['cells'])
rng = np.random.default_rng(4)
with tempfile.TemporaryDirectory(dir=root) as temporary:
    temporary = Path(temporary)
    data_root = temporary / 'data'
    for modality, frames, features in [('mat', 500, 6), ('imu', 300, 36), ('vitpose_kpc', 250, 30)]:
        folder = data_root / f'data_{modality}_9fold'
        folder.mkdir(parents=True)
        prefix = 'vitpose' if modality == 'vitpose_kpc' else modality
        for split, n in [('train', 16), ('test', 4)]:
            np.savetxt(folder / f'input_{prefix}_9fold_fold_1_{split}.txt',
                       rng.normal(size=(n, frames * features)), fmt='%.5f')
            np.savetxt(folder / f'output_{prefix}_9fold_fold_1_{split}.txt', np.arange(n) % 2)
    indices = data_root / 'random_indices_9fold'
    indices.mkdir()
    np.savetxt(indices / 'random_indices_train_9fold_fold_1.txt', np.arange(1, 17), fmt='%d')
    scope = dict(Path=Path, subprocess=subprocess, sys=sys, SOURCE_DIR=root, REVISION='test',
                 TEST_OUTPUT_BASE=temporary / 'outputs')
    exec(''.join(notebook['cells'][0]['source']), scope)
    scope.update(DATA_ROOT=data_root, EPOCHS=1)
    config_cell = ''.join(notebook['cells'][2]['source']).replace(
        'Path("/kaggle/working/outputs")', 'TEST_OUTPUT_BASE')
    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'get_device_name', return_value='test-cpu'):
        exec(config_cell, scope)
    with patch.dict(os.environ, {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][3]['source']), scope)
    with scope['CSV_PATH'].open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row['model'] for row in rows} == {'vid', 'imu', 'mat', 'fusion'}
    for row in rows:
        assert int(row['epochs_trained']) == 1
        tn, fp, fn, tp = [int(row[key]) for key in ['tn', 'fp', 'fn', 'tp']]
        assert tn + fp + fn + tp == 4
        assert float(row['test_accuracy']) == (tn + tp) / 4
        assert float(row['balanced_accuracy']) == (tp / (tp + fn) + tn / (tn + fp)) / 2
    assert len(list(scope['OUTPUT_DIR'].rglob('*.csv'))) == 1
    assert len(list(scope['OUTPUT_DIR'].rglob('*.pt'))) == 4
    with contextlib.redirect_stdout(io.StringIO()):
        exec(''.join(notebook['cells'][4]['source']), scope)
    print('All four models trained from a directory; one CSV, metrics, checkpoints and display OK.')
