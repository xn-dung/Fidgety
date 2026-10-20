"""Verify notebook orchestration, validation selection, fusion, statistics and resume on CPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from paper_protocol import collect_results, mean_ci, metrics, signed_rank_exact, validate_data

root = Path(__file__).resolve().parent
notebook = json.loads((root / 'train_kaggle.ipynb').read_text(encoding='utf-8'))
assert all(cell['cell_type'] == 'code' for cell in notebook['cells'])
# Known confusion matrix, sample-CI and paired-fold statistics.
assert metrics([0, 0, 1, 1], [0.1, 0.9, 0.6, 0.2]) == dict(tn=1, fp=1, fn=1, tp=1, sensitivity=0.5, specificity=0.5, balanced_accuracy=0.5, accuracy=0.5)
assert signed_rank_exact(np.arange(9), np.arange(9)) == 1
assert signed_rank_exact(np.arange(9), np.arange(9) + 1) == 2 / 512
left = np.array([1., 4, 7, 11, 16, 22, 29, 37, 46])
assert signed_rank_exact(left, np.zeros(9)) == wilcoxon(left, method='exact').pvalue
mean, low, high = mean_ci([1., 2., 3.])
np.testing.assert_allclose([mean, low, high], [2, 2 - 1.96 / np.sqrt(3), 2 + 1.96 / np.sqrt(3)])

rng = np.random.default_rng(4)
with tempfile.TemporaryDirectory(dir=root) as temporary:
    temporary = Path(temporary)
    data_root = temporary / 'data'
    for fold in [1, 2]:
        for modality, frames, features in [('mat', 500, 6), ('imu', 300, 36), ('vitpose_kpc', 250, 30)]:
            folder = data_root / f'data_{modality}_9fold'
            folder.mkdir(parents=True, exist_ok=True)
            prefix = 'vitpose' if modality == 'vitpose_kpc' else modality
            listing = data_root / f'sample_list_{prefix}_9fold'
            listing.mkdir(exist_ok=True)
            for split, n in [('train', 16), ('test', 4)]:
                np.savetxt(folder / f'input_{prefix}_9fold_fold_{fold}_{split}.txt', rng.normal(size=(n, frames * features)), fmt='%.5f')
                np.savetxt(folder / f'output_{prefix}_9fold_fold_{fold}_{split}.txt', np.arange(n) % 2)
                names = []
                for i in range(n):
                    infant = 100 + i // 4 if split == 'train' else 200 + fold
                    frame = f'{i * 250:07d}'
                    end = f'{i * 250 + 249:07d}'
                    suffix = f'framestart-{frame}_frameend-{end}' if prefix == 'vitpose' else f'frames_{frame}-{end}_{prefix}_samples_0000001-0000300'
                    names.append(f'idxp{infant}_00028d_camera1_{suffix}_label_{i % 2}.txt')
                (listing / f'sample_list_{prefix}_9fold_fold_{fold}_{split}.txt').write_text('\n'.join(names))
        indices = data_root / 'random_indices_9fold'
        indices.mkdir(exist_ok=True)
        np.savetxt(indices / f'random_indices_train_9fold_fold_{fold}.txt', np.arange(1, 17), fmt='%d')
    scope = dict(Path=Path, subprocess=subprocess, sys=sys, SOURCE_DIR=root, REVISION='test')
    exec(''.join(notebook['cells'][0]['source']), scope)
    scope.update(MODE='smoke', DATA_ROOT=data_root, OUTPUT_DIR=temporary / 'output',
                 FOLDS=[1, 2], ARCHITECTURES=[1], RUNS=2, EPOCHS=1)
    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'get_device_name', return_value='test-cpu'):
        exec(''.join(notebook['cells'][2]['source']), scope)
    with patch.dict(os.environ, {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][3]['source']), scope)
    frame = pd.read_csv(scope['CSV_PATH'])
    assert len(frame[frame.row_type == 'fold']) == 16
    assert len(frame[frame.row_type == 'table2']) == 8
    assert len(frame[frame.row_type == 'table3']) == 28
    assert not frame.full_paper_schedule_complete.any()
    assert frame.all_requested_jobs_complete.all()
    assert len(list(scope['OUTPUT_DIR'].rglob('*.csv'))) == 1
    assert len(list(scope['OUTPUT_DIR'].rglob('*.pt'))) == 8
    for fold in [1, 2]:
        probabilities = []
        for modality in ['mat', 'imu', 'vid', 'fusion']:
            job = scope['OUTPUT_DIR'] / modality / 'architecture_01' / f'fold_{fold}'
            scores = np.loadtxt(next(job.glob('res_*.txt')), ndmin=2)
            with np.load(job / 'selected_predictions.npz') as selected:
                winner = int(selected['selected_run'])
                assert np.isclose(scores[winner - 1, 0], min(scores[:, 0]), atol=1e-6)
                if modality != 'fusion':
                    probabilities.append(selected['probabilities'])
                    labels = selected['labels']
        expected = metrics(labels, np.mean(probabilities, axis=0))
        actual = frame[(frame.row_type == 'fold') & (frame.condition == 'ALL 3-Nets') & (frame.fold == fold)].iloc[0]
        for key, value in expected.items():
            assert np.isclose(actual[key], value)
    # Resume completed jobs without retraining or changing predictions.
    mtimes = {p: p.stat().st_mtime_ns for p in scope['OUTPUT_DIR'].rglob('selected_predictions.npz')}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(''.join(notebook['cells'][3]['source']), scope)
    assert all(p.stat().st_mtime_ns == mtime for p, mtime in mtimes.items())
    with contextlib.redirect_stdout(io.StringIO()):
        exec(''.join(notebook['cells'][4]['source']), scope)
    import matplotlib
    matplotlib.use('Agg')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(''.join(notebook['cells'][5]['source']), scope)
    assert (scope['OUTPUT_DIR'] / 'paper_comparison.png').is_file()
    # Reorder one modality's sample list: preflight must refuse late/early fusion.
    sample_file = data_root / 'sample_list_imu_9fold/sample_list_imu_9fold_fold_1_test.txt'
    sample_file.write_text('\n'.join(reversed(sample_file.read_text().splitlines())))
    with patch.dict(os.environ, {'FIDGETY_DATA_ROOT': str(data_root)}):
        os.environ.pop('FIDGETY_DATA_ZIP', None)
        try:
            validate_data([1])
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned modalities were accepted.')
    print('Paper notebook OK: 4 networks, 2 folds, validation-selected restarts, 4 late-fusion combinations, CI/Wilcoxon, one CSV and resume.')
