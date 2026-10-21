"""Published architectures, subject-separated folds and paper-style statistics."""
import itertools
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from data_io import loadtxt

MODALITIES = {'mat': 'mat', 'imu': 'imu', 'vid': 'vitpose_kpc'}
CONDITIONS = ['MAT', 'IMU', 'VID', 'MAT+IMU', 'MAT+VID', 'IMU+VID', 'ALL 3-Nets', 'ALL 1-Net']
SCRIPTS = {'mat': 'mat_conv_best_model_9fold.py', 'imu': 'imu_acc_gyr_conv_best_model_9fold.py',
           'vid': 'vitpose_kpc_vel_conv_best_model_9fold.py', 'fusion': 'mat_vitpose_imu_conv_best_model_9fold.py'}


def sample_ids(modality, fold, split):
    prefix = 'vitpose' if modality == 'vid' else modality
    path = f'../data/sample_list_{prefix}_9fold/sample_list_{prefix}_9fold_fold_{fold}_{split}.txt'
    names = loadtxt(path, dtype=str, ndmin=1)
    keys, labels = [], []
    pattern = r'^(idxp\d+)_(\d+d)_camera(\d+)_(?:frames_(\d+)-(\d+)|framestart-(\d+)_frameend-(\d+))'
    for name in names:
        match = re.match(pattern, str(name))
        label = re.search(r'_label_([01])\.txt$', str(name))
        if not match or not label:
            raise ValueError('Unrecognized sample name: ' + str(name))
        infant, age, camera, a, b, c, d = match.groups()
        keys.append('|'.join([infant, age, camera, a or c, b or d]))
        labels.append(int(label[1]))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate sample IDs: ' + path)
    return np.asarray(keys), np.asarray(labels)


def validate_data(folds, require_paper_subject_counts=False):
    details, test_infants = [], set()
    for fold in folds:
        canonical = {}
        for split in ['train', 'test']:
            reference_ids, reference_labels = None, None
            for model, modality in MODALITIES.items():
                prefix = 'vitpose' if model == 'vid' else model
                ids, file_labels = sample_ids(model, fold, split)
                labels = loadtxt(f'../data/data_{modality}_9fold/output_{prefix}_9fold_fold_{fold}_{split}.txt', ndmin=1)
                if not np.array_equal(labels, file_labels):
                    raise ValueError(f'Sample-list labels disagree: {model}, fold {fold}, {split}')
                if reference_ids is not None and (not np.array_equal(ids, reference_ids) or not np.array_equal(labels, reference_labels)):
                    raise ValueError(f'Modalities are not aligned: {model}, fold {fold}, {split}')
                reference_ids, reference_labels = ids, labels
                data_path = f'data_{modality}_9fold/input_{prefix}_9fold_fold_{fold}_{split}.txt'
                if os.environ.get('FIDGETY_DATA_ROOT') and not (Path(os.environ['FIDGETY_DATA_ROOT']) / data_path).is_file():
                    raise FileNotFoundError(data_path)
            canonical[split] = set(i.split('|')[0] for i in reference_ids)
            details.append(dict(fold=fold, split=split, samples=len(reference_ids),
                                infants=len(canonical[split]), positive=int(np.sum(reference_labels == 1)),
                                negative=int(np.sum(reference_labels == 0))))
        if canonical['train'] & canonical['test']:
            raise ValueError(f'Infant leakage between train and test, fold {fold}')
        if test_infants & canonical['test']:
            raise ValueError('An infant occurs in more than one test fold.')
        test_infants |= canonical['test']
        if require_paper_subject_counts and (len(canonical['train']) != 32 or len(canonical['test']) != 4):
            raise ValueError('Paper protocol expects 32 train infants and 4 test infants per fold.')
        ind = loadtxt(f'../data/random_indices_9fold/random_indices_train_9fold_fold_{fold}.txt', dtype=int, ndmin=1)
        n = details[-2]['samples']
        if not np.array_equal(np.sort(ind), np.arange(1, n + 1)):
            raise ValueError(f'Invalid random permutation for fold {fold}')
    return pd.DataFrame(details)


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid binary predictions.')
    # np.round follows the supplied evaluation scripts (including exact ties).
    predictions = np.round(probabilities).astype(int)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else float('nan')
    specificity = tn / (tn + fp) if tn + fp else float('nan')
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2, accuracy=(tn + tp) / len(labels))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Cannot summarize undefined metrics.')
    mean = float(np.mean(values))
    # Matches class_results/plot_results.m: 1.96 * sample std / sqrt(n).
    margin = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float('nan')
    return mean, mean - margin, mean + margin


def signed_rank_exact(a, b):
    difference = np.round(np.asarray(a) - np.asarray(b), 12)
    difference = difference[difference != 0]
    if len(difference) == 0:
        return 1.0
    if len(difference) > 20:
        raise ValueError('Exact enumeration is intended for the nine paper folds.')
    ranks = rankdata(np.abs(difference), method='average')
    observed = abs(float(np.sum(np.sign(difference) * ranks)))
    extreme = sum(abs(float(np.dot(signs, ranks))) >= observed - 1e-10
                  for signs in itertools.product([-1, 1], repeat=len(ranks)))
    return extreme / 2 ** len(ranks)


def collect_results(output_dir, config):
    output_dir = Path(output_dir)
    fold_rows, predictions = [], {}
    for modality in SCRIPTS:
        for architecture in config['architectures']:
            for fold in config['folds']:
                job = output_dir / modality / f'architecture_{architecture:02d}' / f'fold_{fold}'
                if not (job / 'completed.json').exists():
                    continue
                with np.load(job / 'selected_predictions.npz', allow_pickle=False) as selected:
                    probabilities = selected['probabilities'].copy()
                    labels = selected['labels'].copy()
                    # Independently check the saved probability/logit transformation.
                    np.testing.assert_allclose(probabilities, expit(selected['logits'].astype(float)), atol=1e-12)
                    row = dict(row_type='fold', condition={'mat': 'MAT', 'imu': 'IMU', 'vid': 'VID', 'fusion': 'ALL 1-Net'}[modality],
                               architecture=architecture, fold=fold, selected_run=int(selected['selected_run']),
                               best_val_loss=float(selected['best_val_loss']), epochs_trained=int(selected['epochs_trained']),
                               selection='minimum_validation_loss_over_restarts', **metrics(labels, probabilities))
                fold_rows.append(row)
                if architecture == 1 and modality != 'fusion':
                    predictions[(modality, fold)] = (labels, probabilities)
    for fold in config['folds']:
        for modalities in [('mat', 'imu'), ('mat', 'vid'), ('imu', 'vid'), ('mat', 'imu', 'vid')]:
            if not all((m, fold) in predictions for m in modalities):
                continue
            labels = predictions[(modalities[0], fold)][0]
            for modality in modalities[1:]:
                if not np.array_equal(labels, predictions[(modality, fold)][0]):
                    raise ValueError('Labels are not aligned for late fusion.')
            probability = np.mean([predictions[(m, fold)][1] for m in modalities], axis=0)
            name = 'ALL 3-Nets' if len(modalities) == 3 else '+'.join(m.upper() for m in modalities)
            fold_rows.append(dict(row_type='fold', condition=name, architecture=1, fold=fold,
                                  selection='mean_probability_of_published_architecture_1', **metrics(labels, probability)))
    rows = list(fold_rows)
    if not rows:
        return pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    for (condition, architecture), group in folds.groupby(['condition', 'architecture'], sort=False):
        row = dict(row_type='architecture_summary', condition=condition, architecture=int(architecture), n_folds=len(group))
        for metric in ['sensitivity', 'specificity', 'balanced_accuracy', 'accuracy']:
            mean, low, high = mean_ci(group[metric].to_numpy() * 100)
            row.update({metric + '_mean_pct': mean, metric + '_ci_low_pct': low, metric + '_ci_high_pct': high})
        rows.append(row)
        if architecture == 1:
            rows.append(dict(row, row_type='table2', selection='published_architecture_1_fixed_before_retraining'))
    # Pair only the same fold IDs; never pair unmatched or flattened run rows.
    main = folds[folds['architecture'] == 1]
    for a, b in itertools.combinations(CONDITIONS, 2):
        left = main[main['condition'] == a].set_index('fold')['balanced_accuracy']
        right = main[main['condition'] == b].set_index('fold')['balanced_accuracy']
        common = sorted(set(left.index) & set(right.index))
        if len(common) < 2:
            continue
        pvalue = signed_rank_exact(left.loc[common].to_numpy(), right.loc[common].to_numpy())
        rows.append(dict(row_type='table3', condition_a=a, condition_b=b, n_folds=len(common),
                         p_value=pvalue, significant_005=pvalue < 0.05, correction='none',
                         selection='published_architecture_1_fixed_before_retraining'))
    frame = pd.DataFrame(rows)
    reference_path = Path(__file__).with_name('paper_reference.json')
    reference = json.loads(reference_path.read_text())['conditions']
    for index in frame.index[frame['row_type'] == 'table2']:
        condition = frame.at[index, 'condition']
        for metric in ['sensitivity', 'specificity', 'balanced_accuracy']:
            published = reference[condition][metric]['mean_pct']
            frame.at[index, metric + '_paper_pct'] = published
            frame.at[index, metric + '_delta_pp'] = frame.at[index, metric + '_mean_pct'] - published
    expected_jobs = 4 * len(config['architectures']) * len(config['folds'])
    completed_jobs = sum((output_dir / m / f'architecture_{a:02d}' / f'fold_{f}' / 'completed.json').exists()
                         for m in SCRIPTS for a in config['architectures'] for f in config['folds'])
    frame['completed_training_jobs'] = completed_jobs
    frame['expected_training_jobs'] = expected_jobs
    frame['protocol'] = config['mode']
    frame['all_requested_jobs_complete'] = completed_jobs == expected_jobs
    frame['reduced_baseline_complete'] = (config['mode'] == 'baseline' and config['folds'] == list(range(1, 10))
                                         and config['architectures'] == [1] and config['epochs'] == 1000
                                         and completed_jobs == expected_jobs)
    frame['training_restarts_per_fold'] = config['runs']
    frame['full_paper_schedule_complete'] = (config['mode'] == 'paper' and config['folds'] == list(range(1, 10))
                                             and config['architectures'] == list(range(1, 11))
                                             and config['runs'] == 20 and config['epochs'] == 1000
                                             and completed_jobs == expected_jobs)
    temporary = output_dir / 'results.tmp'
    frame.to_csv(temporary, index=False)
    temporary.replace(output_dir / 'results.csv')
    return frame
