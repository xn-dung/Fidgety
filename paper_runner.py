"""Execute published architectures with the author's original training scripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from paper_protocol import SCRIPTS, collect_results, validate_data


def run_experiment(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source = Path(__file__).resolve().parent
    output = config_path.parent
    catalog = json.loads((source / 'paper_architectures.json').read_text())['models']
    if config['mode'] not in ['paper', 'smoke']:
        raise ValueError('Choose mode paper or smoke.')
    if not config['folds'] or len(set(config['folds'])) != len(config['folds']) or any(f not in range(1, 10) for f in config['folds']):
        raise ValueError('Invalid folds.')
    if not config['architectures'] or len(set(config['architectures'])) != len(config['architectures']) or any(a not in range(1, 11) for a in config['architectures']) or 1 not in config['architectures']:
        raise ValueError('Choose distinct architecture IDs from 1 to 10, including 1.')
    if config['runs'] < 1 or config['epochs'] < 1:
        raise ValueError('Runs and epochs must be positive.')
    if config['mode'] == 'paper' and (config['folds'] != list(range(1, 10)) or config['architectures'] != list(range(1, 11)) or config['runs'] != 20 or config['epochs'] != 1000):
        raise ValueError('Paper mode requires all 9 folds, 10 architectures, 20 restarts and max 1000 epochs.')
    os.environ.pop('FIDGETY_DATA_ZIP', None)
    os.environ['FIDGETY_DATA_ROOT'] = config['data_root']
    details = validate_data(config['folds'], require_paper_subject_counts=config['mode'] == 'paper')
    print(details.to_string(index=False), flush=True)
    # Prevent silently resuming with changed data or implementation.
    signature = hashlib.sha256()
    for name in sorted(set(SCRIPTS.values()) | {'torch_utils.py', 'data_io.py', 'paper_protocol.py', 'paper_architectures.json', 'paper_runner.py'}):
        signature.update((source / name).read_bytes())
    for path in sorted(Path(config['data_root']).rglob('*.txt')):
        stat = path.stat()
        signature.update(str((path.relative_to(config['data_root']).as_posix(), stat.st_size, stat.st_mtime_ns)).encode())
    manifest = dict(config=config, source_and_data_metadata_signature=signature.hexdigest())
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Source or dataset metadata changed. Use a new experiment directory.')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = 4 * len(config['architectures']) * len(config['folds'])
    step = 0
    for fold in config['folds']:
        for modality_index, (modality, script) in enumerate(SCRIPTS.items()):
            for architecture in config['architectures']:
                step += 1
                job = output / modality / f'architecture_{architecture:02d}' / f'fold_{fold}'
                job.mkdir(parents=True, exist_ok=True)
                done = job / 'completed.json'
                selected_path = job / 'selected_predictions.npz'
                if done.exists():
                    if not selected_path.exists() or not list(job.glob('*.pt')):
                        raise FileNotFoundError('Completed job is missing predictions or checkpoint: ' + str(job))
                    print(f'[{step}/{total}] Skipping completed {modality} #{architecture}, fold {fold}', flush=True)
                    continue
                # An interrupted job restarts all its random initializations.
                selected_path.unlink(missing_ok=True)
                env = os.environ.copy()
                for key in ['FIDGETY_RESULTS_CSV', 'FIDGETY_DATA_ZIP']:
                    env.pop(key, None)
                env.update(FIDGETY_DATA_ROOT=config['data_root'], FIDGETY_FOLDS=str(fold),
                           FIDGETY_RUNS=str(config['runs']), FIDGETY_EPOCHS=str(config['epochs']),
                           FIDGETY_ARCHITECTURE=json.dumps(catalog[modality][str(architecture)]),
                           FIDGETY_SELECTED_PREDICTIONS=str(selected_path),
                           FIDGETY_SEED=str(config['seed'] + modality_index * 100000 + architecture * 1000 + fold),
                           PYTHONUNBUFFERED='1')
                print(f'[{step}/{total}] Training {modality} #{architecture}, fold {fold}, restarts={config["runs"]}', flush=True)
                start = time.monotonic()
                with (job / 'train.log').open('w', encoding='utf-8') as log:
                    with subprocess.Popen([sys.executable, '-u', str(source / script)], cwd=job, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                        try:
                            for line in child.stdout:
                                if line.startswith(('Fold:', 'best val_loss')):
                                    print(line, end='', flush=True)
                                log.write(line)
                                log.flush()
                            returncode = child.wait()
                        except BaseException:
                            child.terminate()
                            child.wait()
                            raise
                if returncode:
                    raise RuntimeError(f'Training failed: {job}; exit={returncode}')
                if not selected_path.exists() or not list(job.glob('*.pt')):
                    raise RuntimeError('No selected checkpoint/predictions: ' + str(job))
                # Check the saved winner against all restart validation scores.
                import numpy as np
                result_files = list(job.glob('res_*.txt'))
                if len(result_files) != 1:
                    raise RuntimeError('Expected one restart score file.')
                scores = np.loadtxt(result_files[0], ndmin=2)
                with np.load(selected_path, allow_pickle=False) as selected:
                    winning_run = int(selected['selected_run'])
                    if scores.shape != (config['runs'], 3) or not np.isfinite(scores).all():
                        raise RuntimeError('Invalid restart scores.')
                    if winning_run != int(np.argmin(scores[:, 0])) + 1:
                        # Original TXT uses six decimals; allow ties introduced by rounding.
                        if not np.isclose(scores[winning_run - 1, 0], scores[:, 0].min(), atol=1e-6, rtol=0):
                            raise RuntimeError('Selected checkpoint is not the validation-loss winner.')
                    if not np.isclose(float(selected['best_val_loss']), scores[:, 0].min(), atol=1e-6, rtol=0):
                        raise RuntimeError('Selected checkpoint validation loss disagrees with restart scores.')
                done.write_text(json.dumps(dict(modality=modality, architecture=architecture, fold=fold,
                                               completed_restarts=config['runs'], seconds=time.monotonic() - start), indent=2))
                collect_results(output, config)
                print(f'Completed {modality} #{architecture}, fold {fold}; seconds={time.monotonic() - start:.1f}', flush=True)
    results = collect_results(output, config)
    print('Results CSV:', output / 'results.csv', flush=True)
    print(results[results['row_type'] == 'table2'].to_string(index=False), flush=True)
    return results


if __name__ == '__main__':
    def stop_requested(signum, frame):
        raise KeyboardInterrupt('Training interrupted.')

    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run_experiment(parser.parse_args().config)
