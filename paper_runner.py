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
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from paper_protocol import SCRIPTS, collect_results, validate_data



def execute_job(item, device, config, source, output, catalog, stop_event, registry, registry_lock):
    modality, modality_index, architecture, fold = item
    script = SCRIPTS[modality]
    job = output / modality / f'architecture_{architecture:02d}' / f'fold_{fold}'
    job.mkdir(parents=True, exist_ok=True)
    done = job / 'completed.json'
    selected_path = job / 'selected_predictions.npz'
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
    print(f'[GPU {device}] Training {modality} #{architecture}, fold {fold}, restarts={config["runs"]}', flush=True)
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    env['OMP_NUM_THREADS'] = str(config.get('cpu_threads_per_worker', 2))
    env['MKL_NUM_THREADS'] = env['OMP_NUM_THREADS']
    start = time.monotonic()
    with (job / 'train.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen([sys.executable, '-u', str(source / script)], cwd=job, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            with registry_lock:
                if stop_event.is_set():
                    child.terminate()
                    child.wait()
                    raise RuntimeError('Training cancelled.')
                registry.add(child)
            try:
                for line in child.stdout:
                    if line.startswith(('Fold:', 'best val_loss')):
                        print(f'[GPU {device}] ' + line, end='', flush=True)
                    log.write(line)
                    log.flush()
                returncode = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
            finally:
                with registry_lock:
                    registry.discard(child)
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
    marker_tmp = done.with_suffix('.tmp')
    marker_tmp.write_text(json.dumps(dict(modality=modality, architecture=architecture, fold=fold,
                                   gpu=str(device), completed_restarts=config['runs'], seconds=time.monotonic() - start), indent=2))
    marker_tmp.replace(done)
    return job


def worker_devices(requested=0):
    import torch
    available = torch.cuda.device_count()
    if type(requested) is not int or requested < 0 or requested > available:
        raise ValueError(f'gpu_workers must be from 0 (auto) to {available}.')
    if not available:
        return [None]
    count = requested or available
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = visible.split(',') if visible is not None else [str(i) for i in range(available)]
    if len(tokens) < available:
        raise ValueError('CUDA device enumeration and CUDA_VISIBLE_DEVICES disagree.')
    return [token.strip() for token in tokens[:count]]


def dispatch_jobs(items, devices, execute, completed, cancel):
    """Keep one independent subprocess per device; serialize result publication."""
    iterator = iter(items)
    pool = ThreadPoolExecutor(max_workers=len(devices))
    active = {}
    try:
        for device in devices:
            item = next(iterator, None)
            if item is not None:
                active[pool.submit(execute, item, device)] = device
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                device = active.pop(future)
                result = future.result()
                completed(result)
                item = next(iterator, None)
                if item is not None:
                    active[pool.submit(execute, item, device)] = device
    except BaseException:
        cancel()
        for future in active:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def print_eta(output, config, workers):
    durations = {modality: [] for modality in SCRIPTS}
    remaining = {modality: 0 for modality in SCRIPTS}
    for modality in SCRIPTS:
        for architecture in config['architectures']:
            for fold in config['folds']:
                marker = output / modality / f'architecture_{architecture:02d}' / f'fold_{fold}' / 'completed.json'
                if marker.exists():
                    data = json.loads(marker.read_text())
                    durations[modality].append(data['seconds'] / data['completed_restarts'])
                else:
                    remaining[modality] += config['runs']
    if not all(durations.values()):
        print('ETA: waiting for at least one completed job from each network family.', flush=True)
        return
    average = {m: sum(values) / len(values) for m, values in durations.items()}
    if config['epochs'] < 1000:
        print('Smoke epoch cap: timings cannot reliably estimate the full paper schedule. Use MODE=benchmark for an early-stopping timing sample.', flush=True)
        return
    remaining_hours = sum(average[m] * remaining[m] for m in SCRIPTS) / workers / 3600
    requested_runs = 4 * len(config['architectures']) * len(config['folds']) * config['runs']
    if config.get('mode', 'paper') == 'baseline':
        total_hours = sum(average.values()) * len(config['folds']) * config['runs'] / workers / 3600
        print(f'Measured BASELINE ETA: remaining ~{remaining_hours:.1f} hours; total {requested_runs} runs ~{total_hours:.1f} hours on {workers} GPU worker(s).', flush=True)
    else:
        full_hours = sum(average.values()) * 1800 / workers / 3600
        print(f'Measured ETA: remaining ~{remaining_hours:.1f} hours; full 7200-restart schedule ~{full_hours:.1f} hours on {workers} GPU worker(s).', flush=True)
    print('Estimate uses observed time per restart by network family; architecture and stopping-epoch differences can change it.', flush=True)


def run_experiment(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source = Path(__file__).resolve().parent
    output = config_path.parent
    catalog = json.loads((source / 'paper_architectures.json').read_text())['models']
    if config['mode'] not in ['paper', 'baseline', 'smoke', 'benchmark']:
        raise ValueError('Choose baseline, paper, smoke or benchmark.')
    if not config['folds'] or len(set(config['folds'])) != len(config['folds']) or any(f not in range(1, 10) for f in config['folds']):
        raise ValueError('Invalid folds.')
    if not config['architectures'] or len(set(config['architectures'])) != len(config['architectures']) or any(a not in range(1, 11) for a in config['architectures']) or 1 not in config['architectures']:
        raise ValueError('Choose distinct architecture IDs from 1 to 10, including 1.')
    if config['runs'] < 1 or config['epochs'] < 1:
        raise ValueError('Runs and epochs must be positive.')
    if config['mode'] == 'paper' and (config['folds'] != list(range(1, 10)) or config['architectures'] != list(range(1, 11)) or config['runs'] != 20 or config['epochs'] != 1000):
        raise ValueError('Paper mode requires all 9 folds, 10 architectures, 20 restarts and max 1000 epochs.')
    if config['mode'] == 'baseline' and (config['folds'] != list(range(1, 10)) or config['architectures'] != [1] or config['epochs'] != 1000):
        raise ValueError('Baseline requires all 9 folds, published architecture 1 and max 1000 epochs.')
    os.environ.pop('FIDGETY_DATA_ZIP', None)
    os.environ['FIDGETY_DATA_ROOT'] = config['data_root']
    details = validate_data(config['folds'], require_paper_subject_counts=config['mode'] in ['paper', 'baseline'])
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
    if type(config.get('cpu_threads_per_worker', 2)) is not int or config.get('cpu_threads_per_worker', 2) < 1:
        raise ValueError('cpu_threads_per_worker must be a positive integer.')
    devices = worker_devices(config.get('gpu_workers', 0))
    print('GPU worker devices:', devices, flush=True)
    pending_jobs = []
    for fold in config['folds']:
        for architecture in config['architectures']:
            for modality_index, modality in enumerate(SCRIPTS):
                job = output / modality / f'architecture_{architecture:02d}' / f'fold_{fold}'
                if (job / 'completed.json').exists():
                    if not (job / 'selected_predictions.npz').exists() or not list(job.glob('*.pt')):
                        raise FileNotFoundError('Completed job is missing predictions or checkpoint: ' + str(job))
                    print(f'Skipping completed {modality} #{architecture}, fold {fold}', flush=True)
                else:
                    pending_jobs.append((modality, modality_index, architecture, fold))
    stop_event = threading.Event()
    registry, registry_lock = set(), threading.Lock()

    def execute(item, device):
        return execute_job(item, device, config, source, output, catalog, stop_event, registry, registry_lock)

    def completed(job):
        # Only the coordinator writes the shared CSV.
        collect_results(output, config)
        print('Completed:', job.relative_to(output), flush=True)
        print_eta(output, config, len(devices))

    def cancel():
        stop_event.set()
        with registry_lock:
            for child in list(registry):
                if child.poll() is None:
                    child.terminate()

    dispatch_jobs(pending_jobs, devices, execute, completed, cancel)
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
