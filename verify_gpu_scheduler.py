"""Verify two-device scheduling, CUDA pinning, cancellation and measured ETA."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from paper_runner import dispatch_jobs, print_eta, worker_devices

with patch('torch.cuda.device_count', return_value=2):
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'}):
        assert worker_devices(0) == ['GPU-first', 'GPU-second']
        assert worker_devices(1) == ['GPU-first']
        try:
            worker_devices(3)
        except ValueError:
            pass
        else:
            raise AssertionError('Requested more GPUs than available.')

active, peak, lock = {}, [0], threading.Lock()
barrier = threading.Barrier(2)
assigned, results = [], []
main_thread = threading.get_ident()

def execute(item, device):
    with lock:
        assert device not in active
        active[device] = item
        peak[0] = max(peak[0], len(active))
        assigned.append((item, device))
    if item < 2:
        barrier.wait(timeout=5)
    time.sleep(0.02)
    with lock:
        del active[device]
    return item

def completed(result):
    assert threading.get_ident() == main_thread
    results.append(result)

def unexpected_cancel():
    raise AssertionError('Unexpected cancellation.')

dispatch_jobs(list(range(8)), ['0', '1'], execute, completed, unexpected_cancel)
assert peak[0] == 2 and sorted(results) == list(range(8))
assert {device for _, device in assigned} == {'0', '1'}
stop = threading.Event()
def fail(item, device):
    if item == 0:
        raise RuntimeError('Expected worker failure')
    assert stop.wait(timeout=5)
    return item
try:
    dispatch_jobs([0, 1], ['0', '1'], fail, lambda result: None, stop.set)
except RuntimeError:
    assert stop.is_set()
else:
    raise AssertionError('Worker error was swallowed.')

with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
    output = Path(temporary)
    for modality in ['mat', 'imu', 'vid', 'fusion']:
        job = output / modality / 'architecture_01' / 'fold_1'
        job.mkdir(parents=True)
        (job / 'completed.json').write_text(json.dumps(dict(seconds=600, completed_restarts=20)))
    config = dict(architectures=list(range(1, 11)), folds=list(range(1, 10)), runs=20, epochs=1000)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_eta(output, config, 2)
    assert 'full 7200-restart schedule ~30.0 hours' in buffer.getvalue()
    config['epochs'] = 5
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_eta(output, config, 2)
    assert 'cannot reliably estimate' in buffer.getvalue()
print('Two-device dispatch, CUDA device mapping, coordinator-only publication, cancellation and ETA math OK (simulated devices).')
