"""Check the fast default and nine-fold paper-style baseline reporting."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import nbformat
from scipy.special import expit

from paper_protocol import collect_results
from paper_runner import print_eta

root = Path(__file__).resolve().parent
notebook = json.loads((root / 'train_kaggle.ipynb').read_text())
nbformat.validate(notebook)
for cell in notebook['cells']:
    assert cell['cell_type'] == 'code' and not cell['outputs']
    ast.parse(''.join(cell['source']))
scope = {}
exec(''.join(notebook['cells'][0]['source']), scope)
assert scope['MODE'] == 'baseline'
assert scope['FOLDS'] == list(range(1, 10))
assert scope['ARCHITECTURES'] == [1] and scope['RUNS'] == 1 and scope['EPOCHS'] == 1000
assert 'baseline' in str(scope['OUTPUT_DIR'])
with tempfile.TemporaryDirectory(dir=root) as tmp:
    output = Path(tmp)
    config = dict(mode='baseline', architectures=[1], folds=list(range(1, 10)), runs=1, epochs=1000)
    for modality in ['mat', 'imu', 'vid', 'fusion']:
        for fold in config['folds']:
            job = output / modality / 'architecture_01' / f'fold_{fold}'
            job.mkdir(parents=True)
            logits = np.array([-1., .4, -.2, .8]) * fold / 5
            np.savez(job / 'selected_predictions.npz', labels=[0, 1, 0, 1], logits=logits,
                     probabilities=expit(logits), selected_run=1, best_val_loss=.4, epochs_trained=15)
            (job / 'completed.json').write_text(json.dumps(dict(seconds=300, completed_restarts=1)))
    result = collect_results(output, config)
    assert result.reduced_baseline_complete.all()
    assert not result.full_paper_schedule_complete.any()
    assert (result.training_restarts_per_fold == 1).all()
    assert len(result[result.row_type == 'fold']) == 72
    assert len(result[result.row_type == 'table2']) == 8
    assert (result[result.row_type == 'table2'].n_folds == 9).all()
    assert len(result[result.row_type == 'table3']) == 28
    assert len(list(output.glob('*.csv'))) == 1
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_eta(output, config, 2)
    assert 'total 36 runs ~1.5 hours' in buffer.getvalue()
    assert '7200' not in buffer.getvalue()
print('Fast baseline default: 36 runs, nine-fold tables/CI/Wilcoxon, honest completion flags and baseline ETA OK (synthetic predictions).')
