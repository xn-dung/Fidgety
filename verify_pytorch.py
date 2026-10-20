import ast
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from torch_utils import *

torch.set_num_threads(2)
shapes = {'mat': (500, 6), 'imu_acc_gyr': (300, 36), 'vitpose_kpc_vel': (250, 60), 'mat_vitpose_imu': (250, 102)}
class HP:
    def Int(self, name, min_value, max_value, step=1): return max_value
    def Choice(self, name, values): return values[0]

for path in Path('.').glob('*hyperpar*.py'):
    tree = ast.parse(path.read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    scope = dict(globals())
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(path), 'exec'), scope)
    model = scope['model_builder'](HP())
    shape = shapes[path.name.split('_hyperpar')[0]]
    x = np.random.default_rng(1).normal(size=(5, *shape)).astype('float32')
    result = predict(model, x)
    assert result.shape == (5, 1) and np.isfinite(result).all()
    with tempfile.TemporaryDirectory(dir='.') as tmp:
        filename = str(Path(tmp) / 'model.pt')
        save_model(model, filename)
        np.testing.assert_allclose(result, predict(load_model(filename), x))
    print('Architecture + checkpoint OK:', path.name)

for path in Path('.').glob('*conv_best_model_9fold.py'):
    tree = ast.parse(path.read_text())
    scope = dict(globals())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    fold = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'fi')
    run = next(n for n in fold.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
    end = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'initialize_model')
    shape = shapes[path.name.split('_conv_best')[0]]
    scope['x_train'] = np.random.default_rng(2).normal(size=(5, *shape)).astype('float32')
    exec(compile(ast.Module(body=run.body[start:end+1], type_ignores=[]), str(path), 'exec'), scope)
    model = scope['model']
    hist = fit(model, scope['x_train'], np.array([0,1,0,1,0]), validation_split=0.2, batch_size=3, epochs=2, verbose=0,
               callbacks=[EarlyStopping(restore_best_weights=True)])
    assert len(hist.history['val_loss']) == 2
    assert np.isfinite(evaluate(model, scope['x_train'], np.array([0,1,0,1,0]), verbose=0)).all()
    print('Training + validation OK:', path.name)
