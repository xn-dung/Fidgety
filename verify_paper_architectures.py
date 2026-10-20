"""Check every published architecture against the original script builders."""
import ast
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_utils import ChannelsLastFlatten, initialize_model, predict, EarlyStopping
from paper_protocol import SCRIPTS

root = Path(__file__).resolve().parent
catalog = json.loads((root / 'paper_architectures.json').read_text())['models']
shapes = {'mat': (500, 6), 'imu': (300, 36), 'vid': (250, 60), 'fusion': (250, 102)}
torch.set_num_threads(2)
for modality, filename in SCRIPTS.items():
    tree = ast.parse((root / filename).read_text())
    fold = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'fi')
    run = next(n for n in fold.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
    end = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'initialize_model')
    code = compile(ast.Module(body=run.body[start:end + 1], type_ignores=[]), filename, 'exec')
    for architecture, parameters in catalog[modality].items():
        x = np.zeros((1, *shapes[modality]), dtype=np.float32)
        scope = dict(nn=nn, EarlyStopping=EarlyStopping, ChannelsLastFlatten=ChannelsLastFlatten, initialize_model=initialize_model, x_train=x, **parameters)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, scope)
        model = scope['model']
        convolutions = [layer for layer in model if isinstance(layer, nn.Conv2d)]
        assert len(convolutions) == 3
        for i, layer in enumerate(convolutions, 1):
            assert layer.out_channels == parameters[f'kn{i}']
            assert layer.kernel_size == (parameters[f'kw{i}'], 1)
        assert len([layer for layer in model if isinstance(layer, nn.BatchNorm2d)]) == 3
        assert len([layer for layer in model if isinstance(layer, nn.BatchNorm1d)]) == 1
        assert predict(model, x).shape == (1, 1)
        kn1, kn2, kn3 = [parameters[k] for k in ['kn1', 'kn2', 'kn3']]
        kw1, kw2, kw3 = [parameters[k] for k in ['kw1', 'kw2', 'kw3']]
        fc = parameters['fc1']
        expected = kn1 * (kw1 + 1) + kn2 * (kn1 * kw2 + 1) + kn3 * (kn2 * kw3 + 1)
        expected += fc * (shapes[modality][1] * kn3 + 1) + fc + 1 + 2 * (kn1 + kn2 + kn3 + fc)
        assert sum(p.numel() for p in model.parameters()) == expected
    print(f'{modality}: all 10 architecture shapes, layer counts and parameter counts OK')
