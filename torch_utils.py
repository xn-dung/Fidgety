"""Shared PyTorch training routines; scripts retain the original experiment flow."""
import copy
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChannelsLastFlatten(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)


class EarlyStopping:
    def __init__(self, monitor='val_loss', min_delta=0, patience=10,
                 mode='auto', restore_best_weights=False, verbose=0):
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights


def _inputs(x):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    return x.permute(0, 3, 1, 2).contiguous()


def initialize_model(model, x):
    model.to(DEVICE).eval()
    with torch.no_grad():
        model(_inputs(x).to(DEVICE))
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    return model


def _loader(x, y, batch_size, shuffle=False):
    return DataLoader(TensorDataset(_inputs(x), torch.as_tensor(
        np.asarray(y), dtype=torch.float32).reshape(-1, 1)),
        batch_size=batch_size, shuffle=shuffle)


def _score(model, loader):
    model.eval()
    total_loss, correct, count = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='sum').item()
            # The original BinaryAccuracy uses a 0.5 threshold on raw logits.
            correct += ((logits > 0.5) == y.bool()).sum().item()
            count += len(y)
    return [total_loss / count, correct / count]


def fit(model, x, y, batch_size=4, validation_split=0.2, epochs=100,
        verbose=1, callbacks=None):
    epochs = int(os.environ.get('FIDGETY_EPOCHS', epochs))
    if epochs < 1:
        raise ValueError('FIDGETY_EPOCHS must be positive.')
    split = int(len(x) * (1 - validation_split))
    if split < 1 or split >= len(x):
        raise ValueError('Training and validation partitions must both be nonempty.')
    train_loader = _loader(x[:split], y[:split], batch_size, shuffle=True)
    val_loader = _loader(x[split:], y[split:], batch_size)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    stop = (callbacks or [EarlyStopping()])[0]
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best, wait, weights = float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        total_loss, correct, count = 0., 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # Dense BatchNorm cannot estimate variance from a single sample.
            # Use stored statistics for that final batch; retain every sample.
            if len(yb) == 1:
                for layer in model.modules():
                    if isinstance(layer, nn.BatchNorm1d):
                        layer.eval()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += ((logits.detach() > 0.5) == yb.bool()).sum().item()
            count += len(yb)
        val_loss, val_accuracy = _score(model, val_loader)
        for key, value in zip(history, [total_loss/count, correct/count, val_loss, val_accuracy]):
            history[key].append(value)
        if verbose:
            print('Epoch:', epoch+1, 'loss:', history['loss'][-1], 'val_loss:', val_loss, 'val_accuracy:', val_accuracy)
        if val_loss < best - stop.min_delta:
            best, wait = val_loss, 0
            if stop.restore_best_weights:
                weights = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= stop.patience:
                break
    if weights is not None:
        model.load_state_dict(weights)
    return SimpleNamespace(history=history)


def evaluate(model, x, y, verbose=1, batch_size=32):
    score = _score(model.to(DEVICE), _loader(x, y, batch_size))
    if verbose:
        print('loss:', score[0], 'accuracy:', score[1])
    return score


def predict(model, x, verbose=0, batch_size=32):
    model.to(DEVICE).eval()
    outputs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(_inputs(x)), batch_size=batch_size):
            outputs.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(outputs)


def save_model(model, filename):
    layers = []
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            name, args, kwargs = 'Conv2d', [layer.in_channels, layer.out_channels, layer.kernel_size], {'padding': layer.padding}
        elif isinstance(layer, nn.Linear):
            name, args, kwargs = 'Linear', [layer.in_features, layer.out_features], {}
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            name, args, kwargs = type(layer).__name__, [layer.num_features], {'eps': layer.eps, 'momentum': layer.momentum}
        elif isinstance(layer, nn.Dropout):
            name, args, kwargs = 'Dropout', [layer.p], {}
        elif isinstance(layer, nn.AvgPool2d):
            name, args, kwargs = 'AvgPool2d', [layer.kernel_size], {'stride': layer.stride}
        elif isinstance(layer, nn.ReLU):
            name, args, kwargs = 'ReLU', [], {}
        elif isinstance(layer, ChannelsLastFlatten):
            name, args, kwargs = 'ChannelsLastFlatten', [], {}
        else:
            raise TypeError(type(layer))
        layers.append((name, args, kwargs))
    torch.save({'layers': layers, 'state_dict': model.state_dict()}, filename)


def load_model(filename):
    checkpoint = torch.load(filename, map_location=DEVICE, weights_only=True)
    model = nn.Sequential(*[(ChannelsLastFlatten if name == 'ChannelsLastFlatten' else getattr(nn, name))(*args, **kwargs)
                            for name, args, kwargs in checkpoint['layers']]).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()
