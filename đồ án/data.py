from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np


@dataclass
class FoldData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class TextStore:
    def __init__(self, location):
        self.location = Path(location)
        if not self.location.exists():
            raise FileNotFoundError(self.location)
        self.archive = self.location.is_file()
        self.members = None
        if self.archive:
            with zipfile.ZipFile(self.location) as stream:
                self.members = [name for name in stream.namelist() if not name.endswith("/")]

    def loadtxt(self, relative, **kwargs):
        relative = relative.replace("\\", "/").lstrip("/")
        if not self.archive:
            candidates = [self.location / relative, self.location / "data" / relative]
            for candidate in candidates:
                if candidate.is_file():
                    return np.loadtxt(candidate, **kwargs)
            raise FileNotFoundError(relative)
        matches = [name for name in self.members if name == relative or name.endswith("/" + relative)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one ZIP member for {relative}, found {len(matches)}")
        with zipfile.ZipFile(self.location) as archive:
            with archive.open(matches[0]) as stream:
                return np.loadtxt(stream, **kwargs)


def _positions(values):
    values = np.asarray(values, dtype=np.float32)
    return np.reshape(values, (values.shape[0], 250, 30), order="F")


def _features(positions):
    result = np.zeros((positions.shape[0], 250, 60), dtype=np.float32)
    result[:, :, :30] = positions
    result[:, 1:, 30:] = np.diff(positions, axis=1)
    return result


def _normalize(train, test):
    result_train = train.copy()
    result_test = test.copy()
    for start, end in [(0, 30), (30, 60)]:
        mean = float(result_train[:, :, start:end].mean())
        std = float(result_train[:, :, start:end].std())
        if std == 0:
            raise ValueError("Training standard deviation is zero")
        result_train[:, :, start:end] = (result_train[:, :, start:end] - mean) / std
        result_test[:, :, start:end] = (result_test[:, :, start:end] - mean) / std
    return result_train, result_test


def load_video_fold(store, fold):
    base = "data_vitpose_kpc_9fold"
    train_input = f"{base}/input_vitpose_9fold_fold_{fold}_train.txt"
    test_input = f"{base}/input_vitpose_9fold_fold_{fold}_test.txt"
    train_output = f"{base}/output_vitpose_9fold_fold_{fold}_train.txt"
    test_output = f"{base}/output_vitpose_9fold_fold_{fold}_test.txt"
    train_positions = _positions(store.loadtxt(train_input))
    test_positions = _positions(store.loadtxt(test_input))
    y_train = np.asarray(store.loadtxt(train_output), dtype=np.int64).reshape(-1)
    y_test = np.asarray(store.loadtxt(test_output), dtype=np.int64).reshape(-1)
    x_train, x_test = _normalize(_features(train_positions), _features(test_positions))
    indices = np.asarray(store.loadtxt(f"random_indices_9fold/random_indices_train_9fold_fold_{fold}.txt", dtype=int), dtype=np.int64).reshape(-1) - 1
    if len(indices) != len(x_train) or set(indices.tolist()) != set(range(len(x_train))):
        raise ValueError(f"Invalid random indices for fold {fold}")
    x_train = x_train[indices]
    y_train = y_train[indices]
    split = 7 * len(x_train) // 8
    return FoldData(x_train, y_train, x_train[:split], y_train[:split], x_train[split:], y_train[split:], x_test, y_test)
