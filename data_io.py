"""Read the author's text files from disk or directly from a ZIP archive."""
import os
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


def loadtxt(filename, *args, **kwargs):
    archive_path = os.environ.get('FIDGETY_DATA_ZIP')
    data_root = os.environ.get('FIDGETY_DATA_ROOT')
    if not archive_path and not data_root:
        return np.loadtxt(filename, *args, **kwargs)
    parts = PurePosixPath(str(filename).replace('\\', '/')).parts
    if 'data' not in parts:
        raise ValueError('Expected a data/ path: ' + str(filename))
    relative = '/'.join(parts[parts.index('data') + 1:])
    if data_root and not archive_path:
        return np.loadtxt(Path(data_root) / relative, *args, **kwargs)
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist()
                      if not name.endswith('/') and
                      (name == relative or name.endswith('/' + relative))]
        if len(candidates) != 1:
            raise FileNotFoundError('Expected exactly one ZIP member for ' + relative +
                                    '; found ' + str(len(candidates)))
        with archive.open(candidates[0]) as stream:
            return np.loadtxt(stream, *args, **kwargs)


def fold_indices():
    value = os.environ.get('FIDGETY_FOLDS', '1,2,3,4,5,6,7,8,9')
    folds = [int(part.strip()) for part in value.split(',')]
    if not folds or len(set(folds)) != len(folds) or any(f < 1 or f > 9 for f in folds):
        raise ValueError('FIDGETY_FOLDS must contain distinct fold numbers from 1 to 9.')
    return [fold - 1 for fold in folds]


def run_count(default=20):
    count = int(os.environ.get('FIDGETY_RUNS', default))
    if count < 1:
        raise ValueError('FIDGETY_RUNS must be positive.')
    return count
