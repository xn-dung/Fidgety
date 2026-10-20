"""Optuna search for the PyTorch experiments (TPE and exhaustive grid)."""
from pathlib import Path
import math
import optuna
from torch_utils import fit


class HyperParameters:
    def __init__(self, trial=None):
        self.trial = trial
        self.space = {}

    def Choice(self, name, values):
        self.space[name] = list(values)
        return self.trial.suggest_categorical(name, values) if self.trial else values[0]

    def Int(self, name, min_value, max_value, step=1):
        return self.Choice(name, list(range(min_value, max_value + 1, step)))


class BayesianOptimization:
    def __init__(self, model_builder, objective='val_loss', max_trials=1000,
                 max_retries_per_trial=0, max_consecutive_failed_trials=3,
                 directory='hyperpar_tuning', project_name='run_1'):
        self.model_builder = model_builder
        self.max_trials = max_trials
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.study = optuna.create_study(
            study_name=project_name, direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            storage='sqlite:///' + (self.directory / (project_name + '.sqlite3')).resolve().as_posix(),
            load_if_exists=True)

    def search(self, x, y, **kwargs):
        def objective(trial):
            model = self.model_builder(HyperParameters(trial))
            history = fit(model, x, y, **kwargs)
            return min(history.history['val_loss'])
        remaining = max(0, self.max_trials - len(self.study.trials))
        if remaining:
            self.study.optimize(objective, n_trials=remaining)

    def results_summary(self):
        trials = [trial for trial in self.study.trials if trial.value is not None]
        for trial in sorted(trials, key=lambda trial: trial.value)[:10]:
            print('Trial:', trial.number, 'val_loss:', trial.value, 'parameters:', trial.params)


class GridSearch(BayesianOptimization):
    def __init__(self, model_builder, **kwargs):
        hp = HyperParameters()
        model_builder(hp)
        # These grid scripts use fixed layer counts, so every parameter is active.
        size = math.prod(len(values) for values in hp.space.values())
        super().__init__(model_builder, max_trials=size, **kwargs)
        self.study.sampler = optuna.samplers.GridSampler(hp.space)
