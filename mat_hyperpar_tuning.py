from data_io import loadtxt, fold_indices, run_count
import torch
from torch import nn
from torch_utils import initialize_model, EarlyStopping, ChannelsLastFlatten

import torch_tuning as kt
import numpy as np
from contextlib import redirect_stdout


def model_builder(hp):
  model = nn.Sequential()
  channels = 1


  # Tune the number of units and the number of Conv layers
  for i in range(hp.Int('conv_layers', min_value=1, max_value=3, step=1)):
    model.append(nn.Conv2d(channels, hp.Choice('conv_units_' + str(i), [4, 8, 16, 32, 64, 128]), (hp.Choice('kernel_size_' + str(i), [5, 7, 9, 13, 17, 25, 33]), 1), padding="same"))
    model.append(nn.ReLU())
    channels = hp.Choice('conv_units_' + str(i), [4, 8, 16, 32, 64, 128])
    model.append(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01))
    model.append(nn.Dropout(0.2))

  model.append(nn.AvgPool2d((500, 1), stride=(500, 1)))
  # Preserve the author's channels-last flatten order.
  model.append(ChannelsLastFlatten())

  # Tune the number of units and the number of Dense layers
  for j in range(hp.Int('dense_layers', min_value=1, max_value=2, step=1)):
    units = hp.Choice('dense_units_' + str(j), [64, 128, 256])
    model.append(nn.LazyLinear(units))
    model.append(nn.ReLU())
    model.append(nn.BatchNorm1d(units, eps=1e-3, momentum=0.01))
    model.append(nn.Dropout(0.2))

  units = 1
  model.append(nn.LazyLinear(units))

  initialize_model(model, np.zeros((1, 500, 6), dtype=np.float32))

  return model

stop_early = EarlyStopping(monitor='val_loss', patience=10)



x_train = loadtxt('../data/data_mat_hyperpar/input_mat_hyperpar.txt')
x_train = np.reshape(x_train, (x_train.shape[0],500,6), order='F')
y_train = loadtxt('../data/data_mat_hyperpar/output_mat_hyperpar.txt')

ind = loadtxt('../data/random_indices_hyperpar/random_indices_hyperpar.txt', dtype = 'int')
ind = ind-1
x_train = x_train[ind,:,:]
y_train = y_train[ind]

print('x_train size:', x_train.shape)
print('y_train size:', y_train.shape)


for run in range(1,5):

    tuner = kt.BayesianOptimization(model_builder,
                        objective='val_loss',
                        max_trials=1000,
                        max_retries_per_trial=0,
                        max_consecutive_failed_trials=3,
                        directory='hyperpar_tuning_mat_bayes',
                        project_name='run_'+str(run+1))


    tuner.search(x_train, y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[stop_early])

    tuner.results_summary()
    with open('hyperpar_tuning_mat_bayes/best_models_run_' + str(run+1) + '.txt', 'w') as f:
        with redirect_stdout(f):
            tuner.results_summary()

    del tuner
