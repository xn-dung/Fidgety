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

  model.append(nn.AvgPool2d((250, 1), stride=(250, 1)))
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

  initialize_model(model, np.zeros((1, 250, 102), dtype=np.float32))

  return model

stop_early = EarlyStopping(monitor='val_loss', patience=10)



### load mat data
x_train_mat = loadtxt('../data/data_mat_hyperpar/input_mat_hyperpar.txt')
x_train_mat = np.reshape(x_train_mat, (x_train_mat.shape[0],500,6), order='F')

### resample
smpl = np.round(np.linspace(1,500,250)).astype(int) - 1
x_train_mat = x_train_mat[:,smpl,:]


### load vitpose data
x_train_pos = loadtxt('../data/data_vitpose_kpc_hyperpar/input_vitpose_hyperpar.txt')
x_train_pos = np.reshape(x_train_pos, (x_train_pos.shape[0],250,30), order='F')

x_train_vit = np.zeros((x_train_pos.shape[0],250,60))

x_train_vit[:,:,0:30] = x_train_pos
x_train_vit[:,1:250,30:60] = np.diff(x_train_pos, axis=1)

del x_train_pos

### apply z-score normalisation for position and velocity
x_mean_pos = np.mean(x_train_vit[:,:,0:30])
x_sd_pos = np.std(x_train_vit[:,:,0:30])

x_mean_vel = np.mean(x_train_vit[:,:,30:60])
x_sd_vel = np.std(x_train_vit[:,:,30:60])

x_train_vit[:,:,0:30] = (x_train_vit[:,:,0:30] - x_mean_pos) / x_sd_pos
x_train_vit[:,:,30:60] = (x_train_vit[:,:,30:60] - x_mean_vel) / x_sd_vel


### load imu data
x_train_imu = loadtxt('../data/data_imu_hyperpar/input_imu_hyperpar.txt')
x_train_imu = np.reshape(x_train_imu, (x_train_imu.shape[0],300,36), order='F')

### resample
smpl = np.round(np.linspace(1,300,250)).astype(int) - 1
x_train_imu = x_train_imu[:,smpl,:]

### apply z-score normalisation for acceleration and gyro
ind_acc = np.array([0,1,2,   6,7,8, 12,13,14, 18,19,20, 24,25,26, 30,31,32])
ind_gyr = np.array([3,4,5, 9,10,11, 15,16,17, 21,22,23, 27,28,29, 33,34,35])

x_mean_acc = np.mean(x_train_imu[:,:,ind_acc])
x_sd_acc = np.std(x_train_imu[:,:,ind_acc])

x_mean_gyr = np.mean(x_train_imu[:,:,ind_gyr])
x_sd_gyr = np.std(x_train_imu[:,:,ind_gyr])

x_train_imu[:,:,ind_acc] = (x_train_imu[:,:,ind_acc] - x_mean_acc) / x_sd_acc
x_train_imu[:,:,ind_gyr] = (x_train_imu[:,:,ind_gyr] - x_mean_gyr) / x_sd_gyr


### pool data
x_train = np.zeros((x_train_mat.shape[0],250,102))

x_train[:,:,0:6] = x_train_mat
x_train[:,:,6:66] = x_train_vit
x_train[:,:,66:102] = x_train_imu

del x_train_mat
del x_train_vit
del x_train_imu


### load output data
y_train = loadtxt('../data/data_mat_hyperpar/output_mat_hyperpar.txt')


ind = loadtxt('../data/random_indices_hyperpar/random_indices_hyperpar.txt', dtype = 'int')
ind = ind-1
x_train = x_train[ind,:,:]
y_train = y_train[ind]

print('x_train size:', x_train.shape)
print('y_train size:', y_train.shape)


for run in range(5):

    tuner = kt.BayesianOptimization(model_builder,
                        objective='val_loss',
                        max_trials=1000,
                        max_retries_per_trial=0,
                        max_consecutive_failed_trials=3,
                        directory='hyperpar_tuning_mat_vitpose_imu_bayes',
                        project_name='run_'+str(run+1))


    tuner.search(x_train, y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[stop_early])

    tuner.results_summary()
    with open('hyperpar_tuning_mat_vitpose_imu_bayes/best_models_run_' + str(run+1) + '.txt', 'w') as f:
        with redirect_stdout(f):
            tuner.results_summary()

    del tuner
