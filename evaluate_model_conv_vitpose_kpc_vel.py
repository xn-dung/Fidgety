from data_io import loadtxt, fold_indices, run_count
import torch
from torch_utils import predict, load_model


import numpy as np
from sklearn.metrics import confusion_matrix

import glob



for fi in fold_indices():

    print('Fold: ', fi+1)

    file = open('prediction_vitpose_kpc_vel_cnn_best_model_fold_' + str(fi+1) + '.txt','a+t')

    x_train_pos = loadtxt('../data/data_vitpose_kpc_9fold/input_vitpose_9fold_fold_' + str(fi+1) + '_train.txt')
    x_test_pos = loadtxt('../data/data_vitpose_kpc_9fold/input_vitpose_9fold_fold_' + str(fi+1) + '_test.txt')

    y_train = loadtxt('../data/data_vitpose_kpc_9fold/output_vitpose_9fold_fold_' + str(fi+1) + '_train.txt')
    y_test = loadtxt('../data/data_vitpose_kpc_9fold/output_vitpose_9fold_fold_' + str(fi+1) + '_test.txt')

    x_train_pos = np.reshape(x_train_pos, (x_train_pos.shape[0],250,30), order='F')
    x_test_pos = np.reshape(x_test_pos, (x_test_pos.shape[0],250,30), order='F')

    x_train = np.zeros((x_train_pos.shape[0],250,60))
    x_test = np.zeros((x_test_pos.shape[0],250,60))

    x_train[:,:,0:30] = x_train_pos
    x_train[:,1:250,30:60] = np.diff(x_train_pos, axis=1)

    x_test[:,:,0:30] = x_test_pos
    x_test[:,1:250,30:60] = np.diff(x_test_pos, axis=1)

    del x_train_pos
    del x_test_pos

    ### apply z-score normalisation for position and velocity
    x_mean_pos = np.mean(x_train[:,:,0:30])
    x_sd_pos = np.std(x_train[:,:,0:30])

    x_mean_vel = np.mean(x_train[:,:,30:60])
    x_sd_vel = np.std(x_train[:,:,30:60])

    del x_train

    x_test[:,:,0:30] = (x_test[:,:,0:30] - x_mean_pos) / x_sd_pos
    x_test[:,:,30:60] = (x_test[:,:,30:60] - x_mean_vel) / x_sd_vel

    # print('x_test size:', x_test.shape)
    # print('y_test size:', y_test.shape)

    for ri in range(10):

        model_file = glob.glob( '../results_9fold_grid/vitpose_kpc_vel_cnn_best_model_' + str(ri+1) + '/model_vitpose_kpc_vel_9fold_fold_' + str(fi+1) + '*.pt')
        if not model_file:
            raise FileNotFoundError('No PyTorch checkpoint found for fold ' + str(fi+1) + ', run ' + str(ri+1))
        # print(model_file[0])

        model = load_model(model_file[0])

        # print(model)

        y_pred = predict(model, x_test.reshape( x_test.shape[0],x_test.shape[1],x_test.shape[2],1 ), verbose=0).reshape(-1)

        np.savetxt(file, y_pred, fmt = '%12.6f', newline=' ')
        file.write('\n')

        y_pred = torch.round(torch.sigmoid(torch.as_tensor(y_pred))).numpy()
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        acc = (tp+tn)/(tp+fp+tn+fn)
        bal_acc = (sens+spec)/2

        print(tp, tn, fp, fn, sens, spec, acc, bal_acc)

    file.close()
