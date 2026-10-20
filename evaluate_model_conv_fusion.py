from data_io import loadtxt, fold_indices, run_count
import torch
from torch_utils import predict, load_model


import numpy as np
from sklearn.metrics import confusion_matrix

import glob



for fi in fold_indices():

    print('Fold: ', fi+1)

    file = open('prediction_fusion_cnn_best_model_fold_' + str(fi+1) + '.txt','a+t')

    ### load mat data
    x_train_mat = loadtxt('../data/data_mat_9fold/input_mat_9fold_fold_' + str(fi+1) + '_train.txt')
    x_test_mat = loadtxt('../data/data_mat_9fold/input_mat_9fold_fold_' + str(fi+1) + '_test.txt')

    x_train_mat = np.reshape(x_train_mat, (x_train_mat.shape[0],500,6), order='F')
    x_test_mat = np.reshape(x_test_mat, (x_test_mat.shape[0],500,6), order='F')

    ### resample
    smpl = np.round(np.linspace(1,500,250)).astype(int) - 1
    x_train_mat = x_train_mat[:,smpl,:]
    x_test_mat = x_test_mat[:,smpl,:]


    ### load vitpose data
    x_train_pos = loadtxt('../data/data_vitpose_kpc_9fold/input_vitpose_9fold_fold_' + str(fi+1) + '_train.txt')
    x_test_pos = loadtxt('../data/data_vitpose_kpc_9fold/input_vitpose_9fold_fold_' + str(fi+1) + '_test.txt')

    x_train_pos = np.reshape(x_train_pos, (x_train_pos.shape[0],250,30), order='F')
    x_test_pos = np.reshape(x_test_pos, (x_test_pos.shape[0],250,30), order='F')

    x_train_vit = np.zeros((x_train_pos.shape[0],250,60))
    x_test_vit = np.zeros((x_test_pos.shape[0],250,60))

    x_train_vit[:,:,0:30] = x_train_pos
    x_train_vit[:,1:250,30:60] = np.diff(x_train_pos, axis=1)

    x_test_vit[:,:,0:30] = x_test_pos
    x_test_vit[:,1:250,30:60] = np.diff(x_test_pos, axis=1)

    del x_train_pos
    del x_test_pos

    ### apply z-score normalisation for position and velocity
    x_mean_pos = np.mean(x_train_vit[:,:,0:30])
    x_sd_pos = np.std(x_train_vit[:,:,0:30])

    x_mean_vel = np.mean(x_train_vit[:,:,30:60])
    x_sd_vel = np.std(x_train_vit[:,:,30:60])

    x_train_vit[:,:,0:30] = (x_train_vit[:,:,0:30] - x_mean_pos) / x_sd_pos
    x_train_vit[:,:,30:60] = (x_train_vit[:,:,30:60] - x_mean_vel) / x_sd_vel

    x_test_vit[:,:,0:30] = (x_test_vit[:,:,0:30] - x_mean_pos) / x_sd_pos
    x_test_vit[:,:,30:60] = (x_test_vit[:,:,30:60] - x_mean_vel) / x_sd_vel


    ### load imu data
    x_train_imu = loadtxt('../data/data_imu_9fold/input_imu_9fold_fold_' + str(fi+1) + '_train.txt')
    x_test_imu = loadtxt('../data/data_imu_9fold/input_imu_9fold_fold_' + str(fi+1) + '_test.txt')

    x_train_imu = np.reshape(x_train_imu, (x_train_imu.shape[0],300,36), order='F')
    x_test_imu = np.reshape(x_test_imu, (x_test_imu.shape[0],300,36), order='F')

    ### resample
    smpl = np.round(np.linspace(1,300,250)).astype(int) - 1
    x_train_imu = x_train_imu[:,smpl,:]
    x_test_imu = x_test_imu[:,smpl,:]

    ### apply z-score normalisation for acceleration and gyro
    ind_acc = np.array([0,1,2,   6,7,8, 12,13,14, 18,19,20, 24,25,26, 30,31,32])
    ind_gyr = np.array([3,4,5, 9,10,11, 15,16,17, 21,22,23, 27,28,29, 33,34,35])

    x_mean_acc = np.mean(x_train_imu[:,:,ind_acc])
    x_sd_acc = np.std(x_train_imu[:,:,ind_acc])

    x_mean_gyr = np.mean(x_train_imu[:,:,ind_gyr])
    x_sd_gyr = np.std(x_train_imu[:,:,ind_gyr])

    x_train_imu[:,:,ind_acc] = (x_train_imu[:,:,ind_acc] - x_mean_acc) / x_sd_acc
    x_train_imu[:,:,ind_gyr] = (x_train_imu[:,:,ind_gyr] - x_mean_gyr) / x_sd_gyr

    x_test_imu[:,:,ind_acc] = (x_test_imu[:,:,ind_acc] - x_mean_acc) / x_sd_acc
    x_test_imu[:,:,ind_gyr] = (x_test_imu[:,:,ind_gyr] - x_mean_gyr) / x_sd_gyr


    ### pool data
    x_test = np.zeros((x_test_mat.shape[0],250,102))
    x_test[:,:,0:6] = x_test_mat
    x_test[:,:,6:66] = x_test_vit
    x_test[:,:,66:102] = x_test_imu

    del x_train_mat
    del x_test_mat
    del x_train_vit
    del x_test_vit
    del x_train_imu
    del x_test_imu


    ### load output data
    y_test = loadtxt('../data/data_mat_9fold/output_mat_9fold_fold_' + str(fi+1) + '_test.txt')


    # print('x_test size:', x_test.shape)
    # print('y_test size:', y_test.shape)

    for ri in range(10):

        model_file = glob.glob( '../results_9fold_grid/fusion_mat_vitpose_imu_cnn_best_model_' + str(ri+1) + '/model_mat_vitpose_imu_9fold_fold_' + str(fi+1) + '*.pt')
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
