from data_io import loadtxt, fold_indices, run_count
import torch
from torch_utils import predict, load_model


import numpy as np
from sklearn.metrics import confusion_matrix

import glob



for fi in fold_indices():

    print('Fold: ', fi+1)

    file = open('prediction_mat_cnn_best_model_fold_' + str(fi+1) + '.txt','a+t')

    x_test = loadtxt('../data/data_mat_9fold/input_mat_9fold_fold_' + str(fi+1) + '_test.txt')
    x_test = np.reshape(x_test, (x_test.shape[0],500,6), order='F')
    y_test = loadtxt('../data/data_mat_9fold/output_mat_9fold_fold_' + str(fi+1) + '_test.txt')

    # print('x_test size:', x_test.shape)
    # print('y_test size:', y_test.shape)

    for ri in range(10):

        model_file = glob.glob( '../results_9fold_grid/mat_cnn_best_model_' + str(ri+1) + '/model_mat_9fold_fold_' + str(fi+1) + '*.pt' )
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
