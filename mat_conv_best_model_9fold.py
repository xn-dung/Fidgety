from data_io import loadtxt, fold_indices, run_count
import torch
from torch import nn
from torch_utils import initialize_model, fit, evaluate, save_model, record_result, EarlyStopping, ChannelsLastFlatten



import numpy as np
import matplotlib.pyplot as plt


### PARAMETERS

# number of kernels and width of kernels in CONV layer
kn1 = 8
kw1 = 13

kn2 = 64
kw2 = 17

kn3 = 16
kw3 = 25

# number of neurons in FC layers
fc1 = 256
fc2 = 0

# output file prefix
out_file_pref = 'res_mat_9fold'

# model file prefix
model_file_pref = 'model_mat_9fold'

### END PARAMETERS


for fi in fold_indices():

    print('Prepare data for training ...')

    x_train = loadtxt('../data/data_mat_9fold/input_mat_9fold_fold_' + str(fi+1) + '_train.txt')
    x_test = loadtxt('../data/data_mat_9fold/input_mat_9fold_fold_' + str(fi+1) + '_test.txt')

    y_train = loadtxt('../data/data_mat_9fold/output_mat_9fold_fold_' + str(fi+1) + '_train.txt')
    y_test = loadtxt('../data/data_mat_9fold/output_mat_9fold_fold_' + str(fi+1) + '_test.txt')

    x_train = np.reshape(x_train, (x_train.shape[0],500,6), order='F')
    x_test = np.reshape(x_test, (x_test.shape[0],500,6), order='F')

    print('x_train size:', x_train.shape)
    print('y_train size:', y_train.shape)
    print('x_test size:', x_test.shape)
    print('y_test size:', y_test.shape)


    # ind = np.random.permutation(x_train.shape[0])
    ind = loadtxt('../data/random_indices_9fold/random_indices_train_9fold_fold_' + str(fi+1) + '.txt', dtype = 'int')
    ind = ind-1
    print(ind)
    x_train = x_train[ind,:,:]
    y_train = y_train[ind]

    x_val = x_train[int(7*x_train.shape[0]/8):x_train.shape[0],:,:]
    y_val = y_train[int(7*y_train.shape[0]/8):y_train.shape[0]]
    print('x_val size:', x_val.shape)
    print('y_val size:', y_val.shape)

    res = np.zeros((run_count(),3))
    min_val_score = np.inf
    max_val_score = 0

    for ri in range(run_count()):
        print('Fold: ', fi+1, ', Run: ', ri+1)

        model = nn.Sequential()
        channels = 1

        model.append(nn.Conv2d(channels, kn1, (kw1, 1), padding="same"))
        model.append(nn.ReLU())
        channels = kn1
        model.append(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01))
        model.append(nn.Dropout(0.2))

        if kn2>0:
            model.append(nn.Conv2d(channels, kn2, (kw2, 1), padding="same"))
            model.append(nn.ReLU())
            channels = kn2
            model.append(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01))
            model.append(nn.Dropout(0.2))

        if kn3>0:
            model.append(nn.Conv2d(channels, kn3, (kw3, 1), padding="same"))
            model.append(nn.ReLU())
            channels = kn3
            model.append(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01))
            model.append(nn.Dropout(0.2))

        model.append(nn.AvgPool2d((500, 1), stride=(500, 1)))

        # Preserve the author's channels-last flatten order.
        model.append(ChannelsLastFlatten())

        units = fc1
        model.append(nn.LazyLinear(units))
        model.append(nn.ReLU())
        model.append(nn.BatchNorm1d(units, eps=1e-3, momentum=0.01))
        model.append(nn.Dropout(0.2))

        if fc2>0:
            units = fc2
            model.append(nn.LazyLinear(units))
            model.append(nn.ReLU())
            model.append(nn.BatchNorm1d(units, eps=1e-3, momentum=0.01))
            model.append(nn.Dropout(0.2))

        units = 1
        model.append(nn.LazyLinear(units))

        print(model)


        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', restore_best_weights=True, verbose=1)]

        initialize_model(model, x_train[:1])

        train_hist = fit(model, x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1), y_train, batch_size=4, validation_split=1/8, epochs=1000, verbose=1, callbacks=callbacks)

        val_loss = np.min(train_hist.history['val_loss'])
        print("best val_loss = ", val_loss)

        print("Evaluate model on validation data ...")
        val_score = evaluate(model, x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1), y_val, verbose=1)
        print("val_accuracy = ", val_score[1])

        print("Evaluate model on test data ...")
        test_score = evaluate(model, x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1), y_test, verbose=1)
        print("test_accuracy = ", test_score[1])

        res[ri,0] = val_loss
        res[ri,1] = val_score[1]
        res[ri,2] = test_score[1]

        record_result(model, x_test, y_test, fi+1, ri+1, train_hist, val_score, test_score)

        if val_loss < min_val_score:
           min_val_score = val_loss
           save_model(model, model_file_pref + '_fold_' + str(fi+1) +
           '_kn1_' + str(kn1) + '_kw1_' + str(kw1) +
           '_kn2_' + str(kn2) + '_kw2_' + str(kw2) +
           '_kn3_' + str(kn3) + '_kw3_' + str(kw3) +
           '_fc1_' + str(fc1) + '_fc2_' + str(fc2) + '.pt')

        del model, train_hist

    fname = (out_file_pref + '_fold_' + str(fi+1) +
    '_kn1_' + str(kn1) + '_kw1_' + str(kw1) +
    '_kn2_' + str(kn2) + '_kw2_' + str(kw2) +
    '_kn3_' + str(kn3) + '_kw3_' + str(kw3) +
    '_fc1_' + str(fc1) + '_fc2_' + str(fc2) + '.txt')
    np.savetxt(fname, res, fmt = '%12.6f')
