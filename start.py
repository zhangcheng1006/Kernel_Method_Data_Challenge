import numpy as np
import pandas as pd
from QDA import *
from KernelLogisticRegression import *
from SVM import *
from DimensionReduction import *
from LogisticRegression import *
from kNN import *

# np.random.seed(2019)

total_count = [0, 0, 0]
true_count = [0, 0, 0]
sigmas = [1.0, 1.0, 1.0]
output_file = open('./Yte.csv', "w")
output_file.write("Id,Bound\n")
idx = 0
for i in range(3):
    print("treating dataset {}".format(i))
    # loading dataset
    train_X = pd.read_csv('./data/Xtr{}.csv'.format(i), sep=',')['seq'].values
    test_X = pd.read_csv('./data/Xte{}.csv'.format(i), sep=',')['seq'].values
    train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i))['Bound'].values

    train_Y[train_Y == 0] = -1 # change the label 0 to -1

    ###################### SVM (Spectrum Kernel) ##########################
    # one-hold-out cross validation
    data = np.hstack((np.reshape(train_X, (-1, 1)), np.reshape(train_Y, (-1, 1))))
    np.random.shuffle(data)
    one_fold = data.shape[0] / 5
    cv_valid = data[0:int(one_fold)]
    cv_train = data[int(one_fold):]
    print(cv_train.shape)
    print(cv_valid.shape)
    cv_train_X = cv_train[:, :-1].flatten()
    cv_train_Y = cv_train[:, -1:].flatten()
    cv_valid_X = cv_valid[:, :-1].flatten()
    cv_valid_Y = cv_valid[:, -1:].flatten()
    print(cv_train_X.shape)
    print(cv_valid_X.shape)

    K_train, K_valid = compute_kernel_matrix(cv_train_X, cv_valid_X, kernel='spectrum', k=10)
    K_train = np.nan_to_num(K_train)
    K_valid = np.nan_to_num(K_valid)

    K_train = empirical_kernel_map(K_train)
    K_valid = np.dot(K_valid.T, K_train).T

    alpha = SVM_train(K_train, cv_train_Y, lamda=1.0)
    pred_train = SVM_predict(K_train, alpha=alpha)
    pred_cv = SVM_predict(K_valid, alpha=alpha)
    total_train = data.shape[0] - one_fold
    total_valid = one_fold
    cv_train_Y[cv_train_Y == -1] = 0
    cv_valid_Y[cv_valid_Y == -1] = 0
    pred_train_true = sum(pred_train.flatten() == cv_train_Y.flatten())
    pred_valid_true = sum(pred_cv.flatten() == cv_valid_Y.flatten())
    print("Cross validation {}: accuracy on cv_train {}".format(1, pred_train_true / total_train))
    print("Cross validation {}: accuracy on cv_valid {}".format(1, pred_valid_true / total_valid))

    K_train, K_test = compute_kernel_matrix(train_X, test_X, kernel='spectrum', k=10)
    K_train = np.nan_to_num(K_train)
    K_test = np.nan_to_num(K_test)
    
    K_train = empirical_kernel_map(K_train)
    K_test = np.dot(K_test.T, K_train).T

    alpha_res = SVM_train(K_train, train_Y, lamda=1.0)
    pred_train = SVM_predict(K_train, alpha=alpha_res)
    total_count[i] = train_X.shape[0]
    train_Y[train_Y == -1] = 0
    true_count[i] = sum(pred_train.flatten() == train_Y)
    print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))

    pred_test = SVM_predict(K_test, alpha=alpha_res)



    ############################## using fasttext pretrained embedding ###############################
    # train_X = pd.read_csv('./ft2vec/Xtr{}_ft2vec.csv'.format(i), header=None, sep=' ').values
    # test_X = pd.read_csv('./ft2vec/Xte{}_ft2vec.csv'.format(i), header=None, sep=' ').values
    # train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i))['Bound'].values

    # ############# kernel PCA ################
    # transformed = kernelPCA(np.concatenate((train_X, test_X), axis=0), kernel='RBF', sigma=sigmas[i])

    # train_X_transformed = transformed[0:train_X.shape[0]]
    # test_X_transformed = transformed[train_X.shape[0]:]

    # assert train_X_transformed.shape[0] == 2000
    # assert test_X_transformed.shape[0] == 1000

    # ###################### Logistic Regression ##########################
    # w, b = Logistic_train(train_X_transformed, train_Y)
    # pred_train = Logistic_predict(train_X_transformed, w, b)
    # total_count[i] = train_X.shape[0]
    # true_count[i] = sum(pred_train.flatten() == train_Y)
    # print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))
    # pred_test = Logistic_predict(test_X_transformed, w, b)



    # # ################################# kNN does not work ###################################
    # # pred_train = kNN_predict(train_X, train_X, train_Y, k=5)
    # # total_count[i] = train_X.shape[0]
    # # true_count[i] = sum(pred_train.flatten() == train_Y)
    # # print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))

    # # pred_test = kNN_predict(test_X, train_X, train_Y, k=5)

    

    # ####################### Kernel Logistic Regression on the given embedding ####################
    # train_X = pd.read_csv('./data/Xtr{}_mat100.csv'.format(i), header=None, sep=' ').values
    # test_X = pd.read_csv('./data/Xte{}_mat100.csv'.format(i), header=None, sep=' ').values
    # train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i))['Bound'].values

    # # train_X = pd.read_csv('./ft2vec/Xtr{}_ft2vec.csv'.format(i), header=None, sep=' ').values
    # # test_X = pd.read_csv('./ft2vec/Xte{}_ft2vec.csv'.format(i), header=None, sep=' ').values

    # # perform PCA before fitting to the classifier
    # # train_X, mean, pca_mat = PCA(train_X)
    # # test_X = np.dot(test_X - mean.reshape((1, -1)), pca_mat)

    # train_Y[train_Y == 0] = -1

    # ##################### one-hold-out cross validation ##########################
    # data = np.concatenate((train_X, train_Y.reshape((-1, 1))), axis=1)
    # np.random.shuffle(data)
    # one_fold = data.shape[0] / 5
    # cv_valid = data[0:int(one_fold)]
    # cv_train = data[int(one_fold):]
    # print(cv_train.shape)
    # print(cv_valid.shape)
    # cv_train_X = cv_train[:, :-1]
    # cv_train_Y = cv_train[:, -1:]
    # cv_valid_X = cv_valid[:, :-1]
    # cv_valid_Y = cv_valid[:, -1:]
    # alpha = SVM_train(cv_train_X, cv_train_Y, kernel=None, sigma=sigmas[i], lamda=1.0)
    # pred_train = SVM_predict(cv_train_X, cv_train_X, alpha=alpha, kernel=None, sigma=sigmas[i])
    # pred_cv = SVM_predict(cv_valid_X, cv_train_X, alpha=alpha, kernel=None, sigma=sigmas[i])
    # total_train = data.shape[0] - one_fold
    # total_valid = one_fold
    # cv_train_Y[cv_train_Y == -1] = 0
    # cv_valid_Y[cv_valid_Y == -1] = 0
    # pred_train_true = sum(pred_train.flatten() == cv_train_Y.flatten())
    # pred_valid_true = sum(pred_cv.flatten() == cv_valid_Y.flatten())
    # print("Cross validation {}: accuracy on cv_train {}".format(1, pred_train_true / total_train))
    # print("Cross validation {}: accuracy on cv_valid {}".format(1, pred_valid_true / total_valid))
    
    # alpha_res = KLR_train(train_X, train_Y, kernel='RBF', sigma=sigmas[i], lamda=1.0)
    # pred_train = KLR_predict(train_X, train_X, alpha=alpha_res, kernel='RBF', sigma=sigmas[i])
    # total_count[i] = train_X.shape[0]
    # train_Y[train_Y == -1] = 0
    # true_count[i] = sum(pred_train.flatten() == train_Y)
    # print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))
    # pred_test = KLR_predict(test_X, train_X, alpha=alpha_res, kernel=None, sigma=sigmas[i])


    print('start to writing results')
    for pred in pred_test:
        output_file.write('{},{}\n'.format(int(idx), int(pred)))
        idx += 1

output_file.close()
