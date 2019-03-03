import numpy as np
import pandas as pd
from QDA import *
from KernelLogisticRegression import *
from SVM import *

np.random.seed(2019)

total_count = [0, 0, 0]
true_count = [0, 0, 0]
sigmas = [0.1, 0.1, 0.1]
output_file = open('./output/klr_dna.csv', "w")
output_file.write("Id,Bound\n")
idx = 0
for i in range(3):
    print("treating dataset {}".format(i))
    # train_X = pd.read_csv('./data/Xtr{}_mat100.csv'.format(i), header=None, sep=' ').values
    # test_X = pd.read_csv('./data/Xte{}_mat100.csv'.format(i), header=None, sep=' ').values
    # train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i))['Bound'].values

    train_X = pd.read_csv('./dna2vec/Xtr{}_dna2vec.csv'.format(i), header=None, sep=' ').values
    test_X = pd.read_csv('./dna2vec/Xte{}_dna2vec.csv'.format(i), header=None, sep=' ').values
    train_Y = pd.read_csv('./data/Ytr{}.csv'.format(i))['Bound'].values
    train_Y[train_Y == 0] = -1

    ####################### Kernel Logistic Regression ####################
    # # 5-fold cross validation
    # data = np.concatenate((train_X, train_Y.reshape((-1, 1))), axis=1)
    # np.random.shuffle(data)
    # one_fold = data.shape[0] / 5
    # for j in range(5):
    #     cv_valid = data[int(j*one_fold):int((j+1)*one_fold)]
    #     cv_train = np.concatenate((data[0:int(j*one_fold)], data[int((j+1)*one_fold):]), axis=0)
    #     print(cv_train.shape)
    #     print(cv_valid.shape)
    #     cv_train_X = cv_train[:, :-1]
    #     cv_train_Y = cv_train[:, -1:]
    #     cv_valid_X = cv_valid[:, :-1]
    #     cv_valid_Y = cv_valid[:, -1:]
    #     alpha = KLR_train(cv_train_X, cv_train_Y, kernel="RBF", sigma=sigmas[i], lamda=0.5)
    #     pred_train = KLR_predict(cv_train_X, cv_train_X, alpha=alpha, kernel="RBF", sigma=sigmas[i])
    #     pred_cv = KLR_predict(cv_valid_X, cv_train_X, alpha=alpha, kernel="RBF", sigma=sigmas[i])
    #     total_train = data.shape[0] - one_fold
    #     total_valid = one_fold
    #     cv_train_Y[cv_train_Y == -1] = 0
    #     cv_valid_Y[cv_valid_Y == -1] = 0
    #     pred_train_true = sum(pred_train.flatten() == cv_train_Y.flatten())
    #     pred_valid_true = sum(pred_cv.flatten() == cv_valid_Y.flatten())
    #     print("Cross validation {}: accuracy on cv_train {}".format(j, pred_train_true / total_train))
    #     print("Cross validation {}: accuracy on cv_valid {}".format(j, pred_valid_true / total_valid))
    
    alpha_res = KLR_train(train_X, train_Y, kernel="RBF", sigma=sigmas[i], lamda=0.5)
    pred_train = KLR_predict(train_X, train_X, alpha=alpha_res, kernel="RBF", sigma=sigmas[i])
    total_count[i] = train_X.shape[0]
    train_Y[train_Y == -1] = 0
    true_count[i] = sum(pred_train.flatten() == train_Y)
    print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))
    pred_test = KLR_predict(test_X, train_X, alpha=alpha_res, kernel="RBF", sigma=sigmas[i])


    # ###################### SVM (Kernel) ##########################
    # alpha_res = SVM_train(train_X, train_Y, kernel="RBF", sigma=sigmas[i], lamda=1.0)
    # pred_train = SVM_predict(train_X, train_X, alpha=alpha_res, kernel="RBF", sigma=sigmas[i])
    # total_count[i] = train_X.shape[0]
    # train_Y[train_Y == -1] = 0
    # true_count[i] = sum(pred_train.flatten() == train_Y)
    # print("Accuracy on dataset {}: {}".format(i, true_count[i] / total_count[i]))

    # pred_test = SVM_predict(test_X, train_X, alpha=alpha_res, kernel="RBF", sigma=sigmas[i])

    print('start to writing results')
    for pred in pred_test:
#         rslt.loc[idx] = [idx, pred]
        output_file.write('{},{}\n'.format(int(idx), int(pred)))
        idx += 1
# rslt = rslt.astype(int)
output_file.close()
# print("Accuracy on training set: " + str(true_count / total_count))