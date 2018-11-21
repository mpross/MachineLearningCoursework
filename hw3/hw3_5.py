import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import svds


def read_data():

    test = dok_matrix((24983, 100), dtype=np.float32)
    train = dok_matrix((24983, 100), dtype=np.float32)
    val = dok_matrix((24983, 100), dtype=np.float32)

    f = open('test.txt', 'r')
    for line in f.readlines():
        nums = line.split(',')
        test[int(nums[0])-1, int(nums[1])-1] = float(nums[2])

    f.close()

    f = open('train.txt', 'r')
    for line in f.readlines():
        nums = line.split(',')
        ran = np.random.uniform(0, 10, 1)
        if ran <= 8:
            train[int(nums[0])-1, int(nums[1])-1] = float(nums[2])
        if ran > 8:
            val[int(nums[0]) - 1, int(nums[1]) - 1] = float(nums[2])
    f.close()

    return test, train, val


def av_user_estimator(R):

    ons = np.ones((1, R.shape[0]))

    v = ons*R/ons.size
    return v


def svd_series(test, train):

    R_test=test.todense()
    R_train = train.todense()

    dList = [1, 2, 5, 10, 20, 50]

    train_err_mse = np.zeros(1)
    test_err_mse = np.zeros(1)
    train_err_mae = np.zeros(1)
    test_err_mae = np.zeros(1)

    N_train = np.sum(R_train != 0, axis=1)
    N_test = np.sum(R_test != 0, axis=1)

    for d in dList:
        u, s, vt = svds(R_train, k=d)

        train_err_mse = np.append(train_err_mse, np.sum(np.sum(np.square(np.dot(u, vt) - R_train)))/train.nnz)
        test_err_mse = np.append(test_err_mse, np.sum(np.sum(np.square(np.dot(u, vt) - R_test)))/test.nnz)

        train_err_mae = np.append(train_err_mae, np.sum(np.sum(np.divide(np.absolute(np.dot(u, vt) - R_train), N_train))) / R_train.shape[0])
        test_err_mae = np.append(test_err_mae, np.sum(np.sum(np.divide(np.absolute(np.dot(u, vt) - R_test), N_test))) / R_test.shape[0])

    plt.figure()
    plt.plot(dList, train_err_mse[1:])
    plt.plot(dList, test_err_mse[1:])
    plt.legend(('Training Error', 'Testing Error'))
    plt.title('Mean Squared Error')
    plt.draw()

    plt.figure()
    plt.plot(dList, train_err_mae[1:])
    plt.plot(dList, test_err_mae[1:])
    plt.legend(('Training Error', 'Testing Error'))
    plt.title('Mean Absolute Error')
    plt.draw()


def loss_min(train, d, lamb):

    u = np.random.randn(train.shape[0], d)

    lastErr = 10**100
    for i in range(10**4):
        if i % 2 == 0:
            v = np.linalg.solve(np.dot(u.T, u) + lamb, u.T*train)

        else:
            u = np.linalg.solve(np.dot(v, v.T) + lamb, (train*v.T).T).T

        if np.sum(np.abs(np.dot(u, v) - train)**2) > lastErr:
            break

        lastErr = np.sum(np.abs(np.dot(u, v) - train)**2)

        print(i, lastErr)

    return u, v

def loss_reg(test, train, val, d):

    lamb_Vec = range(10)
    err = np.zeros(1)
    for lamb in lamb_Vec:
        u, v = loss_min(train, d, lamb)
        err = np.append(err, np.sum(np.dot((np.dot(u, v) - train).T, (np.dot(u, v) - train))))

    plt.plot(lamb_Vec, err[1:])
    plt.draw()


test, train, val = read_data()

# v = av_user_estimator(train)
# ons = np.ones((1, test.shape[0]))
# err = np.sum(ons.size * v.T * v - v.T * ons * test - test.T * ons.T * v + test.T * test) / test.nnz
# print('Average User Error: '+str(err))
#
# svd_series(test, train)

loss_reg(test, train, val, 3)

plt.show()