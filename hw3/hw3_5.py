import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix


def read_data():

    test = dok_matrix((24983, 100), dtype=np.float32)
    train = dok_matrix((24983, 100), dtype=np.float32)

    f = open('test.txt', 'r')
    for line in f.readlines():
        nums = line.split(',')
        test[int(nums[0])-1, int(nums[1])-1] = float(nums[2])

    f.close()

    f = open('train.txt', 'r')
    for line in f.readlines():
        nums = line.split(',')
        train[int(nums[0])-1, int(nums[1])-1] = float(nums[2])
    f.close()

    return test, train


def av_user_estimator(R, lamb):

    ons = np.ones(24983)
    v = np.dot(ons, R)/(ons.size+lamb)
    return v


test, train = read_data()

v = av_user_estimator(train, 0)
err = np.dot((np.dot(np.ones(24983).T,v)-test).T,np.dot(np.ones(24983).T,v)-test)