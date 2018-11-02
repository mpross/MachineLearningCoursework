import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random
import time

from numpy.core.multiarray import ndarray

X_train = []
X_test = []
labels_train = []
labels_test = []


def load_dataset():
    global X_train, X_test, labels_test, labels_train
    mndata = MNIST('./python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0


def one_hot(inpt):
    output = np.zeros((inpt.size, 10))
    for i in range(len(inpt)):
        vec = np.zeros(10)
        for j in range(10):
            vec[j] = int(int(inpt[i]) == j)
            output[i] = vec
    return output


def sort(x, inlist):
    xOut=x[np.where(np.logical_or(inlist==2 , inlist==7))]
    outList = inlist[np.where(np.logical_or(inlist == 2, inlist == 7))]

    return xOut, outList


def mu(x, y, b, w):
    return 1/(1+np.exp(-y*(b+np.dot(x,w))))

def J(x, y, b, w, lamb):
    sum=0
    for i in range(y.size):
        sum+=np.log(mu(x[i], y[i], b, w))

    sum/=y.size
    sum+=lamb*np.dot(w,w)
    return sum

def gradDescent(x, y, lamb, w0):
    w=w0
    for i in range(100):
        print(x.shape)
        print(w.shape)
        b=np.sum(y-np.dot(x,w))
        w= 0.1*np.dot(np.dot(y,x), (1-mu(x, y, b, w)))+(1-lamb)*w
        print(np.dot(w,w))

if __name__ == "__main__":
    load_dataset()

    xTrain, trainLabels=sort(X_train, labels_train)
    y_train = one_hot(trainLabels)

    xTest, testLabels = sort(X_test, labels_test)
    y_test = one_hot(testLabels)

    gradDescent(xTrain.T,y_train,0.1,np.zeros(y_train.shape))
