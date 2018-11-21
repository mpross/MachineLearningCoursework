import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random

X_train = []
X_test = []
labels_test = []
labels_train = []


def load_dataset():
    global X_train, X_test, labels_test, labels_train
    mndata = MNIST('./python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0


def lloyds(x, k):

    index = random.sample(range(x.shape[1]), k)
    cent = x[index]
    label = np.zeros(x.shape[0])

    plt.figure()
    plt.imshow(cent[1].reshape(28, 28))
    plt.pause(0.05)

    for l in range(20):
        err = 0
        for i in range(x.shape[0]):
            dist = np.sum((cent-x[i, :])**2, axis=1)
            label[i] = np.argmin(dist)

            err += min(dist)

        print(err)

        for j in range(k):
            if(x[label == j].size > 0):
                cent[j] = np.sum(x[label == j], axis=0)/(x[label == j].size)
            # else:
            #     cent[j] = np.random.rand(784)

        plt.imshow(cent[0].reshape(28, 28))
        plt.pause(0.05)

    for p in range(k):

        plt.figure()
        plt.imshow(cent[p].reshape(28, 28))
        plt.draw()

    plt.show()

load_dataset()

lloyds(X_train, 10)