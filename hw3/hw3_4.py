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

    # plt.figure()
    # plt.imshow(cent[1].reshape(28, 28))
    # plt.pause(0.05)
    errList = np.zeros(1)
    for l in range(30):
        err = 0
        for i in range(x.shape[0]):
            dist = np.sum((cent-x[i, :])**2, axis=1)
            label[i] = np.argmin(dist)

            err += min(dist)

        errList = np.append(errList, err)

        for j in range(k):
            if (x[label == j].size) > 0:
                cent[j] = np.sum(x[label == j], axis=0)/(x[label == j].size)

        # plt.imshow(cent[0].reshape(28, 28))
        # plt.pause(0.05)

    plt.figure()
    plt.plot(range(3,30), errList[4:])
    plt.xlabel('Iteration')
    plt.ylabel('Distance to center')
    plt.yscale('log')
    plt.title('k='+str(k))
    plt.savefig('Figures/error' + str(k) + '.pdf')

    for p in range(k):

        plt.figure()
        plt.imshow(cent[p].reshape(28, 28))
        plt.title('k=' + str(k))
        plt.set_cmap('gray_r')
        plt.savefig('Figures/means'+str(k)+'_'+str(p)+'.pdf')


def kmeanspp(x, k):

    index = random.sample(range(x.shape[1]), 1)
    cent = x[index]

    for o in range(1,k):
        prob = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            dist = np.sum((cent - x[i, :]) ** 2, axis=1)
            prob[i] = np.min(dist) ** 2 / np.sum(dist ** 2)

        prob /= np.sum(prob)

        cent = np.concatenate((cent, x[np.random.choice(range(x.shape[0]), 1, p=prob)]))

    label = np.zeros(x.shape[0])

    # plt.figure()
    # plt.imshow(cent[1].reshape(28, 28))
    # plt.pause(0.05)
    errList = np.zeros(1)
    for l in range(30):
        err = 0
        for i in range(x.shape[0]):
            dist = np.sum((cent-x[i, :])**2, axis=1)
            label[i] = np.argmin(dist)
            prob[i] = np.min(dist)**2/np.sum(dist**2)
            err += min(dist)

        errList = np.append(errList, err)

        for j in range(k):
            if (x[label == j].size) > 0:
                cent[j] = np.sum(x[label == j], axis=0)/(x[label == j].size)

        # plt.imshow(cent[0].reshape(28, 28))
        # plt.pause(0.05)

    plt.figure()
    plt.plot(range(3,30), errList[4:])
    plt.xlabel('Iteration')
    plt.ylabel('Distance to center')
    plt.yscale('log')
    plt.title('k='+str(k))
    plt.savefig('Figures/error++' + str(k) + '.pdf')

    for p in range(k):

        plt.figure()
        plt.imshow(cent[p].reshape(28, 28))
        plt.title('k=' + str(k))
        plt.set_cmap('gray_r')
        plt.savefig('Figures/means++'+str(k)+'_'+str(p)+'.pdf')


load_dataset()

kList=[5, 10, 20]
for k in kList:
    lloyds(X_train, k)
    kmeanspp(X_train, k)