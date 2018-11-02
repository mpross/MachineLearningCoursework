import numpy as np
import matplotlib.pyplot as plt
import random
import time

n=4000

def loadData():
    # Load a csv of floats:
    X = np.genfromtxt("upvote_data.csv", delimiter=",")
    # Load a text file of integers:
    y = np.loadtxt("upvote_labels.txt", dtype=np.int)
    # Load a text file of strings:
    featureNames = open("upvote_features.txt").read().splitlines()
    return X, y, featureNames


def splitData(x, y, index):
    xmajor = x[:index]
    xminor = x[index:]
    ymajor = y[:index]
    yminor = y[index:]

    return xmajor, ymajor, xminor, yminor


def coordDescent(x, y, lamb, delta, w0):
    d = x.T[0].size

    a = np.zeros(d)
    c = np.zeros(d)
    diff = np.zeros(d)
    w = w0

    while True:
        b = np.sum(y-np.dot(w, x))/y.size
        for k in range(1, d):
            a[k] = 2 * np.dot(x[k], x[k])
            wCut = np.copy(w)
            wCut[k] = 0
            c[k] = 2 * np.dot(x[k], (y - (b + np.dot(wCut, x))))
            if c[k] < -lamb:
                diff[k] = w[k]-(c[k]+lamb)/a[k]
                w[k] = (c[k]+lamb)/a[k]
            elif c[k] > lamb:
                diff[k] = w[k] - (c[k] - lamb) / a[k]
                w[k] = (c[k] - lamb) / a[k]
            else:
                diff[k] = 0
                w[k] = 0


        print(np.dot(y-np.dot(x.T, w)-b, y-np.dot(x.T, w)-b))
        print(np.max(np.abs(diff)))

        if np.max(np.abs(diff)) < delta:
            break

    return w


def regularization(x, y, delta, w0, xTrain, yTrain):

    lamb = np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))
    w = coordDescent(x, y, lamb, delta, w0)
    valErr = np.dot(np.square(yTrain) - np.square(np.dot(w, xTrain)), np.square(yTrain) - np.square(np.dot(w, xTrain)))
    testErr = np.dot(np.square(y) - np.square(np.dot(w, x)), np.square(y) - np.square(np.dot(w, x)))

    lambVec = np.array((0, lamb))
    valErrVec = np.array((0, valErr))
    testErrVec = np.array((0, testErr))

    while np.sum(w != 0) <= 0.75 * x.T[0].size:

        lamb = lambVec[-1]*0.75
        w = coordDescent(x, y, lamb, delta, w)
        valErr = np.dot(np.square(yTrain) - np.square(np.dot(w, xTrain)), np.square(yTrain) - np.square(np.dot(w, xTrain)))
        testErr = np.dot(np.square(y) - np.square(np.dot(w, x)), np.square(y) - np.square(np.dot(w, x)))

        lambVec = np.append(lambVec, lamb)
        valErrVec = np.append(valErrVec, valErr)
        testErrVec = np.append(testErrVec, testErr)

    print(valErrVec[1:].size)
    print(testErrVec[1:].size)

    plt.figure(1)
    plt.plot(lambVec[1:], valErrVec[1:]/x[1].size, label='Validation Error')
    plt.plot(lambVec[1:], testErrVec[1:]/xTrain[1].size, label='Test Error')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.legend()
    plt.draw()
    plt.savefig('ErrorvsLambda.pdf', bbox_inches='tight')
    plt.gca().invert_xaxis()

    return w


if __name__ == "__main__":
    start=time.time()
    x, y, featureNames = loadData()

    trainX, trainY, valX, valY = splitData(x, np.sqrt(y), n)

    w = regularization(trainX.T, trainY, 0.5, np.zeros(trainX[0].size), valX.T, valY)

    print("Execution Time: "+str(time.time()-start))
    plt.show()