import numpy as np
import matplotlib.pyplot as plt
import random
import time


def loadData():
    # Load a csv of floats:
    X = np.genfromtxt("upvote_data.csv", delimiter=",")
    # Load a text file of integers:
    y = np.loadtxt("upvote_labels.txt", dtype=np.int)
    # Load a text file of strings:
    featureNames = open("upvote_features.txt").read().splitlines()
    return X, y, featureNames


def splitData(x, y, index1, index2):
    trainX = x[:index1]
    valX = x[index1:index2]
    testX = x[index2:]

    trainY = y[:index1]
    valY = y[index1:index2]
    testY = y[index2:]

    return trainX, trainY, valX, valY, testX, testY


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

        if np.max(np.abs(diff)) < delta:
            break

    return w


def regularization(x, y, delta, w0, xVal, yVal):

    lamb = np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))
    w = coordDescent(x, y, lamb, delta, w0)
    valErr = np.dot(np.square(yVal) - np.square(np.dot(w, xVal)), np.square(yVal) - np.square(np.dot(w, xVal)))
    trainErr = np.dot(np.square(y) - np.square(np.dot(w, x)), np.square(y) - np.square(np.dot(w, x)))
    non0 = np.sum(w != 0)

    lambVec = np.array((0, lamb))
    valErrVec = np.array((0, valErr))
    trainErrVec = np.array((0, trainErr))
    non0Vec = np.array((0, non0))

    while np.sum(w != 0) <= 0.75 * x.T[0].size:

        lamb = lambVec[-1]*0.75
        w = coordDescent(x, y, lamb, delta, w)
        valErr = np.dot(np.square(yVal) - np.square(np.dot(w, xVal)), np.square(yVal) - np.square(np.dot(w, xVal)))
        trainErr = np.dot(np.square(y) - np.square(np.dot(w, x)), np.square(y) - np.square(np.dot(w, x)))
        non0 = np.sum(w != 0)

        lambVec = np.append(lambVec, lamb)
        valErrVec = np.append(valErrVec, valErr)
        trainErrVec = np.append(trainErrVec, trainErr)
        non0Vec = np.append(non0Vec, non0)

        print(str(round(np.sum(w != 0)/(0.75 * x.T[0].size)*100, 1))+"% Done")

    minLamb=lambVec[np.argmin(valErrVec[1:])]

    plt.figure(1)
    plt.plot(lambVec[1:], valErrVec[1:]/xVal[1].size, label='Validation Error')
    plt.plot(lambVec[1:], trainErrVec[1:]/x[1].size, label='Train Error')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.legend()
    plt.draw()
    plt.gca().invert_xaxis()
    plt.savefig('ErrorvsLambda.pdf', bbox_inches='tight')


    plt.figure(2)
    plt.plot(lambVec[1:], non0Vec[1:])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('Non-Zero Elements')
    plt.draw()
    plt.savefig('NonzerovsLambdaYelp.pdf', bbox_inches='tight')

    return minLamb


if __name__ == "__main__":
    start=time.time()
    x, y, featureNames = loadData()
    trainX, trainY, valX, valY, testX, testY = splitData(x, np.sqrt(y), 4000, 5000)
    minLamb = regularization(trainX.T, trainY, 0.5, np.zeros(trainX[0].size), valX.T, valY)

    print("Minimum lambda: " + str(minLamb))

    w = coordDescent(trainX.T, trainY, minLamb, 0.5, np.zeros(trainX[0].size))

    print("Validation Error: " + str(
        np.dot(np.square(valY) - np.square(np.dot(w, valX.T)), np.square(valY) - np.square(np.dot(w, valX.T)))/valY.size))
    print("Training Error: " + str(
        np.dot(np.square(trainY) - np.square(np.dot(w, trainX.T)), np.square(trainY) - np.square(np.dot(w, trainX.T)))/trainY.size))
    print("Testing Error: " + str(
        np.dot(np.square(testY) - np.square(np.dot(w, testX.T)), np.square(testY) - np.square(np.dot(w, testX.T)))/testY.size))

    print("Execution Time: "+str(time.time()-start))

    for i in np.flip(np.argsort(abs(w)))[0:11]:
        print("Feature: "+str(featureNames[i]) + " Weight: "+str(w[i]))
    plt.show()