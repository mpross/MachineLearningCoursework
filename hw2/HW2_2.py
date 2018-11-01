import numpy as np
import matplotlib.pyplot as plt
import random
import time

n=500
d=1000
k=100
sigma=1

def generateSynth(n, d, k, sigma):
    x = sigma*np.random.randn(d, n)
    j = np.array(range(1, k+1))
    w = np.concatenate((j/k, np.zeros(d-k)))
    y = np.dot(w, x)+np.random.randn(n)
    return x, y

def coordDescent(x, y, lamb, delta, w0):
    d = x.T[0].size
    n=x[0].size

    a = np.zeros(d)
    c = np.zeros(d)
    diff = np.zeros(d)
    w = w0

    while True:
        b = np.sum(y-np.dot(w, x))/y.size
        for k in range(1, d):
            a[k] = 2 * np.dot(x[k], x[k])
            c[k] = 2 * np.dot(x[k], (y - (b + np.dot(w, x)- np.dot(w[k], x[k]))))
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


def regularization(x, y, delta, w0):

    lamb = np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))
    w = coordDescent(x, y, lamb, delta, w0)
    non0 = np.nonzero(w)[0].size

    non0Vec = np.array((0, non0))
    lambVec = np.array((0, lamb))

    while np.nonzero(w)[0].size <= 999:

        lamb = lambVec[-1]/1.5
        w = coordDescent(x, y, lamb, delta, w)
        non0 = np.nonzero(w)[0].size

        non0Vec = np.append(non0Vec, non0)
        lambVec = np.append(lambVec, lamb)

    plt.plot(lambVec[1:], non0Vec[1:])
    plt.xscale('log')
    plt.show()

    return w




if __name__ == "__main__":
    x, y = generateSynth(n, d, k, sigma)

    w = regularization(x, y, 0.1, 10*np.ones(d))