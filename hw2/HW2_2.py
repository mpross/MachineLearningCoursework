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
    w = w0*np.ones(d)

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


def regularization(x, y, lamb, delta, w0):
    lambdaMax=np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))

    w = coordDescent(x, y, lambdaMax, 10**-15, 10)


if __name__ == "__main__":
    x, y = generateSynth(n, d, k, sigma)
    lambdaMax = np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))
    print(lambdaMax)

    w = coordDescent(x, y, lambdaMax, 10**-15, 10)
    print(np.nonzero(w)[0].size)