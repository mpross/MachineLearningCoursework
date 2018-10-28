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
    d = x[0].size

    a = np.zeros(d)
    c = np.zeros(d)
    w = np.zeros(d)
    w[0] = w0

    i = 0
    while np.abs(w[i-1]-w[i]) > delta:

        b = sum(y-np.dot(w[i], x))/y.size
        for k in range(1, d):
            a[k] = 2*np.dot(x, x)
            c[k] = 2*np.dot(x, y - (b + np.dot(np.concatenate((w[0:k-1], w[k:])), np.concatenate((x[0:k-1], x[k:])))))
            if c[k] < -lamb:
               w[k] = (c[k]+lamb)/a[k]
            elif c[k] > lamb:
                w[k] = (c[k] - lamb) / a[k]
            else:
                w[k] = 0


if __name__ == "__main__":
    x, y = generateSynth(n, d, k, sigma)

    print(y.size)
