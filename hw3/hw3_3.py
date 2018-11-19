import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n):

    x = np.random.uniform(0, 1, n)
    y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2)) + np.random.randn(n)

    return x, y


def regress(K, y, lamb):

    a = np.linalg.solve(np.dot(K.T, K) + lamb*K, np.dot(K.T, y))

    return a


def k_poly(x, z, d):

    k = 1+np.outer(x, z)**d

    return k


def cross_val_poly(x, y):

    d_n = 10
    lamb_n=20
    errList = np.zeros((lamb_n, d_n))
    bestErr=10**100
    bestD=0
    bestLamb=0
    for d in range(1, d_n):
        for lamb in range(1, lamb_n):
            err = 0
            for i in range(len(x)):
                try:
                    a = regress(k_poly(np.delete(x, i), np.delete(x, i), d), np.delete(y, i), lamb)
                    f = np.dot(a, k_poly(np.delete(x, i), x[i], d))
                    err += (f-y[i])**2

                except np.linalg.linalg.LinAlgError:
                    err = np.inf

            print(err)
            errList[lamb, d]=err
            if err<bestErr:
                bestD = d
                bestLamb = lamb
                bestErr = err

    fig = plt.figure()
    X, Y = np.meshgrid(range(1, lamb_n), range(1, d_n))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, errList)
    fig.show()
    print(bestD, bestLamb)
    return bestD, bestLamb

x, y = generate_data(30)
d, lamb =cross_val_poly(x, y)

