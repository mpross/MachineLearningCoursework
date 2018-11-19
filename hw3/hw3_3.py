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

    k = (1+np.outer(x, z))**d

    return k


def cross_val_poly(x, y):

    d_n = 40
    lamb_n = 40
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

    plt.figure()
    X, Y = np.meshgrid(range(0, d_n), range(0, lamb_n))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.log10(errList))
    plt.xlabel('d')
    plt.ylabel('lambda')
    plt.draw()
    print(bestD, bestLamb)
    return bestD, bestLamb


def boot_poly(x, y, d, lamb, n):

    f = np.zeros((n, x.size))
    for inter in range(n):

        i_cut = np.random.choice(x.size, x.size)
        x_cut = x[i_cut]
        y_cut = y[i_cut]

        print(i_cut)
        print(x_cut)
        print(y_cut)

        try:
            a = regress(k_poly(x_cut, x_cut, d), y_cut, lamb)
            f[inter,:] = np.dot(a, k_poly(x_cut, x_cut, d))

        except np.linalg.LinAlgError:
            print('Err')

    return f

def k_rbf(x, z, gam):

    k = np.exp(-gam*(x**2+z**2-2*np.outer(x, z).T))

    return k.T


def cross_val_rbf(x, y):

    gam_n = 1000
    lamb_n = 10
    errList = np.zeros((lamb_n, gam_n))
    bestErr=10**100
    bestGam=0
    bestLamb=0
    for gam in range(0, gam_n):
        for lamb in range(1, lamb_n):
            err = 0
            for i in range(len(x)):
                try:
                    a = regress(k_rbf(np.delete(x, i), np.delete(x, i), gam*0.001), np.delete(y, i), lamb)
                    f = np.dot(a, k_rbf(np.delete(x, i), x[i], gam*0.001))
                    err += (f-y[i])**2

                except np.linalg.linalg.LinAlgError:
                    err = np.inf

            print(err)
            errList[lamb, gam]=err
            if err<bestErr:
                bestGam = gam*0.001
                bestLamb = lamb
                bestErr = err

    plt.figure()
    X, Y = np.meshgrid(range(0, gam_n), range(0, lamb_n))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.log10(errList))
    plt.draw()
    print(bestGam, bestLamb)
    return bestGam, bestLamb

x, y = generate_data(30)

d, lamb =cross_val_poly(x, y)
a = regress(k_poly(x, x, d), y, lamb)
f = np.dot(a, k_poly(x, x, d))

boot_f = boot_poly(x, y, d, lamb, 10)
print(boot_f)

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, f, '.')
plt.legend('Original Data', 'Estimate')
plt.draw()
#
# gam, lamb =cross_val_rbf(x, y)
# a = regress(k_rbf(x, x, gam), y, lamb)
# f = np.dot(a, k_rbf(x, x, gam))
#
# plt.figure()
# plt.plot(x, y, '.')
# plt.plot(x, f, '.')
# plt.legend('Original Data', 'Estimate')
# plt.draw()

plt.show()

