import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n):

    x = np.random.uniform(0, 1, n)
    y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2)) + np.random.randn(n)

    return x, y


def regress(K, y, lamb):

    try:
        a = np.linalg.solve(np.dot(K.T, K) + lamb*K, np.dot(K.T, y))

    except np.linalg.LinAlgError:
        a = np.zeros(K.shape[0])
        print('Err')

    return a


def k_poly(x, z, d):

    k = (1 + np.outer(x, z)) ** d

    return k


def cross_val_poly(x, y):

    d_n = 50
    lamb_n = 20
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
            errList[lamb, d] = err
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
    return bestD, bestLamb


def boot_poly(x, y, d, lamb, n):

    f = np.zeros((n, x.size))
    for inter in range(n):

        i_cut = np.random.choice(x.size, x.size)
        x_cut = x[i_cut]
        y_cut = y[i_cut]
        try:
            a = regress(k_poly(x_cut, x_cut, d), y_cut, lamb)
            f[inter] = np.dot(a, k_poly(x_cut, x_cut, d))

        except np.linalg.LinAlgError:
            print('Err')

    return f

def k_rbf(x, z, gam):

    k=np.zeros((x.size,z.size))
    for i in range(x.size):
        for j in range(z.size):
            if(z.size>1):
                k[i,j] = np.exp(-gam*np.dot(x[i]-z[j], x[i]-z[j]))
            else:
                k[i]=np.exp(-gam*np.sum(np.dot(x[i]-z, x[i]-z)))
    return k


def cross_val_rbf(x, y):

    gam_n = 20
    lamb_n = 20
    errList = np.zeros((lamb_n, gam_n))
    bestErr=10**100
    bestGam=0
    bestLamb=0

    for gam in range(1, gam_n):
        for lamb in range(1, lamb_n):
            err = 0
            for i in range(len(x)):
                try:
                    a = regress(k_rbf(np.delete(x, i), np.delete(x, i), gam), np.delete(y, i), lamb)
                    f = np.dot(a, k_rbf(np.delete(x, i), x[i], gam))
                    err += (f-y[i])**2

                except np.linalg.linalg.LinAlgError:
                    err = np.inf
            print(err)
            errList[lamb, gam]=err
            if err<bestErr:
                bestGam = gam
                bestLamb = lamb
                bestErr = err

    plt.figure()
    X, Y = np.meshgrid(range(0, gam_n), range(0, lamb_n))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.log10(errList))
    plt.draw()
    return bestGam, bestLamb

x, y = generate_data(30)

d, lamb =cross_val_poly(x, y)
a = regress(k_poly(x, x, d), y, lamb)
f = np.dot(a, k_poly(x, x, d))

print('d: ' + str(d))
print('Lambda: ' + str(lamb))
# boot_f = boot_poly(x, y, d, lamb, 10)


plt.figure()
plt.plot(x, y, '.')
plt.plot(x, f, '.')
plt.legend(('Original Data', 'Estimate'))
plt.title('Poly Kernel')
plt.savefig('Figures/poly_kernel.pdf')
plt.draw()

gam, lamb =cross_val_rbf(x, y)
a = regress(k_rbf(x, x, gam), y, lamb)
f = np.dot(a, k_rbf(x, x, gam))


print('Gamma: ' + str(gam))
print('Lambda: ' + str(lamb))

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, f, '.')
plt.legend('Original Data', 'Estimate')
plt.title('RBF Kernel')
plt.savefig('Figures/rbf_kernel.pdf')
plt.draw()

x, y = generate_data(300)

d, lamb =cross_val_poly(x, y)
a = regress(k_poly(x, x, d), y, lamb)
f = np.dot(a, k_poly(x, x, d))

print('d: ' + str(d))
print('Lambda: ' + str(lamb))
# boot_f = boot_poly(x, y, d, lamb, 10)

plt.show()

