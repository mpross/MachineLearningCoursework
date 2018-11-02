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
    return x, y, w

def coordDescent(x, y, lamb, delta, w0):
    d = x.T[0].size
    n=x[0].size

    a = np.zeros(d)
    c = np.zeros(d)
    diff = np.zeros(d)
    w = w0
    wLast=np.zeros(d)

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


def regularization(x, y, delta, w0, wTrue):

    lamb = np.max(2*np.abs(np.dot(x, (y-(np.sum(y)/y.size)))))
    w = coordDescent(x, y, lamb, delta, w0)
    non0 = np.sum(w != 0)

    FDR = np.sum(np.logical_and(wTrue == 0, w != 0)) / np.sum(w != 0)
    TPR = np.sum(np.logical_and(wTrue != 0, w != 0)) / np.sum(wTrue != 0)

    non0Vec = np.array((0, non0))
    lambVec = np.array((0, lamb))
    FDRVec = np.array((0, FDR))
    TPRVec = np.array((0, TPR))

    while np.sum(w != 0) <= 0.99 * x.T[0].size:

        lamb = lambVec[-1]*0.75
        w = coordDescent(x, y, lamb, delta, w)
        non0 = np.sum(w != 0)

        FDR = np.sum(np.logical_and(wTrue == 0, w != 0)) / np.sum(w != 0)
        TPR = np.sum(np.logical_and(wTrue != 0, w != 0)) / np.sum(wTrue != 0)

        non0Vec = np.append(non0Vec, non0)
        lambVec = np.append(lambVec, lamb)
        FDRVec = np.append(FDRVec, FDR)
        TPRVec = np.append(TPRVec, TPR)

    plt.figure(1)
    plt.plot(lambVec[1:], non0Vec[1:])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('Non-Zero Elements')
    plt.draw()
    plt.savefig('NonzerovsLambda.pdf', bbox_inches='tight')

    plt.figure(2)
    plt.plot(lambVec[1:], FDRVec[1:], label="FDR")
    plt.plot(lambVec[1:], TPRVec[1:], label="TPR")
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.legend()
    plt.draw()
    plt.savefig('FDR&TPRvsLambda.pdf', bbox_inches='tight')

    plt.figure(3)
    plt.plot(FDRVec[1:], TPRVec[1:])
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.axis([0, 1, 0, 1])
    plt.draw()
    plt.savefig('FDRvsTPR.pdf', bbox_inches='tight')

    return w




if __name__ == "__main__":
    start=time.time()
    x, y, wTrue = generateSynth(n, d, k, sigma)

    w = regularization(x, y, 0.01, np.zeros(d), wTrue)
    print("Execution Time: "+str(time.time()-start))
    plt.show()