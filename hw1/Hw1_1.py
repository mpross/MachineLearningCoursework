import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

mu=np.zeros([1,2,3])
mu[:,:,0]=np.array([1,2])
mu[:,:,1]=np.array([-1,1])
mu[:,:,2]=np.array([2,-2])

sigma=np.zeros([2,2,3])
sigma[:,:,0]=np.array([[1,0],[0,2]])
sigma[:,:,1]=np.array([[2,-1.8],[-1.8,2]])
sigma[:,:,2]=np.array([[3,1],[1,2]])



for k in range(3):
    draw = np.random.randn(2,100)
    w, v = np.linalg.eig(sigma[:,:,k])
    sigHalf=scipy.linalg.sqrtm(sigma[:,:,k])
    x = np.transpose(np.dot(sigHalf,draw))+mu[:,:,k]

    mean = np.sum(x,axis=0)/len(x)
    sm = np.zeros([2,2])
    for i in range(len(x)):
        sm += np.outer(x[i, :]-mean,x[i, :]-mean)
    covar = sm/(len(x)-1)

    w, v = np.linalg.eig(covar)
    xSquiggle=np.zeros(x.shape)

    for i in range(len(x)):
        xSquiggle[i,:] = 1/np.sqrt(w)*np.dot(np.transpose(v),x[i,:]-mean)

    plt.figure(k)
    plt.scatter(x[:, 0], x[:, 1], 50, 'b', '^',label='Original Draw')
    plt.scatter(xSquiggle[:, 0], xSquiggle[:, 1], 50, 'g', 'o', label='Transformed')
    plt.plot((mean[0], np.sqrt(w[1]) * v[0, 0] + mean[0]), (mean[1], np.sqrt(w[1]) * v[0, 1] + mean[1]),'r', linewidth=2,label='Eigen Vectors')
    plt.plot((mean[0], np.sqrt(w[0]) * v[1, 0] + mean[0]), (mean[1], np.sqrt(w[0]) * v[1, 1] + mean[1]),'r', linewidth=2)
    plt.axis((-6,6,-6,6))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig("GaussianDraws"+str(k)+".pdf")

plt.show()
