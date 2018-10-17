import numpy as np
import matplotlib.pyplot as plt

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
    print(w)
    print(v)
    x = np.transpose(np.dot(v*np.sqrt(w)*np.linalg.inv(v),draw))+mu[:,:,k]

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
    plt.scatter(x[:,0],x[:,1],None,'b','^')
    plt.quiver(mean[0],mean[1],np.sqrt(w[0])*v[0,:]+mean[0],np.sqrt(w[1])*v[1,:]+mean[1],color='red',scale=1, units='xy',headlength=0,headaxislength=0)
    plt.scatter(xSquiggle[:,0],xSquiggle[:,1],None,'g','o')
    plt.axis((-6,6,-6,6))

plt.show()
