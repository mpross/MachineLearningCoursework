import matplotlib.pyplot as plt
import numpy as np

supMax=0.0025
n=int(1.0/(2.0*supMax)**2)
Z=np.random.randn(n)

k=[1,8,64,512]
i=1000
for j in range(len(k)):
	x=np.sum(np.sign(np.random.randn(i, k[j]))*np.sqrt(1./k[j]), axis=1)
	plt.step(sorted(x),np.arange(1,i+1)/float(i))

plt.step(sorted(Z),np.arange(1,n+1)/float(n))
plt.xlabel("Observations")
plt.ylabel("Probability")
plt.xlim((-3,3))
plt.ylim((0,1))
k.append('Gaussian')
plt.legend(k)
plt.grid()
plt.show()


