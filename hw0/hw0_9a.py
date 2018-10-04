import matplotlib.pyplot as plt
import numpy as np

supMax=0.0025
n=int(1.0/(2.0*supMax)**2)
Z=np.random.randn(n)

plt.step(sorted(Z),np.arange(1,n+1)/float(n))
plt.xlabel("Observations")
plt.ylabel("Probability")
plt.xlim((-3,3))
plt.ylim((0,1))
plt.grid()
plt.show()


