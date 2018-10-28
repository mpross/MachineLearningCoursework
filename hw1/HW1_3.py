import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random
import time

from numpy.core.multiarray import ndarray

X_train = []
X_test = []
labels_train = []
labels_test = []


def load_dataset():
    global X_train, X_test, labels_test, labels_train
    mndata = MNIST('./python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0


def one_hot(inpt):
    output = np.zeros((inpt.size, 10))
    for i in range(len(inpt)):
        vec = np.zeros(10)
        for j in range(10):
            vec[j] = int(int(inpt[i]) == j)
            output[i] = vec
    return output


def train(X, y, lamb):
    w = np.linalg.solve(np.dot(X.T, X)+lamb*np.identity(X.shape[1]), np.dot(X.T, y))
    return w


def predict(w, x):
    y=np.dot(w.T, x.T)
    return np.argmax(y, axis=0)


def generateTransform(inpt, p, sigma):
    G = sigma*np.random.randn(inpt.shape[1], p)
    b = np.random.uniform(0, 2*np.pi, (1, p))

    return G, b


def transform(inpt, G, b):
    out = np.cos(np.dot(inpt,G)+b)
    return out


def split(x, y, ylist, frac):
    index = random.sample(range(x.shape[0]), int(x.shape[0]*frac))
    xmajor = x[index]
    xminor = np.delete(x, index, 0)
    ymajor = y[index]
    yminor = np.delete(y, index, 0)
    listmajor = ylist[index]
    listminor = np.delete(ylist, index, 0)

    return xmajor, xminor, ymajor, yminor, listmajor, listminor

if __name__=="__main__":
	load_dataset()
	
	y_train=one_hot(labels_train)
	w = train(X_train, y_train, 10**-4)

	print("No transformation")
	print("Training Accuracy: "+str(sum(predict(w, X_train) == labels_train)/len(X_train)*100)+"%")
	print("Testing Accuracy: "+str(sum(predict(w, X_test) == labels_test)/len(X_test)*100)+"%")

	pmax=1*10**4
	trainErr = np.zeros((pmax, 1))
	valErr = np.zeros((pmax, 1))
	for p in range(1, pmax):
	    start=time.time()
	    G, b= generateTransform(X_train, p, np.sqrt(0.1))
	    transX = transform(X_train, G, b)
	    trainX, valX, trainY, valY, trainList, valList = split(transX, y_train, labels_train, 0.8)
	    w= train(trainX, trainY, 10**-4)
	    trainErr[p] = sum( predict(w, trainX) == trainList) / len(trainX)*100
	    valErr[p] = sum(predict(w, valX) == valList) / len(valX)*100
	    end=time.time()
	    print(str(round(p/pmax*100, 3))+"% Done")
	    print(str(round((end-start)*(pmax-p),2))+" s left")
	
	
	pOpt = np.argmax(valErr)
	G, b = generateTransform(X_train, p, 0.01)
	transX = transform(X_train, G, b)
	trainX, valX, trainY, valY, trainList, valList = split(transX, y_train, labels_train, 0.8)
	w = train(trainX, trainY, 10**-4)
	
	print("With transformation")
	print("Training Accuracy: "+str(sum(predict(w, trainX) == trainList)/len(trainX)*100)+"%")
	print("Testing Accuracy: "+str(sum(predict(w, transform(X_test, G, b)) == labels_test)/len(X_test)*100)+"%")
	
	plt.plot(trainErr)
	plt.plot(valErr)
	plt.grid()
	plt.ylabel("Accuracy (%)")
	plt.xlabel("p")
	plt.legend(("Training Accuracy", "Validation Accuracy"))
	plt.tight_layout()
	plt.savefig("CrossVal.pdf")

load_dataset()

y_train=one_hot(labels_train)
w = train(X_train, y_train, 10**-4)

print("No transformation")
print("Training Accuracy: "+str(sum(predict(w, X_train) == labels_train)/len(X_train)*100)+"%")
print("Testing Accuracy: "+str(sum(predict(w, X_test) == labels_test)/len(X_test)*100)+"%")

pmax=1*10**3
trainErr = np.zeros((pmax, 1))
valErr = np.zeros((pmax, 1))

for p in range(1, pmax):
    start=time.time()
    G, b= generateTransform(X_train, p, np.sqrt(0.1))
    transX = transform(X_train, G, b)
    trainX, valX, trainY, valY, trainList, valList = split(transX, y_train, labels_train, 0.8)
    w= train(trainX, trainY, 10**-4)
    trainErr[p] = sum( predict(w, trainX) == trainList) / len(trainX)*100
    valErr[p] = sum(predict(w, valX) == valList) / len(valX)*100
    end=time.time()
    print(str(round(p/pmax*100, 3))+"% Done")
    print(str(round((end-start)*(pmax-p),2))+" s left")


pOpt = np.argmax(valErr)
G, b = generateTransform(X_train, p, 0.01)
transX = transform(X_train, G, b)
trainX, valX, trainY, valY, trainList, valList = split(transX, y_train, labels_train, 0.8)
w = train(trainX, trainY, 10**-4)

print("With transformation")
print("Training Accuracy: "+str(sum(predict(w, trainX) == trainList)/len(trainX)*100)+"%")
print("Testing Accuracy: "+str(sum(predict(w, transform(X_test, G, b)) == labels_test)/len(X_test)*100)+"%")

plt.plot(trainErr)
plt.plot(valErr)
plt.grid()
plt.ylabel("Accuracy (%)")
plt.xlabel("p")
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show()
