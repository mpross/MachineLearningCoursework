import numpy as np
import matplotlib as plt
from mnist import MNIST

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
            vec[j] = (int(int(inpt[i]) == j))
            output[i] = vec
    return output


def train(X, y, lamb):
    w = np.linalg.solve(np.dot(np.transpose(X), X)+lamb*np.identity(X.shape[1]), np.dot(np.transpose(X), y))
    return w


def predict(w,x ):
    y=np.dot(np.transpose(w), np.transpose(x))
    return np.argmax(y, axis=0)


def transform(inpt,p,sigma):
    G = sigma*np.random.randn(inpt.shape[1],p)
    b = np.random.uniform(0, 2*np.pi, (p,1))
    out = np.cos(np.dot(np.transpose(G), np.transpose(inpt))+b)
    return out


load_dataset()

y_train=one_hot(labels_train)
w = train(X_train, y_train, 10**-4)

transX=transform(X_train, 10, 0.01)

print(transX.shape)

print("Training Error: "+str(sum(predict(w,X_train)==labels_train)/len(X_train)*100)+"%")
print("Testing Error: "+str(sum(predict(w,X_test)==labels_test)/len(X_test)*100)+"%")



