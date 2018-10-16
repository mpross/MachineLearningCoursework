import numpy as np
import matplotlib as plt
from mnist import  MNIST

X_train=[]
X_test=[]
labels_train=[]
labels_test=[]
def load_dataset():
    global X_train, X_test, labels_test, labels_train
    mndata = MNIST('./python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0

def one_hot(input):
    output = np.zeros((input.size, 10))
    for i in range(len(input)):
        vec = np.zeros(10)
        for j in range(10):
            vec[j]=(int(int(labels_train[i]) == j))
            output[i]=vec
    return output

def train(X, y, lamb):
    w=np.linalg.solve(np.dot(np.transpose(X), X)+lamb*np.identity(X.shape[1]), np.dot(np.transpose(X), y))
    return w

def predict(w,x):
    y=np.dot(np.transpose(w), x)
    return np.argmax(y)

load_dataset()

y_train=one_hot(labels_train)
w=train(X_train, y_train, 1)
print(labels_train[10])
print(predict(w,X_train[10]))


