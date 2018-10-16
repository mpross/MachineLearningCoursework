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
    output = []
    for i in range(len(input)):
        vec = []
        for j in range(10):
            vec.append(int(int(labels_train[i]) == j))
            output.append(vec)
    return output


load_dataset()

y=one_hot()

