import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class baseNet(nn.Module):
    def __init__(self):
        super(baseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(100, 100, 5)
        self.fc1 = nn.Linear(100 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 100 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class logRegNet(nn.Module):
    def __init__(self):
        super(logRegNet, self).__init__()
        self.fc1 = nn.Linear(3052, 10)

    def forward(self, x):
        x = x.view(4, -1)
        x = self.fc1(x)
        return x


class singleHidNet(nn.Module):

    def __init__(self):
        super(singleHidNet, self).__init__()

        M = 1000 #100

        self.fc1 = nn.Linear(3052, M)
        self.fc2 = nn.Linear(M, 10)

    def forward(self, x):
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class convNet(nn.Module):

    M = 200
    p = 5
    N = 14

    def __init__(self):
        super(convNet, self).__init__()


        self.conv1 = nn.Conv2d(3, self.M, self.p)
        self.pool = nn.MaxPool2d(self.N, self.N)
        self.fc1 = nn.Linear(int(self.M*(33-self.p)**2/self.N**2), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, int(self.M*(33-self.p)**2/self.N**2))
        x = self.fc1(x)
        return x



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = baseNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.5)

epochLim = 25

testAcc = np.zeros(epochLim)
trainAcc = np.zeros(epochLim)

for epoch in range(epochLim):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d %%' % (
            100 * correct / total))
    testAcc[epoch] = 100 * correct / total

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train accuracy: %d %%' % (
            100 * correct / total))

    trainAcc[epoch] = 100 * correct / total

print('Finished Training')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
# plt.savefig('3_convNet.pdf')
plt.show()
