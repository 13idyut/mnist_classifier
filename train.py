
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.net import CNN
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as dataloader

import torchvision.transforms as transforms
batch_size = 100
num_epochs = 50
mean_grey = 0.1307
stddev_grey = 0.3081
transforms = transforms.Compose([transforms.Resize((28, 28)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean_grey, ), (stddev_grey, ))])
train_loader = dataloader(dataset=train_datasets,
                          shuffle=True,
                          batch_size=batch_size)
train_datasets = datasets.MNIST(root=r'./data',
                                train=True,
                                transform=transforms,
                                download=True)
net = CNN()
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

loss_function = nn.CrossEntropyLoss()

train_loss = []
train_accuracy = []

for epoch in range(num_epochs):

    correct = 0
    iteration = 0
    iter_loss = 0.0

    net.train()

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)

        if CUDA:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output = net(images)

        loss = loss_function(output, labels)
        iter_loss += loss.item()

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(output, 1)

        correct += (predicted == labels).sum()
        iteration += 1

    train_loss.append(iter_loss/iteration)

    train_accuracy.append((100 * correct / len(train_datasets)))
    print('epoch: [{}/{}], training loss: {:.3f}, training accuracy: {}%'.format(epoch +
                                                                                 1, num_epochs, train_loss[-1],        train_accuracy[-1]))


torch.save(net.state_dict(), './weights/mnist_weights.pth.tar')
