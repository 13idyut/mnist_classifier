import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.net import CNN
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader
from torch.autograd import Variable
mean_grey = 0.1307
stddev_grey = 0.3081
batch_size = 100
transforms = transforms.Compose([transforms.Resize((28, 28)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean_grey, ), (stddev_grey, ))])


test_datasets = datasets.MNIST(root=r'./data',
                               train=False,
                               transform=transforms)
test_loader = dataloader(dataset=test_datasets,
                         shuffle=False,
                         batch_size=batch_size)


net = CNN()
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()

loss_function = nn.CrossEntropyLoss()

weights = torch.load('./weights/mnist_weights.pth.tar', map_location='cpu')

net.load_state_dict(weights)


net.eval()
loss = 0.0
iteration = 0
correct = 0
test_loss = []
test_accuracy = []
for i, (images, labels) in enumerate(test_loader):
    images = Variable(images)
    labels = Variable(labels)
    if CUDA:
        images = images.cuda()
        labels = labels.cuda()
    output = net(images)
    loss = loss_function(output, labels)
    loss += loss.item()
    _, predicted = torch.max(output, 1)
    correct += (predicted == labels).sum()
    iteration += 1
test_loss.append(loss/iteration)
test_accuracy.append((100 * correct / len(test_datasets)))
print('testing loss: {:.3f}, testing accuracy: {}%'.format(test_loss[-1], test_accuracy[-1]))
