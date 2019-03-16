

from PIL import Image
import torch
import cv2
from model.net import CNN
import torchvision.transforms as transforms
from torch.autograd import Variable
mean_grey = 0.1307
stddev_grey = 0.3081
model = CNN()
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()
weights = torch.load('./weights/mnist_weights.pth.tar')
model.load_state_dict(weights)

transforms = transforms.Compose([transforms.Resize((28, 28)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean_grey, ), (stddev_grey, ))])


def predict(img_name, model):
    image = cv2.imread(img_name, 0)
    ret, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    img = 255 - threshold

    img = Image.fromarray(img)

    img = transforms(img)
    img = img.view(1, 1, 28, 28)
    img = Variable(img)

    model.eval()
    if CUDA:
        model = model.cuda()
        img = img.cuda()
    output = model(img)

    _, predicted = torch.max(output, 1)
    return predicted.item()


pred = predict('./images/8.jpg', model)
print('the predicted output is: {}'.format(pred))
