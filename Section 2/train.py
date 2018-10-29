"""
Train a custom CNN network using CIFAR100 dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
from torchvision import models

from prep import get_data,beavernet_transform,alexnet_transform

class BeaverNet(nn.Module):
    """
    Since training AlexNet is time consuming,
    we will use a much simpler CNN architecture.
    """
    def __init__(self, num_classes=100):
        super(BeaverNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_nn(net, epochs, trainloader, loss_function, optimizer):
    """
    Train net epochs number of times using data from trainloader
    and use loss_function and optimizer to get better.
    """
    for epoch in range(epochs):
        print('Epoch:', epoch+1)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get one batch of both images and labels.
            images, classes = data

            # Forward pass: predict classes for a given image.
            outputs = net(images)

            # Calculate the difference between
            # what we've predicted and what we should
            # predict.
            loss = loss_function(outputs, classes)

            # Because changes('gradients') are accumulated
            # from one iteration to another we need to
            # clean up the last ones, so we can propagate
            # the ones from this iteration.
            # Note: always call it before
            # loss.backward() and optimizer.step()
            optimizer.zero_grad()

            # Backward pass: accumulate changes('gradients')
            # that we've learned about in this iteration.
            loss.backward()

            # Backward pass: propagate changes trough the network.
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Trained on %d images' % i)

def test_nn(net, testloader, classes, batch_size=4):
    """
    Quickly test net on a small amount of data.
    """
    # Get the first image from a test data set
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print('Trying to predict ')
    print(' '.join(['%s' % classes[labels[j]] for j in range(batch_size)]))
    # Feed the image to the network and
    # get the classified classes.
    outputs = net(images)
    # Get the most probable classes first.
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ')
    print(' '.join(['%s' % classes[predicted[j]] for j in range(batch_size)]))

def test_nn_all(net, testloader):
    """
    Test data on all test dataset, calculate how
    much images have been classified correctly.
    """
    correct = 0
    total = 0
    # When testing we don't need to adjust
    # the network parameters, so we can
    # turn off accumulating changes('gradients').
    # (this will save us memory)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # Get a single batch of images
            # and associated classes.
            images, classes = data
            # Feed the network with those images
            # to check how they will be classified.
            outputs = net(images)
            # Get the most probable classes first.
            _, predicted = torch.max(outputs.data, 1)
            # Add current number images we process to the total.
            total += len(images)
            # How much images were classified correctly?
            correct += (predicted == classes).sum().item()

    print('Test accuracy on %d test images: %d %%' % (i, 100 * correct / total))

if __name__ == '__main__':
    from sys import argv
    epochs=int(argv[1])
    # To train models defined in pytorch
    # use get_data(transform=alexnet_transform)
    # and get_nn(models.alexnet(num_classes=100))
    train, test, classes=get_data(transform=beavernet_transform)
    net=BeaverNet(num_classes=100)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    train_nn(net, epochs, train, loss_function, optimizer)
    test_nn(net, test, classes)
    test_nn_all(net, test)
    torch.save(net.state_dict(), 'model.ckpt')
    print('Model saved in model.ckpt')
