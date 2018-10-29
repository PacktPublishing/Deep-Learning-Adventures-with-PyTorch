"""
Get and prepare the CIFAR100 dataset and its classes
to train a custom CNN Beaver detector.
"""
import torch
torch.manual_seed(1)

from torchvision import transforms, utils
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_data(batch_size=4, transform=None):
    """
    Get training and test sets ready to use
    with our network.

    Return also the class names/labels that are
    available in our dataset.

    batch_size - a number of samples to split our dataset into
    transform - transform.Compose, list of transforms to do on each image
    """
    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # After data has been downloaded we can access the mapping
    # between class indexes and the name of the classes.
    classes=pickle.load(open('./data/cifar-100-python/meta', 'rb'))
    classes=classes['fine_label_names']

    return trainloader, testloader, classes

def imshow(img):
    """
    Show images.
    """
    # Since we've already transformed
    # our images we need to "undo" it
    # simple to make them more visible.
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg=np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

def show(trainloader,classes,batch_size=4):
    """
    Show some images from training set.
    """
    # Turn data loader to interator,so
    # we can get some images.
    dataiter = iter(trainloader)
    # Get a single batch (4 images).
    images, labels = dataiter.next()
    # Show the name of the classes for those images.
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    # Show images.
    imshow(utils.make_grid(images))

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
beavernet_transform = transforms.Compose([
 transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

alexnet_transform = transforms.Compose([
 transforms.Resize(226),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    from pprint import pprint
    trs, ts, classes = get_data(transform=beavernet_transform)
    show(trs, classes)
    pprint(classes)
    print('Number of classes:', len(classes))
    print("Where's the beaver?")
    print(classes.index('beaver'))
