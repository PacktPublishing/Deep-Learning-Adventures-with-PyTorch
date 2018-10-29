"""
Classify images using a pre trained AlexNet CNN network
as well as a custom CNN model if it's available.

Custom CNN model has to be available in model.ckpt file.
You can generate this file by running ./train.py script.
"""
import torch
from torchvision import transforms, models
from PIL import Image
import os.path

from train import BeaverNet
from prep import get_data

# Getting the mapping between classes index and
# their names.
_,_,cifar100_classes=get_data()

def get_imgnet_classes():
    """
    Get labels/classes for ImageNet (AlexNet).
    Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    """
    return eval(open('imagenet1000_clsid_to_human.txt').read())

def prep_pretrained(imgf):
    """
    Process an image so it can be used with
    pre trained models available in PyTorch
    (including AlexNet).

    imgf - a name of the file to process
    """
    p_transform = transforms.Compose([
     transforms.Resize(226),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(imgf)
    new_img=p_transform(img)
    return new_img

def prep(imgf):
    """
    Process an image to use with our custom
    CNN network.
    """
    p_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(imgf)
    new_img=p_transform(img)
    return new_img

def classify(img_dir, prep, model, classes):
    """
    For each image in img_dir, first process an image
    using prep function,  use a an model to
    classify it according to classes defined in classes.
    """
    for f in os.listdir(img_dir):
        msg=f
        # First we need to preprocess an image
        # so it will be good fit for our model
        ni=prep(os.path.join(img_dir,f))
        # Our preparation function will return
        # a tensor with data from an image
        #
        # Tensor is bascily an array.
        # If you want to use it with a model
        # it has to be in the right "format"
        # unqueeze(0) means add this tensor
        # into an extra array
        uni=ni.unsqueeze(0)
        # No we're ready to use our mode.
        out=model(uni)
        # We need to convert our results from
        # tensor to an array that we can easily examine.
        mout=out.detach().numpy()
        # Creating an array with class indexes,
        # "score" (also called an energy) and name of
        # the classess.
        # The higher the energy for a give class the more
        # probable it is that our image belongs to it.
        aao=[]
        for i, o in enumerate(mout[0]):
            iv='?'
            try:
                iv=classes[i]
            except KeyError:
                pass
            aao.append((i, o, iv))
        # Just sort our array to show most probable classes first.
        aao.sort(key=lambda x: x[1], reverse=True)
        if aao[0][2] == 'beaver':
            msg+=' beaver!'
        else:
            #
            msg+=' not bever!'
        msg+=' Most probable classes: %s' % ','.join([ (aao[ci][2]+'(%f)' % aao[ci][1]) for ci in range(3) ])
        print(msg)


if __name__ ==  '__main__':
    print('Classification using pretrained AlexNet CNN model:')
    an=models.alexnet(pretrained=True)
    imgnc=get_imgnet_classes()
    classify('img_data/', prep_pretrained, an, imgnc)
    print(' ')
    print('Classification our custom CNN model:')
    net = BeaverNet(num_classes=100)
    if os.path.exists('model.ckpt'):
        net.load_state_dict(torch.load('model.ckpt'))
        net.eval()
    else:
        print('No model found! You need to train it first, just run train.py')
        exit(1)
    classify('img_data/',prep, net, cifar100_classes)
