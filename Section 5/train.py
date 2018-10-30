"""
Train our style transfer model to
extract "style" features from
one image and apply it to the other.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from prep import device, get_data, imshow
from pprint import pprint


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Define how we will prepare our images to work with this
# network.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# This is the first layer in our network that will
# prepare images for our network.
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tensor(cnn_normalization_std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Different CNN layers extract different features
# related to both "style" and "content".
# The deeper we are in network the more abstract or general features are.
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):

    # Copy all cnn model.
    cnn = copy.deepcopy(cnn)

    # Layer that will prepare images to work with our network.
    normalization = Normalization().to(device)

    # Since we will be culculating our losses
    # manally we need to have direct access to all of our losses layers.
    content_losses = []
    style_losses = []

    # This is our new model that we will be
    # propagating with layers from pretrained VGG19 model
    # and our loss layers.
    # Normalization layer comes first to prepare images
    # to work with network.
    model = nn.Sequential(normalization)

    # Interating trough our VGG model,
    # and add each layer to our new model.
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        model.add_module(name, layer)

        # We add content loss layer right after
        # conv4 (this is where content specific features appear)
        if name in content_layers:
            print('Adding content loss layer...', name)
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # The general style features appear in all conv layers.
        if name in style_layers:
            print('Adding style loss layer...', name)
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Now we remove layers after the last content and style losses.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, style_img, content_img, input_img, n_steps=300,
                       style_weight=1000000, content_weight=1):
    """
    Run the style transfer.
    """
    print('Building the style transfer model..')
    print('Original VGG16 CNN model...')
    pprint(list(cnn.children()))
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    print('Our style transfer model...')
    pprint(list(model.children()))
    # Since we will be optimizing a single image for "style" we're
    # initializing it with the paramaters from it.
    loss_func = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    # To count interations, we have to use a list instead
    # of just a number, because we're using it in a
    # embedded function train() later on.
    run = [0]
    for e in range(n_steps):
        if run[0] > n_steps:
            break

        def train():
            # Clean up the out of range values of updated image.
            # Without that you will see pixels that do not belong
            # to any image.
            input_img.data.clamp_(0, 1)

            loss_func.zero_grad()

            # Run our model.
            # We usually calculate loss
            # based on the difference bettween
            # current and output value, but here
            # we do that inside loss layers themselves.
            model(input_img)

            # Cumulative loss over all loss layers.
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            # Balans losses using weights.
            style_score *= style_weight
            content_score *= content_weight

            # Calculate total loss.
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0]  % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        # Update input image to make it more
        # like both style and content.
        loss_func.step(train)

    # One more clean up before we go...
    input_img.data.clamp_(0, 1)

    return input_img

if __name__ == '__main__':
    import sys
    style_img_name, content_img_name, out_img_name=sys.argv[1], sys.argv[2], sys.argv[3]
    # How much content should be
    # "visible", heavier/higher weight=more content details.
    content_weight=1
    try:
        content_weight=int(sys.argv[4])
    except IndexError:
        pass

    # Get style and image tensors.
    style_img, content_img=get_data(style_img_name, content_img_name)
    # Create input image.
    input_img = content_img.clone()

    # Load pretrained network.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Get blended image.
    output = run_style_transfer(cnn, style_img, content_img, input_img,
                                content_weight=content_weight)

    print('Saving an output image to %s...' % out_img_name)
    # Remove an extra dimension that we needed to add
    # to get data in the right format for you network.
    image = output.squeeze(0)
    # Convert our output which is an image tensort back
    # to an image.
    to_pil_img = transforms.ToPILImage()
    image = to_pil_img(image)

    image.save(out_img_name)

    imshow(output, title='Output Image')
    plt.show()
