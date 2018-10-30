"""
Prepare data for style transfer.

Inspired by:
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""
import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Use GPU if we can, otherwise just use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Size of the output image, keep it small (<512) if you don't have GPU.
# 256 with only cpu is on the edge between quality and speed ()
# 64-128 is good for quick testing, but you won't see much images
# details.
imsize = 512 if torch.cuda.is_available() else 128

# Prepare input images for training
# With Resize/CenterCrop combination
# our input images can be any size.
img_to_tensor = transforms.Compose([
    transforms.Resize(imsize+2),
    transforms.CenterCrop(imsize),
    transforms.ToTensor(),
])

def prep_img(name):
    """
    Open an image, scale it and
    turn it into a tensor.
    """
    image = Image.open(name)
    # Add an additional dimension
    # to our newly created image tensor
    # This just a format that is required
    # by pytorch.
    image = img_to_tensor(image).unsqueeze(0)
    return image.to(device)


def get_data(style_img, content_img):
    """
    Return both style and content images
    converted so they can be used with
    our network.
    """
    style_img = prep_img(style_img)
    content_img = prep_img(content_img)
    return style_img, content_img


def imshow(tensor, title):
    """
    Show tensor image.
    """
    image = tensor.squeeze(0)
    # Convert to PILImage.
    tensor_to_img = transforms.ToPILImage()
    image = tensor_to_img(image)
    plt.title(title)
    plt.imshow(image)
    plt.show()

def show_imgs(style_img, content_img):
    """
    Show both style and content images.
    """
    imshow(style_img, title='Style image')
    imshow(content_img, title='Content image')

if __name__ == '__main__':
    import sys
    style_img_name, content_img_name=sys.argv[1], sys.argv[2]
    si, ci=get_data(sys.argv[1], sys.argv[2])
    show_imgs(si, ci)
