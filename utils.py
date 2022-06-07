
import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
from dataloader import trainloader,classes,batch_size
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def check_data_and_lable():


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images))