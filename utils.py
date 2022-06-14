

import torchvision



# import matplotlib.pyplot as plt
# from dataloader import trainloader,classes,batch_size
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def check_data_and_lable(loder,classes,batch_size):
    train_loader, val_loader = loder
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images))
def torch_model_info(model,optimizer):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])