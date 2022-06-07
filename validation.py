
import torch
from utils import imshow

from dataloader import classes,testloader
import torchvision
from timeit import default_timer as timer
from datetime import timedelta




def random_check(model, checkpoint):
    dataiter = iter(testloader)
    images, labels = dataiter.next()


    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    model.load_state_dict(torch.load(checkpoint))
    outputs = model(images)
    print('tesor size',images.size())
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    # print images
    imshow(torchvision.utils.make_grid(images))

def overall_check(model,checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # original saved file with DataParallel
    state_dict = torch.load(checkpoint)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    #model.load_state_dict(torch.load(checkpoint))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in testloader:
            numpred+=1
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end = timer()
    elapsed = timedelta(seconds=end - start)
    print(f'predction of {numpred} takes {elapsed}')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
def each_class_check(model,checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint))
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():

        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def torch2trt_check(model,model_trt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in testloader:
            numpred+=1
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end = timer()
    elapsed = timedelta(seconds=end - start)
    print(f'Torch predction of {numpred} takes {elapsed}')
    print(f'Torch Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    with torch.no_grad():
        numpred = 0
        start = timer()
        for data in testloader:
            numpred += 1
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model_trt(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end = timer()
    elapsed = timedelta(seconds=end - start)
    print(f'TensorRT predction of {numpred} takes {elapsed}')
    print(f'TensorRT Accuracy of the network on the 10000 test images: {100 * correct // total} %')