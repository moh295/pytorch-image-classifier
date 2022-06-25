
import torch
from utils import imshow
import numpy as np

# from dataloader import classes,testloader
import torchvision
from timeit import default_timer as timer
from datetime import timedelta
from PIL import ImageDraw,Image
import random
from bbox import BBox
from utils import tensor_to_PIL



def random_check(model, checkpoint,loder,classes):
    train_loader, val_loader = loder
    dataiter = iter(val_loader)
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

def overall_check(model,checkpoint,loder,classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = loder

    model.load_state_dict(torch.load(checkpoint))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in val_loader:
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
def each_class_check(model,checkpoint,loder,classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = loder
    model.load_state_dict(torch.load(checkpoint))
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():

        for data in val_loader:
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

def torch2trt_check(model,model_trt,loder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = loder



    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in val_loader:
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

    correct = 0
    total = 0
    with torch.no_grad():
        numpred = 0
        start = timer()
        for data in val_loader:
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

def overall_check2(model,val_loader,batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in val_loader:

            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print('lable size', labels.size(0),labels.size())
            if labels.size(0) == batch_size:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                numpred += 1

    end = timer()
    elapsed = timedelta(seconds=end - start)
    print(f'prediction of {numpred*batch_size} instance takes {elapsed}')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def overall_check3(model,val_loader,batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        numpred=0
        start=timer()
        for data in val_loader:
            if total>=1000:
                break

            total += batch_size
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            for data,image in zip(outputs,images):
                detection_bboxes, detection_classes, detection_probs = data['boxes'].cpu().detach().numpy(), \
                                                                              data['labels'].cpu().detach().numpy(),data['scores'].cpu().detach().numpy()

                if detection_bboxes.size!=0 :
                    if np.amax(detection_probs) >0.7:
                    # print(np.amax(detection_probs))
                        numpred += 1
                        correct +=1

            # # the class with the highest energy is what we choose as prediction
            # detection_probs = torch.max(outputs.data, 1)
            # # print('lable size', labels.size(0),labels.size())
            # if labels.size(0) == batch_size:
            #

            #     correct += (detection_probs >0.7).sum().item()
            #     numpred += 1

    end = timer()
    elapsed = timedelta(seconds=end - start)
    print(f'prediction of {total} instances takes {elapsed}')
    # print(f'Accuracy : {100 * correct // total} %')

def inference_and_save(model,save_dir,images,mean=[0.485, 0.456, 0.406],std= [0.229, 0.224, 0.225]):
    # apply model on images and save the result
    scale = 1
    prob_thresh = 0.7
    cnt = 1
    predictions = model(images)
    path_to_output_image = save_dir


    for data, image in zip(predictions, images):
        # print('result',data['scores'])

        image=tensor_to_PIL(image,mean,std)



        detection_bboxes, detection_classes, detection_probs = data['boxes'].detach().numpy(), \
                                                               data['labels'].detach().numpy(), data[
                                                                   'scores'].detach().numpy()
        detection_bboxes /= scale
        # print(detection_probs)
        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]
        draw = ImageDraw.Draw(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            # category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            # draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
            draw.text((bbox.left, bbox.top), text=f'{prob:.3f}', fill=color)
        image.save(path_to_output_image + str(cnt) + '.png')
        print(f'Output image is saved to {path_to_output_image}{cnt}.png')
        cnt += 1
        image.show()

