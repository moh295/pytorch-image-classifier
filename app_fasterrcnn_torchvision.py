
from utils import check_data_and_lable
import torch
from dataloader import cifar_dataloder
from train import start_training
# from Mymodel.nn300x2_v2 import Net300x2_v2
from Mymodel.nn128x2 import Net128x2

from validation import  random_check , overall_check ,each_class_check,torch2trt_check,overall_check2
# from convert import start_converting
# from convert_wit_onnx import onnx_start_converting
# from torch2trt import TRTModule
from custemDataloader import load_data
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from validation import overall_check3
# import numpy as np
# from PIL import ImageDraw
# import random
# from bbox import BBox
# import torchvision.transforms as T

TORCH_TRAINED= '/App/data/torch_trained_fasterrcnn.pth'
TRT_TRAINED='/App/data/trt_trained_fasterrcnn.pth'
ONNX_TRAINED="/App/data/onxx_trained_fasterrcnn.onnx"

img_dir = 'data/dogsandcats'



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    #loading model




    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    #loading/checking data....

    batch_size=62
    input_size=320

    train_loader, val_loader,classes =load_data(img_dir,input_size,batch_size,'train',True,0.7)

    for d in val_loader:
        x, labels = d[0].to(device), d[1].to(device)
        break
    print('tensor size',x.size(),'lable size',labels.size())
    print('label',labels)



    #trainging ....

    #
    # epochs=5
    # stat_dic=start_training(model,epochs,train_loader,optimizer,criterion)
    # print('saving checkpoint to ',TORCH_TRAINED)
    # torch.save(stat_dic, TORCH_TRAINED)

    #converting...

    # model.load_state_dict(torch.load(TORCH_TRAINED))
    # x = [torch.rand(3, 300, 400).cuda(), torch.rand(3, 500, 400).cuda()]
    # print('start predection')

    #model_trt=start_converting(model,x,batch_size,TRT_TRAINED)
    #model_trt=onnx_start_converting(model,x,batch_size,ONNX_TRAINED)


    #apply model on images and save the result
    # scale = 1
    # prob_thresh = 0.7
    # cnt=1
    # predictions = model(x)
    # path_to_output_image= 'data/output/image'
    # for data,image in zip(predictions , x):
    #     # print('result',data['scores'])
    #
    #     transform = T.ToPILImage()
    #     image = transform(image)
    #     detection_bboxes, detection_classes, detection_probs = data['boxes'].detach().numpy(),\
    #                                                            data['labels'].detach().numpy(),data['scores'].detach().numpy()
    #     detection_bboxes /= scale
    #     # print(detection_probs)
    #     kept_indices = detection_probs > prob_thresh
    #     detection_bboxes = detection_bboxes[kept_indices]
    #     detection_classes = detection_classes[kept_indices]
    #     detection_probs = detection_probs[kept_indices]
    #     draw = ImageDraw.Draw(image)
    #
    #     for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
    #         color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
    #         bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
    #         # category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
    #         draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
    #         #draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
    #         draw.text((bbox.left, bbox.top), text=f'{prob:.3f}', fill=color)
    #     image.save(path_to_output_image+str(cnt)+'.png')
    #     print(f'Output image is saved to {path_to_output_image}{cnt}.png')
    #     cnt+=1
    #



    #validating .....
    #trt_net=TRTModule()
    overall_check3(model, val_loader, batch_size)

    # random_check(model,model_path,data,classes)
    #overall_check(model,model_path,data,classes)
    # each_class_check(model,model_path,data,classes)
    # overall_check2(model_trt,val_loader,batch_size)
    #overall_check2(model, val_loader, batch_size)

    #trt_net.load_state_dict(torch.load(TRT_TRAINED))
    #torch2trt_check (model_trt,model, data)

