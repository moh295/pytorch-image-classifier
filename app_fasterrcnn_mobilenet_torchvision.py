
from utils_local import check_data_and_lable
import torch
from dataloader import cifar_dataloder
from train import start_training,obj_detcetion_training
# from Mymodel.nn300x2_v2 import Net300x2_v2
from Mymodel.nn128x2 import Net128x2

from validation import  random_check , overall_check ,each_class_check,torch2trt_check,overall_check2
# from convert import start_converting
# from convert_wit_onnx import onnx_start_converting
# from torch2trt import TRTModule
from custemDataloader import load_data

from VOCloader import dataloader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from validation import overall_check3
from utils_local import tensor_to_PIL
from PIL import ImageDraw
# import numpy as np
import random


TORCH_TRAINED= '/App/data/torch_trained_fasterrcnn.pth'
TRT_TRAINED='/App/data/trt_trained_fasterrcnn.pth'
ONNX_TRAINED="/App/data/onxx_trained_fasterrcnn.onnx"

img_dir = 'data/dogsandcats'



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    #loading model

    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    model=models.load_state_dict(torch.load(TORCH_TRAINED))
    model.eval()


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)


    #loading/checking data....

    batch_size=30
    input_size=320
    print('batch size',batch_size)

    train_loader, trainval_loader, val_loader= dataloader()

    # trainging ....


    # epochs = 20
    # print_freq = 100
    # stat_dic = obj_detcetion_training(model, epochs, train_loader, val_loader, print_freq)
    # print('saving checkpoint to ', TORCH_TRAINED)
    # torch.save(stat_dic, TORCH_TRAINED)

    # converting...

    # model.load_state_dict(torch.load(TORCH_TRAINED))
    # x = [torch.rand(3, 300, 400).cuda(), torch.rand(3, 500, 400).cuda()]
    # print('start predection')

    # model_trt=start_converting(model,x,batch_size,TRT_TRAINED)

    #onnx
    # model_trt=onnx_start_converting(model,x,batch_size,ONNX_TRAINED)

    # validating .....



