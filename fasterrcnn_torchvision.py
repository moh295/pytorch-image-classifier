
from utils import check_data_and_lable
import torch
from dataloader import cifar_dataloder
from train import start_training
# from Mymodel.nn300x2_v2 import Net300x2_v2
from Mymodel.nn128x2 import Net128x2

from validation import  random_check , overall_check ,each_class_check,torch2trt_check,overall_check2
from convert import start_converting
from convert_wit_onnx import onnx_start_converting
from torch2trt import TRTModule
from custemDataloader import load_data
import torch.nn as nn
import torch.optim as optim
from torchvision import models


TORCH_TRAINED= '/App/data/torch_trained_Net128x2.pth'
TRT_TRAINED='/App/data/trt_trained_Net128x2.pth'
ONNX_TRAINED="/App/data/onxx_trained_net128x2.onnx"

img_dir = 'data/dogsandcats'



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    #loading model




    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    #loading/checking data....

    batch_size=64
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
    x = [torch.rand(3, 300, 400).cuda(), torch.rand(3, 500, 400).cuda()]
    print('start predection')
    predictions = model(x)
    print('result',predictions)

    #model_trt=start_converting(model,x,batch_size,TRT_TRAINED)
    #model_trt=onnx_start_converting(model,x,batch_size,ONNX_TRAINED)

    #validating .....
    #trt_net=TRTModule()

    # random_check(model,model_path,data,classes)
    #
    #overall_check(model,model_path,data,classes)
    # each_class_check(model,model_path,data,classes)
    #overall_check2(model_trt,val_loader,batch_size)
    #overall_check2(model, val_loader, batch_size)

    #trt_net.load_state_dict(torch.load(TRT_TRAINED))
    #torch2trt_check (model_trt,model, data)

