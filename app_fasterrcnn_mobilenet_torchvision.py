
import torch
from train import start_training,obj_detcetion_training

from validation import  inference_and_save_mobilnet_full_data
# from convert import start_converting
# from convert_wit_onnx import onnx_start_converting
# from torch2trt import TRTModule


from VOCloader import dataloader

from torchvision import models


TORCH_TRAINED= '/App/data/torch_trained_fasterrcnn.pth'
TRT_TRAINED='/App/data/trt_trained_fasterrcnn.pth'
ONNX_TRAINED="/App/data/onxx_trained_fasterrcnn.onnx"

img_dir = 'data/dogsandcats'
#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['hand','targetobject']


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    #loading model

    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False).to(device)
    model.load_state_dict(torch.load(TORCH_TRAINED))
    model.eval()


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)


    #loading/checking data....

    batch_size=30
    input_size=320
    print('batch size',batch_size)

    train_loader, trainval_loader, val_loader= dataloader(batch_size, input_size)

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

    #for one portion of data

    # for d in val_loader:
    #     images= d[0]
    #     break
    #
    # images = list(image.to(device) for image in images)
    # inference_and_save_mobilnet(model,'/App/data/output/',images,labels_dict)

    #for all validation dataset

    inference_and_save_mobilnet_full_data(model, '/App/data/output/', val_loader, labels_dict)
