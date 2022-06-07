import torch
from torch2trt import torch2trt

PATH='/App/data/new_trt.pth'
TRT_TRAINED='/App/data/trained_trt.pth'

def start_converting(model,input):
    print('start converting...')
    model_trt = torch2trt(model, [input])
    y = model(input)
    y_trt = model_trt(input)

    # check the output against PyTorch
    print('diff',torch.max(torch.abs(y - y_trt)))
    print('save trt')
    torch.save(model_trt.state_dict(), TRT_TRAINED)

    return model_trt
