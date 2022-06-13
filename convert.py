import torch
from torch2trt import torch2trt


PATH='/App/data/new_trt.pth'
TRT_TRAINED='/App/data/resnet50.pth'

def start_converting(model,input,batch_size):
    print('start converting...')
    #model_trt = torch2trt(model, [input],max_batch_size=batch_size, fp16_mode=True, max_workspace_size=1 << 25)
    model_trt = torch2trt(model, [input], max_batch_size=batch_size,fp16_mode=True)
    y = model(input)
    y_trt = model_trt(input)

    # check the output against PyTorch
    print('diff',torch.max(torch.abs(y - y_trt)))
    print('save trt',TRT_TRAINED)
    torch.save(model_trt.state_dict(), TRT_TRAINED)

    return model_trt
