import torch
from torch2trt import torch2trt
from model import optimizer

PATH='/App/data/new_trt.pth'
TRT_TRAINED='/App/data/trained_trt.pth'

def start_converting(model,input,batch_size):
    print('start converting...')
    model_trt = torch2trt(model, [input],max_batch_size=batch_size, fp16_mode=True, max_workspace_size=1 << 25)
    y = model(input)
    y_trt = model_trt(input)

    # check the output against PyTorch
    print('diff',torch.max(torch.abs(y - y_trt)))
    print('save trt',TRT_TRAINED)
    torch.save(model_trt.state_dict(), TRT_TRAINED)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

        # Print model's state_dict
    print("trt Model's state_dict:")
    for param_tensor in model_trt.state_dict():
        print(param_tensor, "\t", model_trt.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("trt Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    return model_trt
