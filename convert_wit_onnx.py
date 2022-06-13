# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

PATH='/App/data/new_trt.pth'
TRT_TRAINED='/App/data/resnet50.pth'

def onnx_start_converting(model,input,batch_size,TRT_TRAINED):
    print('start converting...')
    input_names='input'
    output_names= 'output'
    model_onnx=torch.onnx.export(model,
                                 input,
                                 "resnet50.onnx",
                                 verbose=False,
                                 export_params=True,
                                 )

    print('model is converted ')



    return model_onnx
