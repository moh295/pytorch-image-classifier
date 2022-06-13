# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx



def onnx_start_converting(model,input,batch_size,ONNX_TRAINED):
    print('start converting...')

    model_onnx=torch.onnx.export(model,input, ONNX_TRAINED,verbose=False)

    print('model is converted ')



    return model_onnx
