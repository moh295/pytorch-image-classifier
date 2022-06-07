import torch
from torch2trt import torch2trt



def start_converting(model,input):

    model_trt = torch2trt(model, [input])