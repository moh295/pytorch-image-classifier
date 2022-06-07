import torch
from torch2trt import torch2trt
from model import net
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(net, [x])