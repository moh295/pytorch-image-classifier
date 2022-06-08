
from utils import check_data_and_lable
import torch
from dataloader import batch_size ,cifar_dataloder
from train import start_training
from Mymodel.nn32x10 import Net32x10
from Mymodel.nn100x2 import Net100x2
from validation import  random_check , overall_check ,each_class_check,torch2trt_check,overall_check2
from convert import start_converting
from torch2trt import TRTModule
from custemDataloader import load_data
import torch.nn as nn
import torch.optim as optim
from torchvision import models

PATH = '/App/data/torch_model.pth'
TRT_PATH = '/App/data/new_trt.pth'
TRT_TRAINED='/App/data/trained_trt.pth'

pc='catsanddogs.pth'



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

 #loading model


 #model = Net32x10().to(device)
 model = Net100x2().to(device)
 #model = models.resnet50().to(device)


 criterion = nn.CrossEntropyLoss()
 optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
 lr = 0.003
 #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


 #loading/checking data....

 batch_size=64
 model_path = PATH
 img_dir = 'data/dogsandcats'
 classes = ('cat', 'dog')
 #classes = ('plane', 'car', 'bird', 'cat',
 #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 #data=cifar_dataloder
 data=load_data(img_dir,batch_size,'train',True,0.8) #('a','b')

 train_loader, val_loader =data

 # check_data_and_lable(data,classes,batch_size)

 for d in val_loader:
  x, labels = d[0].to(device), d[1].to(device)
  break
 print('tensor size',x.size(),'lable size',labels.size())
 print('label',labels)



 #trainging ....
 epochs=3

 stat_dic=start_training(model,epochs,train_loader,optimizer,criterion)
 print('saving checkpoint to ',TRT_TRAINED)
 torch.save(stat_dic, TRT_TRAINED)

 #converting...

 # model.load_state_dict(torch.load(model_path))
 #x = torch.ones((batch_size, 3, 32, 32)).cuda()
 x = torch.ones((batch_size, 3, 100, 100)).cuda()
 model_trt=start_converting(model,x,batch_size)


 #validating .....
 #trt_net=TRTModule()

 # random_check(model,model_path,data,classes)
 #
 #overall_check(model,model_path,data,classes)
 # each_class_check(model,model_path,data,classes)
 overall_check2(model_trt,data)

 #trt_net.load_state_dict(torch.load(TRT_TRAINED))
 #torch2trt_check (model_trt,model, data)

