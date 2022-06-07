
from utils import check_data_and_lable
import torch
from dataloader import trainloader, testloader,batch_size
from train import start_training
from model import net,optimizer,criterion
from validation import  random_check , overall_check ,each_class_check,torch2trt_check
from convert import start_converting

PATH = '/App/data/cifar_net.pth'
TRT_PATH = '/App/data/new_trt.pth'
TRT_TRAINED='/App/data/trained_trt.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

 #loading/checking data....

 #check_data_and_lable()

 for data in testloader:
  x, labels = data[0].to(device), data[1].to(device)
  break
 print('tensor size',x.size())

 #
 #converting...
 #
 # net.load_state_dict(torch.load(PATH))
 # x = torch.ones((4, 3, 32, 32)).cuda()
 # model_trt=start_converting(net,x,batch_size)


 #trainging ....

 # stat_dic=start_training(net,2,trainloader,optimizer,criterion)
 # print('saving checkpoint to ',PATH)
 # torch.save(stat_dic, PATH)


 #validating .....

 # #random_check(net,PATH)
 overall_check(net,TRT_TRAINED)
 each_class_check(net,TRT_TRAINED)


 # net2=net.load_state_dict(torch.load(TRT_TRAINED))
 # torch2trt_check (net,model_trt)
