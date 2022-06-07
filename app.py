
from utils import check_data_and_lable
import torch
from dataloader import trainloader
from train import start_training
from model import net,optimizer,criterion
from validation import  random_check , overall_check ,each_class_check,torch2trt_check
from convert import start_converting

PATH = '/App/data/cifar_net.pth'
TRT_PATH = '/App/data/new_trt.pth'
TRT_TRAINED='/App/data/trained_trt.pth'



if __name__ == '__main__':
 #check_data_and_lable()
 x = torch.ones((4, 3, 32, 32)).cuda()
 model_trt=start_converting(net,x)
 # stat_dic=start_training(model_trt,5,trainloader,optimizer,criterion)
 # print('saving checkpoint to ',TRT_TRAINED)
 # torch.save(stat_dic, TRT_TRAINED)

 # #random_check(net,PATH)
 # overall_check(net,PATH)
 # each_class_check(net,PATH)

 net1=net.load_state_dict(torch.load(PATH))
 net2=net.load_state_dict(torch.load(TRT_TRAINED))
 torch2trt_check (net1,net2,PATH,TRT_TRAINED)
