
from utils import check_data_and_lable
import torch
from dataloader import trainloader
from train import start_training
from model import net,optimizer,criterion
from validation import  random_check , overall_check ,each_class_check
if __name__ == '__main__':
 #check_data_and_lable()
 stat_dic=start_training(net,2,trainloader,optimizer,criterion)
 PATH = './cifar_net.pth'
 # print('saving checkpoint to ',PATH)
 # torch.save(stat_dic, PATH)
 random_check(net,PATH)
 overall_check(net,PATH)
 each_class_check(net,PATH)
