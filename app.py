
from utils import check_data_and_lable
import torch
from dataloader import trainloader
from train import start_training
from model import net,optimizer,criterion
from validation import  random_check , overall_check ,each_class_check
from convert import start_converting

PATH = '/App/data/cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
 #check_data_and_lable()

 # stat_dic=start_training(net,5,trainloader,optimizer,criterion)
 # print('saving checkpoint to ',PATH)
 # torch.save(stat_dic, PATH)

 #random_check(net,PATH)
 # overall_check(net,PATH)
 # each_class_check(net,PATH)
 # x = torch.ones((1, 3, 5, 5)).cuda()
 dataiter = iter(trainloader)
 # print(dataiter.next())
 inputs, labels = dataiter[0].to(device), dataiter[1].to(device)
 start_converting(net,inputs)
