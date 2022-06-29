import torch
from timeit import default_timer as timer
from datetime import timedelta

from engine import train_one_epoch, evaluate
from torchvision.datasets.vision.r import

def start_training(model,epochs,loder,optimizer,criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    start = timer()
    elapsed=0
    print('training started.. ')
    batches_lenght= len(loder)
    print('batches length',batches_lenght)  # = dataset_size / batch_size
    print_loss_each = int(batches_lenght/5)
    for epoch in range(epochs):  # loop over the dataset multiple times
        #print(epoch+1)


        running_loss = 0.0
        count=0



        for i, data in enumerate(loder, 0):
            count+=1
            #print('batch,epoch',count,epoch+1)

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if int(i % print_loss_each) == int(print_loss_each-1) and i >0:    # print every numbers of mini-batches
                end = timer()
                elapsed=timedelta(seconds=end - start)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_loss_each:.3f} elapsed {elapsed}')
                running_loss = 0.0

    print('Finished Training....duration :',elapsed )
    return model.state_dict()



def obj_detcetion_training(model,epochs,loder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loder, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
