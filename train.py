import torch
from timeit import default_timer as timer
from datetime import timedelta

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

