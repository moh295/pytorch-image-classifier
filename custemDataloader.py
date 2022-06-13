
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# imageFolder=
# dataset = DataLoader(training_data, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
# training_data, test_data = torch.utils.data.random_split(dataset, [50000, 10000])


# transforms.Resize(256),
# transforms.RandomCrop(224)

def load_data(data_folder,input_size, batch_size, phase='train', train_val_split=True, train_ratio=.8):
    classes = ('cat', 'dog')
    transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize(int(input_size*1.2)),
             transforms.RandomCrop(input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'test': transforms.Compose(
            [transforms.Resize(input_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}

    data = datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])
    if phase == 'train':
        if train_val_split:
            train_size = int(train_ratio * len(data))
            test_size = len(data) - train_size
            data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True,
                                                    num_workers=4)
            val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=4)
            return train_loader, val_loader ,classes
        else:
            train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                    num_workers=4)
            return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,
                                                    num_workers=4)
        return test_loader