from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import  transforms


def pascal_voc_loder(batch_size):
    transform = transforms.Compose(
        [transforms.Resize(300),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_dataset =datasets.VOCSegmentation(root='./data', year='2007', image_set='val', download=True,transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    train_dataset = datasets.VOCSegmentation(root='./data', year='2007', image_set='train', download=True,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)



    return train_loader, val_loader