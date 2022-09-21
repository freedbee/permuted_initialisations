import torch
from torchvision import datasets, transforms

PATH_TO_DATASETS = 'data'

def get_dataloaders(
        dataset, 
        net_type, 
        batch_size, 
        batch_size_2, 
        num_workers=0):
    transform_list = [transforms.ToTensor()]
    if dataset in ['mnist', 'fashion']:
        input_dim = 784
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    if dataset in ['cifar10']:
        input_dim = 3 * 32**2
        transform_list.append(transforms.Normalize((0.5,),(0.2,)))
    if net_type == 'mlp':
        transform_list.append(lambda x: x.view(input_dim))
    transform = transforms.Compose(transform_list)

    if dataset=='mnist':
        train_data = datasets.MNIST(PATH_TO_DATASETS, train=True, download=True,
                            transform=transform)
        test_data = datasets.MNIST(PATH_TO_DATASETS, train=False,
                            transform=transform)
    if dataset=='fashion':
        train_data = datasets.FashionMNIST(PATH_TO_DATASETS, train=True, download=True,
                            transform=transform)
        test_data = datasets.FashionMNIST(PATH_TO_DATASETS, train=False,
                            transform=transform)
    if dataset=='cifar10':
        train_data = datasets.CIFAR10(root=PATH_TO_DATASETS, train=True, download=True, 
                                    transform=transform)
        test_data = datasets.CIFAR10(root=PATH_TO_DATASETS, train=False, download=True, 
                                    transform=transform)

    
    trainloader1 = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
            pin_memory=False, drop_last=False)
    trainloader2 = torch.utils.data.DataLoader(train_data, batch_size=batch_size_2, shuffle=True, num_workers=num_workers, 
            pin_memory=False, drop_last=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, 
            pin_memory=False, drop_last=False, num_workers=num_workers)
    return trainloader1, trainloader2, testloader
