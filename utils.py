import torch
import numpy as np
from permutation_utils import recompute_bn_runnning_stats


def test_stats(net, loader, device='cuda', n_batches=None, rotation=None, a=None):
    net.eval()
    all_out = None
    all_y = None
    for t, (X,y) in enumerate(loader):
        if t == n_batches:
                break
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            if rotation is None or a is None:
                out = net(X)
            else: 
                out = net(X, a, rotation=rotation)
            if all_out == None:
                all_out = out
                all_y = y
            else:
                all_out = torch.cat([all_out, out])
                all_y = torch.cat([all_y, y])
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    l2 = criterion(all_out, all_y)
    all_pred = all_out.argmax(dim=1, keepdim=True)
    correct = all_pred.eq(all_y.view_as(all_pred)).sum().item()
    loss = l2.item()
    acc = correct/all_y.shape[0]
    # tempered loss
    l2_tempered = 1000.0
    temp_best = 0.0
    for temp in 10**np.linspace(-1, 2, 100):
        l2 = criterion(torch.tensor(temp,device=device)*all_out, all_y)
        if l2.item()<l2_tempered:
            l2_tempered = l2.item()
            temp_best = temp
    #print('inv temp', temp_best, 'loss', l2_best)
    return loss, l2_tempered, acc, temp_best


def get_stats_dict(net, trainloader0, testloader, device='cuda', a=None, rotation=None, empty=0, rec_bn=0):
    if empty: 
        d = {
        'train_acc': 0., 
        'test_acc': 0., 
        'train_nll': 10.,
        'train_nll_T': 10., 
        'test_nll': 10.,
        'test_nll_T': 10., 
        'train_temp': 1.,
        'test_temp': 1.,
        }
        return d
    if rec_bn:
        recompute_bn_runnning_stats(net, trainloader0)

    test_nll, test_nll_T, test_acc, test_temp = test_stats(net, testloader, device=device, a=a, rotation=rotation)
    train_nll, train_nll_T, train_acc, train_temp = test_stats(net, trainloader0, device=device, n_batches=100, a=a, rotation=rotation)
    d = {
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'train_nll': train_nll,
        'train_nll_T': train_nll_T,
        'train_temp': train_temp,
        'test_temp': test_temp, 
        'test_nll': test_nll,
        'test_nll_T': test_nll_T, 
    }
    return d

