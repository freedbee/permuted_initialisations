import torch
import torch.nn as nn
import module_utils as fun

class MySequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def get_device(self):
        return next(self.parameters()).device#weight.device

    def set_hooks(self, hook):
        for layer in self.my_modules():
            layer.execute_hooks = hook

    def my_modules(self):
        module_list = []
        eligible_modules = [fun.HookedLinear, fun.HookedConv2d]
        for layer in self.modules():
            if type(layer) not in eligible_modules:
                continue
            module_list.append(layer)
        return module_list


def FCNet(layer_widths, nonlinearity='relu', bias=False, bn=False):
    """"
    returns a fully connected network. 
    - "layer_widths" should be a list of integers 
        determining the widths of each layer.
    """
    mods = []
    for i in range(len(layer_widths)-1):
        mods.append(fun.HookedLinear(layer_widths[i], layer_widths[i+1],\
            bias=bias, nonlinearity=nonlinearity))
        if nonlinearity=='relu' and (i != len(layer_widths)-2):
            if bn:
                mods.append(nn.BatchNorm1d(layer_widths[i+1]))
            mods.append(nn.ReLU())
        elif nonlinearity=='softplus' and (i != len(layer_widths)-2):
            mods.append(nn.Softplus())
        elif nonlinearity=='linear' or (i == len(layer_widths)-2):
            pass
        else:
            print('UNSUPPORTED NONLINEARITY. Using linear network.')
    return MySequential(*mods)


def conv3x3(in_planes, planes,bias=False):
    return fun.HookedConv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)


def avg2x2():
    return nn.AvgPool2d(2)


def SimpleCNN(arch_str, dataset, bias=False, bn=False):
    """
    To see how arch_str is turned into a CNN, check config.py and the doc string
    for cnn_arch.
    """
    if "MNIST" in dataset:
        input_channels = 1
        width = 28
    elif "CIFAR" in  dataset:
        input_channels = 3
        width = 32
    elif "SVHN" in  dataset:
        input_channels = 3
        width = 32
    else:
        print("Dataset not supported.")
        exit()

    arch_str =arch_str.split('-')
    mod_list = []

    out_c = input_channels
    for a in arch_str:    
        if a == 'avg':
            mod_list.append(avg2x2())
            width //= 2
        else:
            in_c = out_c
            out_c = int(a)
            mod_list.append(conv3x3(in_c, out_c))
            if bn:
                mod_list.append(nn.BatchNorm2d(out_c))
            mod_list.append(nn.ReLU())
    mod_list.append(nn.AvgPool2d(width))
    mod_list.append(nn.Flatten())
    mod_list.append(fun.HookedLinear(out_c, 10, bias=bias, nonlinearity='relu'))
    return MySequential(*mod_list)


def get_new_net(net_type, arch, device, ds, bias=False, bn=False):
    if net_type == 'cnn':
        if ds in ['mnist', 'fashion']:
            net = SimpleCNN(arch, 'MNIST', bias=bias, bn=bn)
        elif ds == 'cifar10':
            net = SimpleCNN(arch, 'CIFAR', bias=bias, bn=bn)
    elif net_type == 'mlp':
        net = FCNet(arch, nonlinearity='relu', bias=bias, bn=bn)
    net.to(device)
    return net
