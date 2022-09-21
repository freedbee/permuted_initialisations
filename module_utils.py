import torch
import torch.nn as nn
import numpy as np


class MyLinear(nn.Linear):
    """
    like nn.Linear, but with proper kaiming initialisation. 
    """
    def __init__(self, in_features, out_features, bias=True, nonlinearity='relu'):
        if nonlinearity=='softplus' or nonlinearity=='sigmoid':
            nonlinearity='relu'
        self.nonlinearity = nonlinearity
        super().__init__(in_features, out_features, bias)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #proper kaiming init, which is not the default
        nn.init.kaiming_normal_(self.weight, nonlinearity=self.nonlinearity)
        if self.bias is not None:
            bound = 0
            nn.init.uniform_(self.bias, -bound, bound)


class HookedLinear(MyLinear):
    """
    Linear Layer with hooks;
    to be adapted still
    """
    def __init__(self, n_in_features, n_out_features, bias=True, nonlinearity='relu'):
        super().__init__(n_in_features, n_out_features, bias, nonlinearity)
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.register_hooks()
        self.register_buffer('pre_act', torch.zeros((0,0)))
        self.register_buffer('post_grad', torch.zeros((0,0)))
                
    def register_hooks(self):
        self.register_full_backward_hook(self.back_hook_fn)
        self.register_forward_hook(self.forward_hook_fn)
        self.execute_hooks = False
    
    def back_hook_fn(self, module, input_grad, output_grad):
        """
        output_grad is tuple of len 1, containing tensor of shape (batch_size, out_dim)
        """
        if not self.execute_hooks:
            return
        self.post_grad = output_grad[0].detach().to('cpu')
        
    def forward_hook_fn(self, module, input_act, output_act):
        """
        input act is tuple of len 1, containing tensor of shape (batch_size, in_dim)
        """
        if not self.execute_hooks:
            return
        self.pre_act = input_act[0].detach().to('cpu')
        self.post_act = output_act.detach().to('cpu')


class MyConv2d(nn.Conv2d):
    """
    like nn.Conv2d, but with proper kaiming initialisation. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
            dilation=1, groups=1, bias=True, padding_mode='zeros', nonlinearity='relu'):
        if nonlinearity=='softplus':
            nonlinearity='relu'
        self.nonlinearity = nonlinearity
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #proper kaiming init, which is not the default
        nn.init.kaiming_normal_(self.weight, nonlinearity=self.nonlinearity)
        if self.bias is not None:
            bound = 0
            nn.init.uniform_(self.bias, -bound, bound)


class HookedConv2d(MyConv2d):
    """
    Linear Layer with hooks;
    to be adapted still
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
            dilation=1, groups=1, bias=True, padding_mode='zeros', nonlinearity='relu'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, nonlinearity='relu')
        
        self.unfold = nn.Unfold(kernel_size, stride=stride,\
                                        padding=padding)
                                        
        self.register_hooks()
        self.register_buffer('pre_act', torch.zeros((0,0)))
        self.register_buffer('post_grad', torch.zeros((0,0)))
            
    def register_hooks(self):
        self.register_full_backward_hook(self.back_hook_fn)
        self.register_forward_hook(self.forward_hook_fn)
        self.execute_hooks = False
    
    def back_hook_fn(self, module, input_grad, output_grad):
        """
        output_grad is tuple of len 1, containing tensor of shape (batch_size, out_dim)
        """
        if not self.execute_hooks:
            return
        self.post_grad = output_grad[0].detach().to('cpu')
        #e = self.post_grad.view(self.post_grad.shape[0], self.post_grad.shape[1], -1)
        #self.EE = torch.bmm(e, e.transpose(1,2)).sum(dim=0)

        
    def forward_hook_fn(self, module, input_act, output_act):
        """
        input act is tuple of len 1, containing tensor of shape (batch_size, in_dim)
        """
        if not self.execute_hooks:
            return
        self.pre_act = input_act[0].detach().to('cpu')
        self.post_act = output_act.detach().to('cpu')
        #a = self.unfold(self.pre_act)
        #self.AA = torch.bmm(a, a.transpose(1,2)).mean(dim=0)
        