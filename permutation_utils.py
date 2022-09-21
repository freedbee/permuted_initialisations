import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as MinMatch
import module_utils


def get_parameters_no_bn(net):
    parameters_no_bn = []
    for n,p in net.named_parameters():
        if len(p.shape) > 1:
            parameters_no_bn.append(p)
    return parameters_no_bn


def create_weight_dict(net, n_models=2):
    ws = {}
    ws_merged = {}
    for model_id in range(n_models):
        ws[model_id] = {}
        ws_merged[model_id] = {}
        for layer in range(len(net.my_modules())):
            ws[model_id][layer] = {}
            ws_merged[model_id][layer] = {}
    return ws


def extend_weight_dict(ws, net, model_id, epoch):
    for layer_id, layer in enumerate(net.my_modules()):
        ws[model_id][layer_id][epoch] = {'w':[], 'b':[]}
    
        ws[model_id][layer_id][epoch]['w'] = layer.weight.data.clone().detach()
        if layer.bias is not None:
            ws[model_id][layer_id][epoch]['b'] = layer.bias.data.clone().detach()


def extend_stored_activations(As, new_As):
    if As is None:
        return new_As
    for t, new_A in enumerate(new_As):
        As[t] = torch.cat((As[t], new_A), dim=0)
    return As


def get_activations_errors_weights(net, X,y, pre_non_linearity=False):
    net.set_hooks(True)
    criterion = torch.nn.CrossEntropyLoss()
    X.requires_grad_(True)
    out = net(X)
    loss = criterion(out, y)
    loss.backward()
    X.requires_grad_(False)
    As, Es, Ws = [], [], []
    for layer in net.my_modules():
        Es.append(layer.post_grad.clone().detach().to('cpu'))
        if not pre_non_linearity:
            As.append(layer.pre_act.clone().detach().to('cpu'))
        else:
            As.append(layer.post_act.clone().detach().to('cpu'))
        Ws.append(layer.weight.data.clone().detach().to('cpu'))
    net.set_hooks(False)
    net.zero_grad()
    if pre_non_linearity:
        return As[:-1], Es[:-1], Ws[:-1]
    else:
        return As[1:], Es[:-1], Ws[:-1]


def get_activations(net0, net1, trainloader, n_batches, device):
    A0s = None
    A1s = None
    for t, (X,y) in enumerate(trainloader):
        if t==n_batches:
            break
        X = X.to(device)
        y = y.to(device)
        A0s_new, _, _ = get_activations_errors_weights(net0, X, y)
        A0s = extend_stored_activations(A0s, A0s_new)
        if net1 is not None:
            A1s_new, _, _ = get_activations_errors_weights(net1, X, y)
            A1s = extend_stored_activations(A1s, A1s_new)
    return A0s, A1s


def recompute_bn_runnning_stats(net, trainloader, n_batches=None):
    net.train()
    with torch.no_grad():
        for t, (X, _) in enumerate(trainloader):
            if t==n_batches:
                break
            net(X.to(net.get_device()))
        

def compute_cost_matrices(A1s, A2s, normalise=True):
    A1s_normalised, A2s_normalised = [], [] 
    costs = []
    for t, (A1, A2) in enumerate(zip(A1s, A2s)):
        if len(A1.shape) == 4:
            #print('conv')
            #print(A1.shape)
            A1 = A1.transpose(0,1)
            A1 = A1.reshape(A1.shape[0], -1)
            A1 = A1.t()
            
            A2 = A2.transpose(0,1)
            A2 = A2.reshape(A2.shape[0], -1)
            A2 = A2.t()
        
        A1_norm = torch.sqrt(A1.pow(2).sum(dim=0))
        A2_norm = torch.sqrt(A2.pow(2).sum(dim=0))
        if normalise:
            A1s_normalised.append( A1 / (A1_norm.view(1,-1)+1e-8) )
            A2s_normalised.append( A2 / (A2_norm.view(1,-1)+1e-8) )
        else:
            A1s_normalised.append( A1 ) 
            A2s_normalised.append( A2 ) 
        c = - A1s_normalised[-1].t() @ A2s_normalised[-1]
        costs.append( c.cpu().numpy() ) 
    return costs
    

def compute_permutations(costs):
    ps = []
    for cost in costs:
        _, p = MinMatch(cost)
        ps.append(p)
    return ps


def merge_networks(
        ws, 
        param_updates, 
        perms=[None, None], 
        a=1, 
        net_ids=[0,1], 
        net=None):
    """
    Merges two networks, which can be permuted before merging.

    Modifies `net` inplace, be careful.

    Arguments:
        - ws: Dictionary containing weights from different nets 
                and different points in time, which are to be merged.
        - param_updates: single int, or pair of ints specifying at which point in time
                the weights of the nets are accessed
        - perms: list of permutations (typically one for each hidden layer)
        - a: merging coefficient. Outputnet is: a*net1 + (1-a)*net2
        - net_ids: specifies which nets stored in `ws` are merged.
        - net: needs to be network of same arch as the one encoded by `ws`, 
            is overwritten inplace.
    """
    for t, layer in enumerate(net.my_modules()):
        layer.weight.data *= 0
        if layer.bias is not None:
            layer.bias.data *= 0
            
    if isinstance(param_updates, int):
        param_updates = [param_updates, param_updates]
    
    if isinstance(a, float) or a==0 or a==1:
        new_a = []
        for layer in enumerate(net.my_modules()):
            new_a.append(a)
        a = new_a
    
    perms = preprocess_perms(net, perms) 
                  
    n0 = net_ids[0]
    n1 = net_ids[1]
    
    for t, layer in enumerate(net.my_modules()):
        if isinstance(layer, module_utils.HookedLinear):
            layer.weight.data +=   a[t] * ws[n0][t][param_updates[0]]['w'][:,perms[0][t]][perms[0][t+1],:]  
            layer.weight.data += (1-a[t]) * ws[n1][t][param_updates[1]]['w'][:,perms[1][t]][perms[1][t+1],:] 

            if layer.bias is not None:
                layer.bias.data += a[t]  * ws[n0][t][param_updates[0]]['b'][perms[0][t+1]]
                layer.bias.data += (1-a[t])  * ws[n1][t][param_updates[1]]['b'][perms[1][t+1]]
        if isinstance(layer, module_utils.HookedConv2d):
            layer.weight.data += a[t] * ws[n0][t][param_updates[0]]['w'][:,perms[0][t],:,:][perms[0][t+1],:,:,:]
            layer.weight.data += (1-a[t]) * ws[n1][t][param_updates[1]]['w'][:,perms[1][t],:,:][perms[1][t+1],:,:,:]
            if layer.bias is not None:
                layer.bias.data += a[t]  * ws[n0][t][param_updates[0]]['b'][perms[0][t+1]]
                layer.bias.data += (1-a[t])  * ws[n1][t][param_updates[1]]['b'][perms[1][t+1]]
        
    return net

def preprocess_perms(net, perms):
    for ii in [0,1]:
        if perms[ii] is None:
            perm = []
            last = 0
            for t, layer in enumerate(net.my_modules()):
                s = layer.weight.data.shape[1]
                perm.append(np.arange(s))
            perms[ii] = perm
    for ii in [0,1]:
        if len(perms[ii][-1]) != net.my_modules()[-1].weight.shape[0]:
            out = net.my_modules()[-1].weight.shape[0]
            perms[ii].append(np.arange(out))
        if len(perms[ii][0]) != net.my_modules()[0].weight.shape[1]:
            perms[ii].insert(0, np.arange(net.my_modules()[0].weight.shape[1]))  
    return perms


