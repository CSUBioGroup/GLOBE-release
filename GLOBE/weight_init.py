import torch
import torch.nn as nn

def uniform_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def normal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def one_init(m):
    if isinstance(m, nn.Linear):
        nn.init.ones_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def eye_init(m):
    if isinstance(m, nn.Linear):
        nn.init.eye_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def init_f(name):
    if name == 'uniform':
        return uniform_init
    elif name == 'normal':
        return normal_init
    elif name == 'zero':
        return zero_init
    elif name == 'one':
        return one_init
    elif name == 'eye':
        return eye_init




