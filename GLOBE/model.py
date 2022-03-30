import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import time

from GLOBE.weight_init import init_f

dropout_rate = 0.3

def Naive_block(in_dim, out_dim):
    layer = nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.ReLU()
    )
    return layer

def Dropout_block(in_dim, out_dim):
    layer = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate)
    )
    return layer

def BatchNorm_block(in_dim, out_dim):
    layer = nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )
    return layer

basic_blocks = [Naive_block, Dropout_block, BatchNorm_block]

class Encoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            lat_dim: int = 20,
            block_level = 0,
            # init: str = 'uniform'
        ):
        super().__init__()
        basic_block = basic_blocks[math.floor(block_level)]

        self.layer1 = basic_block(in_dim, in_dim//4)

        self.layer2 = basic_block(in_dim//4, in_dim//8)

        self.layer3 = basic_block(in_dim//8, lat_dim) if block_level==1.5 else Naive_block(in_dim//8, lat_dim)

        # self.weights_init(init)

    # def weights_init(self, init_name):
    #     init_function = init_f(init_name)

    #     for m in self.modules():
    #         init_function(m)

    def forward(self, cell):
        output = self.layer3(self.layer2(self.layer1(cell)))
        return output

class EncoderL2(nn.Module):
    def __init__(
            self,
            in_dim: int,
            lat_dim: int = 20,
            block_level = 0,
            # init: str = 'uniform'
        ):
        super().__init__()
        self.encoder = Encoder(in_dim, lat_dim, block_level)

    def forward(self, cell):
        output = self.encoder(cell)
        output = F.normalize(output, dim=1)
        return output

class Decoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            lat_dim: int = 20,
            block_level: int=0,
            # init: str='uniform'
        ):
        super().__init__()
        basic_block = basic_blocks[block_level]        

        self.layerm3 = basic_block(lat_dim, in_dim//8)   # 3d before output
        self.layerm2 = basic_block(in_dim//8, in_dim//4)  # 2nd before output

        self.layerm1 = nn.Sequential(
            nn.Linear(in_dim//4, in_dim),
            # nn.ReLU(inplace=True)
        )

    #     self.weights_init(init)

    # def weights_init(self, init_name):
    #     init_function = init_f(init_name)

    #     for m in self.modules():
    #         init_function(m)    

    def forward(self, cell):
        output = self.layerm1(self.layerm2(self.layerm3(cell)))
        return output


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int=2000, 
        lat_dim: int=128,
        block_level: int=0,
        init: str='uniform'
    ):
        super().__init__()

        # encoder part 
        self.encoder = Encoder(
            in_dim,
            lat_dim,
            block_level
        )

        self.decoder = Decoder(
            in_dim,      # the input dim of encoder
            lat_dim,
            block_level
        )

        self.weights_init(init)

    def weights_init(self, init_name):
        init_function = init_f(init_name)

        for m in self.modules():
            init_function(m)

    def forward(self, x):
        x_emb = self.encoder(x)
        x_rec = self.decoder(x_emb)
        return x_rec

# with categoryCrossEntropy Loss
class EntHead(nn.Module):
    def __init__(self, 
        in_dim: int,
        n_class: int, 
        lat_dim: int = 20,
        block_level: int = 0,
        init: str = 'uniform'):
        super().__init__()

        self.encoder = Encoder(
            in_dim, 
            lat_dim, 
            block_level
        )
        self.fc = nn.Sequential(
            nn.Linear(lat_dim, n_class),
        )

        self.weights_init(init)

    def weights_init(self, init_name):
        init_function = init_f(init_name)

        for m in self.modules():
            init_function(m)

    def forward(self, cell):
        output = self.fc(self.encoder(cell))
        return output

'''
    The encoder + Projection head, same as GLOBE
'''
class ConHead(nn.Module):
    def __init__(self,
                in_dim: int,
                lat_dim: int=50,
                proj_dim: int=20,
                header: str = 'mlp',
                block_level: int=0,
                init: str='uniform'
        ):
        super().__init__()

        self.encoder = Encoder(
            in_dim,
            lat_dim,
            block_level
        )

        if header == 'mlp':
            self.fc = nn.Sequential(
                BatchNorm_block(lat_dim, lat_dim) if block_level==2 \
                    else Naive_block(lat_dim, lat_dim),
                nn.Linear(lat_dim, proj_dim)
            )
        elif header == 'linear':
            self.fc = nn.Linear(lat_dim, proj_dim)
            
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(header))

        self.weights_init(init)

    def weights_init(self, init_name):
        init_function = init_f(init_name)

        for m in self.modules():
            init_function(m)

    def forward(self, cell):
        output = self.fc(self.encoder(cell))
        output = F.normalize(output, dim=1)
        return output

