import argparse
import os
import time
import scanpy as sc
from tqdm import tqdm
from os.path import join
from scanorama import assemble

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
# from tensorboardX import SummaryWriter

from GLOBE.config import Config
from GLOBE.utils import *
# from src.infer import Con2Infer, UnConInfer
from GLOBE.loss import GLOBE_SupLoss, GLOBE_UnsupLoss
from GLOBE.dataset import CEDataset, GLOBEDataset
from GLOBE.model import EntHead, ConHead
from GLOBE.trainval import supervised_train_one_epoch, unsupervised_train_one_epoch

import matplotlib.pyplot as plt

config = Config()

class CE(object):
    """docstring for BuildModel"""
    def __init__(self, 
            # in_dim=2000,
            # lat_dim=128, 
            # n_class=64,  # final dim is determined the number of classes
            # header='mlp',
            # block_level=1,
            # init='uniform',
            gpu='0',
            exp_id='Pancreas_1'    # or Pancreas
        ):
        super(CE, self).__init__()

        # self.in_dim = in_dim
        # self.lat_dim = lat_dim
        # self.n_class = n_class
        # self.header = header
        # self.block_level = block_level
        # self.init = init
        self.gpu = gpu
        self.exp_id = exp_id

        # self.device = torch.device("cuda")

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        # create logs dir
        self.logs_dir = '%s/CE/%s' % (config.out_root, self.exp_id)
        create_dirs([self.logs_dir, join(self.logs_dir, 'weights')])

        # self.build_model()


    def build_model(
            self,
            lat_dim = 128,
            block_level=1, 
            init='uniform'
        ):

        print('building model')
        self.model = EntHead(
            in_dim=self.dataset.n_feature,   # n_hvgs
            lat_dim=lat_dim, 
            n_class=self.dataset.n_class, 
            block_level=block_level,
            init=init
        )

        self.model.cuda()

    def build_dataset(
            self, 
            sps_x, 
            gnames, 
            cnames, 
            metadata, 
        ):
        print('building dataset')
        self.dataset = CEDataset(
                sps_x=sps_x,
                gnames=gnames,
                cnames=cnames,
                metadata=metadata,
            )


    def train(
            self, 
            lr=1e-4,
            batch_size=256,
            epochs=60,
            plot_loss=True,
            save_freq=10,
            weight_decay=1e-5,
            num_workers=6,
        ):
        # Supervised encoder training
        criterion = nn.CrossEntropyLoss()  # fuck you
        criterion = criterion.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, 
                                weight_decay=weight_decay
                    )

        # as you see
        train_loader = dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

        # select train mode
        train_one_epoch = supervised_train_one_epoch

        # start training
        train_loss_curve = []
        for epoch in range(epochs):
            ep_loss = train_one_epoch(train_loader, self.model, criterion, optimizer, epoch)
            train_loss_curve.append(ep_loss)

            if ((epoch+1) % save_freq) == 0:
                torch.save(
                    self.model.state_dict(), 
                    join(self.logs_dir, f'weights', 'checkpoints_{:03d}.pth'.format(epoch+1))
                )

        if plot_loss:
            fig = plt.plot(train_loss_curve)
            plt.title('loss curve')
            plt.savefig(join(self.logs_dir, 'loss curve'), facecolor='white')
            plt.show()
    

    def load_ckpt(self, ckpt_idx):
        ckpt_path = join(self.logs_dir, 'checkpoints_{:03d}.pth'.format(ckpt_idx))
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # state_dict = ckpt['model']

        self.model.load_state_dict(state_dict)
        print(f'=> loaded checkpoint {ckpt_idx}.pth')

    def evaluate(self, batch_size=256, num_workers=6):
        val_loader = dataloader.DataLoader(
            dataset = self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        # self.model.eval()

        rec = torch.Tensor(0).cuda()
        with torch.no_grad():
            for cell, _ in val_loader:
                cell = cell.cuda()
                output = self.model.encoder(cell)   # use encoder only, of course
                rec = torch.cat([rec, output], 0)
        rec = rec.cpu().detach().numpy()    
        return rec, self.dataset.metadata






        