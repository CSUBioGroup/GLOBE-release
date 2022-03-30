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
from GLOBE.loss import GLOBE_SupLoss, GLOBE_UnsupLoss, InfoNCE
from GLOBE.dataset import CEDataset, GLOBEDataset
from GLOBE.model import EntHead, ConHead, EncoderL2
from GLOBE.trainval import supervised_train_one_epoch, unsupervised_train_one_epoch
from GLOBE.trainval import supervised_train_one_epoch_withNCE, unsupervised_train_one_epoch_withNCE

import matplotlib.pyplot as plt

config = Config()

class GLOBE(object):
    """docstring for BuildModel"""
    def __init__(self, 
            mode='supervised',          # supervised or unsupervised
            gpu='0',
            exp_id='Pancreas_1'    # or Pancreas
        ):
        super(GLOBE, self).__init__()

        self.mode = mode
        self.gpu = gpu
        self.exp_id = exp_id

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        # create logs dir
        self.logs_dir = '%s/GLOBE/%s' % (config.out_root, self.exp_id)
        create_dirs([self.logs_dir, join(self.logs_dir, f'{mode}_weights')])

        # self.build_model()


    def build_model(
            self,
            lat_dim=128,
            proj_dim=64,
            header='mlp',
            block_level=1,
            init='uniform'
        ):
        print('building model')
        self.header = header
        if header is None:
            self.model = EncoderL2(
                in_dim=self.dataset.n_feature,
                lat_dim=lat_dim,
                block_level=block_level,
            )
        if header:
            self.model = ConHead(
                in_dim=self.dataset.n_feature,   # n_hvgs
                lat_dim=lat_dim, 
                proj_dim=proj_dim, 
                header=header,
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
        anchor_data,
        add_noise=False,
        use_knn=False,
        knn_alone=False,
        ):
        print('building dataset')
        self.dataset = GLOBEDataset(
                mode=self.mode, 
                sps_x=sps_x,
                gnames=gnames,
                cnames=cnames,
                metadata=metadata,
                anchor_data=anchor_data,
                add_noise=add_noise,
                use_knn=use_knn,
                knn_alone=knn_alone
            )


    def train(
            self, 
            loss_type='GLOBE',
            temp=0.7,
            lr=1e-4,
            batch_size=256,
            epochs=60,
            plot_loss=True,
            save_freq=10,
            weight_decay=1e-5,
            num_workers=6,
            view='single'   # double is the same as paper described
        ):
        # Supervised encoder training
        if loss_type=='GLOBE':
            criterion = GLOBE_SupLoss(temp) if self.mode.startswith('supervised') else GLOBE_UnsupLoss(temp)
        else:
            criterion = InfoNCE(batch_size, temp)
        criterion = criterion.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # as you see
        train_loader = dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=self.mode.endswith('InfoNCE')  # bug, leads to cuda error
            )

        # select train mode
        if self.mode.startswith('supervised'):
            if self.mode.endswith('InfoNCE'):
                train_one_epoch = supervised_train_one_epoch_withNCE
            else:
                train_one_epoch = supervised_train_one_epoch
        else:
            if self.mode.endswith('GLOBE'): # my bad, not symmetric
                train_one_epoch = unsupervised_train_one_epoch(self.dataset.mnn_graph, view)
            else:
                train_one_epoch = unsupervised_train_one_epoch_withNCE

        # start training
        train_loss_curve = []
        for epoch in range(epochs):
            ep_loss = train_one_epoch(train_loader, self.model, criterion, optimizer, epoch)
            train_loss_curve.append(ep_loss)

            if ((epoch+1) % save_freq) == 0:
                torch.save(
                    self.model.state_dict(), 
                    join(self.logs_dir, f'{self.mode}_weights', 'checkpoints_{:03d}.pth'.format(epoch+1))
                )

        if plot_loss:
            fig = plt.plot(train_loss_curve)
            plt.title(f'{self.mode} loss curve')
            plt.savefig(join(self.logs_dir, f'{self.mode} loss curve'), facecolor='white')
            plt.show()
    

    def load_ckpt(self, ckpt_idx):
        ckpt_path = join(self.logs_dir, f'{self.mode}_weights', 'checkpoints_{:03d}.pth'.format(ckpt_idx))
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # state_dict = ckpt['model']

        self.model.load_state_dict(state_dict)
        print(f'=> loaded {self.mode} checkpoint {ckpt_idx}')

    def evaluate(self, batch_size=256, num_workers=6, keep_dropout=True):
        val_loader = dataloader.DataLoader(
            dataset = self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        if self.mode.startswith('supervised') and (keep_dropout == False):
            print('discard dropout for inference in supervised mode')
            self.model.eval()

        if self.mode.startswith('unsupervised'):  # a little tricky for supervised methods
            self.model.eval()


        emb1, emb2 = torch.Tensor(0).cuda(), torch.Tensor(0).cuda()
        with torch.no_grad():
            for cell, _ in val_loader:
                if self.mode == 'supervised-GLOBE':
                    cell = cell.cuda()
                else:
                    cell = cell[0].cuda()
                lat_emb = self.model.encoder(cell)   # use encoder only, of course
                emb1 = torch.cat([emb1, lat_emb], 0)
                if self.header is not None:
                    prj_emb = self.model.fc(lat_emb)     # use projection outputs
                    emb2 = torch.cat([emb2, prj_emb], 0)

        emb1 = emb1.cpu().detach().numpy()  
        emb2 = emb2.cpu().detach().numpy()  
        return emb1, emb2, self.dataset.metadata

    def integrate_gene(self, rec=None, knn=10, sigma=20, alpha=0.1):
        # create raw AnnObject
        X, metadata = self.dataset.X, self.dataset.metadata
        gname = self.dataset.gnames

        adata_raw = sc.AnnData(X)
        adata_raw.obs = metadata.copy()
        adata_raw.var_names = gname

        if rec is None:
            raise ValueError('Please compute embeddings first')

        # split batches 
        batch_vect = adata_raw.obs[config.batch_key].values
        batch_set = np.unique(batch_vect)
            
        # create inputs need by Scanorama
        datasets_dimred, datasets, metas = [], [], []
        for bi in batch_set:
            idx = batch_vect == bi
            
            datasets_dimred.append(rec[idx].copy())
            datasets.append(sps.csr_matrix(adata_raw.X[idx]))  # sparse matrix is required

            metas.append(adata_raw.obs.loc[idx])

        ## for debugging
        # datasets_dimred, matches = assemble(
        #         datasets_dimred, # Assemble in low dimensional space.
        #         expr_datasets=datasets, # Modified in place.
        #         verbose=False, knn=knn, sigma=sigma, approx=True,
        #         alpha=alpha, ds_names=None, batch_size=None,
        # )

        datasets_dimred = assemble(
                datasets_dimred, # Assemble in low dimensional space.
                expr_datasets=datasets, # Modified in place.
                verbose=False, knn=knn, sigma=sigma, approx=True,
                alpha=alpha, ds_names=None, batch_size=None,
        )

        print('=> gene integrated')

        return sps.vstack(datasets), pd.concat(metas)
   






        