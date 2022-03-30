import torch
from torch.utils.data import Dataset

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from collections import defaultdict
from scipy.sparse.csgraph import connected_components

from os.path import join

from collections import defaultdict
# from GLOBE.prepare_dataset import prepare_dataset
# from GLOBE.preprocessing import preprocess_dataset
from GLOBE.config import Config
from GLOBE.NNs import reduce_dimensionality, nn_approx
# from GLOBE.mnn import generate_mnns, nn_approx

configs = Config()

# ============================
# Supervised Baseline: CE
# ============================

class CEDataset(Dataset):
    def __init__(
        self,
        sps_x, 
        gnames,
        cnames, 
        metadata,
        ):
        super().__init__()

        self.X = sps_x 
        self.gnames = gnames
        self.cnames = cnames
        self.metadata = metadata

        self.n_sample = sps_x.shape[0]
        self.n_feature = sps_x.shape[1]
        self.n_batch = metadata[configs.batch_key].unique().size
        self.batch_label = metadata[configs.batch_key].values

        y = metadata[configs.label_key]
        self.y = y.cat.codes  # .cat must be accessor of 'category' dtype
        self.y_category = y.cat.categories.values
        self.n_class = self.y_category.size


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i].A
        x = x.squeeze()
        x = x.astype('float32')
        return x, self.y[i]

KNN = 10
class GLOBEDataset(Dataset):
    def __init__(
        self,
        mode,
        sps_x, 
        gnames,
        cnames, 
        metadata,
        anchor_data,
        add_noise=False,  # only work for unsupervised mode
        use_knn=False,     # for unsuperivsed mode, cal knn
        knn_alone=False  # knn only used for sample without mnn 
        ):
        super().__init__()

        self.mode = mode
        self.X = sps_x 
        self.gnames = gnames
        self.cnames = cnames
        self.metadata = metadata
        self.anchor_data = anchor_data
        self.add_noise = add_noise
        self.use_knn = use_knn
        self.knn_alone = knn_alone

        self.n_sample = sps_x.shape[0]
        self.n_feature = sps_x.shape[1]
        self.n_batch = metadata[configs.batch_key].unique().size
        self.batch_label = metadata[configs.batch_key].values

        if mode.startswith('supervised'):
            y = metadata[configs.label_key]
            self.y = y.cat.codes  # .cat must be accessor of 'category' dtype
            self.y_category = y.cat.categories.values
            self.n_class = self.y_category.size

            if mode.endswith('InfoNCE'):
                self.getitem_f = self.get_item_supervised2
            else:
                self.getitem_f = self.get_item_supervised

        elif mode.startswith('unsupervised'):
            # MNN pairs
            anchors = self.anchor_data[['cell1', 'cell2']].values
            n_mnn_pairs = len(anchors)//2
            anchors = anchors[:n_mnn_pairs]

            if use_knn:
                knn_pairs = aug_pos_pairs(sps_x, self.batch_label, knn=KNN)
                if knn_alone:
                    self.knn_dict = knn_pairs[:,1].reshape(self.n_sample, KNN) 
                else:
                    anchors = np.vstack([anchors, knn_pairs])  # all samples have their neighbors as positives

                print(f'MNN pairs = {n_mnn_pairs}, knn_pairs={len(knn_pairs)}, merged={len(anchors)}')

            self.mnn_dict, self.mnn_graph, I = get_mnn_graph(self.n_sample, anchors)
            self.mnn_graph = (self.mnn_graph + I) > 0   # to create mask for criterion

            self.getitem_f = self.get_item_unsupervised


    def __len__(self):
        return self.X.shape[0]

    def get_item_supervised(self, i):
        x = self.X[i].A
        x = x.squeeze()
        x = x.astype('float32')
        return x, self.y[i]

    def get_item_supervised2(self, i):
        xi = self.X[i].A.squeeze()
        yi = self.y[i]

        # random select a sample with the same type label as the positive sample
        j = np.random.choice(np.where(self.y == yi)[0])
        xj = self.X[j].A.squeeze()
        yj = yi
        return [xi, xj], [yi, yj]
        

    def get_item_unsupervised(self, i):
        x = self.X[i].A
        x = x.squeeze()

        pos_anchors = self.mnn_dict[i]
        if len(pos_anchors)<=0:
            if self.use_knn and self.knn_alone:
                pi = np.random.choice(self.knn_dict[i])
                x_p = self.X[pi].A + self.add_noise * np.random.normal(0, 1, self.n_feature)
            else:
                x_p = x + self.add_noise * np.random.normal(0, 1, self.n_feature)
            # if all samples use knn, then pos_anchors > 0
            # if knn only for target samples, then mnn_graph is indeed mnn, pi=i to ensure mask=1
            pi = i
        else:                # use itself as a Positive pair, TO-DO: consider add some noise here
            pi = np.random.choice(pos_anchors)
            x_p = self.X[pi].A + self.add_noise * np.random.normal(0, 1, self.n_feature)

        x, x_p = x.astype('float32').squeeze(), x_p.astype('float32').squeeze()
        return [x, x_p], [i, pi]

    def __getitem__(self, i):
        return self.getitem_f(i)



def get_mnn_graph(n_cells, anchors):
    # sparse Mnn graph, exclude cell itself
    mnn_graph = sps.csr_matrix((np.ones(anchors.shape[0]), (anchors[:, 0], anchors[:, 1])),
                                dtype=np.int8,
                                shape=(n_cells, n_cells))
    mnn_graph = (mnn_graph + mnn_graph.T) > 0  # to be symmetric

    # create a sparse identy matrix
    dta, csr_ind = np.ones(n_cells,), np.arange(n_cells)
    I = sps.csr_matrix((dta, (csr_ind, csr_ind)), dtype=np.int8)  # identity_matrix

    # sparse mnn_list for all cells
    mnn_dict = defaultdict(list)
    for p in anchors:   #  some neighbors may repeat
        mnn_dict[p[0]].append(p[1])
        mnn_dict[p[1]].append(p[0])
    return mnn_dict, mnn_graph, I


def aug_pos_pairs(X, batch_label, knn=KNN):
    n_sample = X.shape[0]

    # calculate knn index for each sample
    nns = np.ones((n_sample, knn+1), dtype='long')  # allocate (N, k+1) space
    bs = batch_label.unique()
    for bi in bs:
        bii = np.where(batch_label==bi)[0]
        print(f'==> {bi}, n_sample={len(bii)}')

        # dim reduction for efficiency
        X_pca = reduce_dimensionality(X, 50)
        batch_nns = nn_approx(X_pca[bii], X_pca[bii], knn=knn+1)  # itself and its nns

        # convert local batch index to global cell index
        nns[bii, :] = bii[batch_nns.ravel()].reshape(batch_nns.shape)

    # convert neighbor adj to pairs
    idx2 = nns[:, 1:(1+knn)].ravel()  # exclude itself
    idx1 = np.repeat(np.arange(n_sample), knn)
    knn_pairs = set([(tp[0], tp[1]) for tp in zip(idx1, idx2)])  # take the unique pairs
    knn_pairs = np.asarray([[tp[0], tp[1]] for tp in knn_pairs])
    return knn_pairs