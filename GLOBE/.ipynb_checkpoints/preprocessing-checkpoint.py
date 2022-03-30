import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
# from imblearn.under_sampling import RandomUnderSampler

from GLOBE.config import Config

configs = Config()
sc.settings.verbosity = configs.verbose 


def preprocess_data(
    sps_x, 
    cnames, 
    gnames, 
    metadata, 
    anchor_data,
    select_hvg=True, 
    scale=False
    ):
    #####  preprocessing raw dataset
    adata = sc.AnnData(sps.csr_matrix(sps_x.T))  # transposed, (gene, cell) -> (cell, gene)
    adata.obs_names = cnames
    adata.var_names = gnames
    adata.obs = metadata.loc[cnames].copy()

    sc.pp.filter_genes(adata, min_cells=configs.min_cells) 
    sc.pp.normalize_total(adata, target_sum=configs.scale_factor)
    sc.pp.log1p(adata)

    if select_hvg:
        sc.pp.highly_variable_genes(adata, 
                                    n_top_genes=min(adata.shape[1], configs.n_hvgs), 
                                    # min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    batch_key=configs.batch_key
                                    )

        adata = adata[:, adata.var.highly_variable].copy()

    if scale:
        # warnings.warn('Scaling per batch! This may cause memory overflow!')
        ada_batches = []
        for bi in adata.obs[configs.batch_key].unique():
            bidx = adata.obs[configs.batch_key] == bi
            adata_batch = adata[bidx].copy()
            sc.pp.scale(adata_batch)

            ada_batches.append(adata_batch)

        adata = sc.concat(ada_batches)

    #####  preprocessing mnn data
    if anchor_data is not None:
        name2idx = dict(zip(cnames, np.arange(len(cnames))))
        anchor_data['cell1'] = anchor_data.name1.apply(lambda x:name2idx[x])
        anchor_data['cell2'] = anchor_data.name2.apply(lambda x:name2idx[x])

    X = sps.csr_matrix(adata.X)    # some times 
    metadata = adata.obs.copy()
    cnames = adata.obs_names
    gnames = adata.var_names

    # reform data type, needed by evaluation methods
    metadata[configs.batch_key] = metadata[configs.batch_key].astype('category')
    metadata[configs.label_key] = metadata[configs.label_key].astype('category')

    return X, gnames, cnames, metadata, anchor_data


def LouvainPipe(sps_x, df_meta, npcs=50, n_neighbors=15, r=1.):
    print(f'preprocessing dataset, shape=({sps_x.shape[0]}, {sps_x.shape[1]})')

    # compute hvg first, anyway
    adata = sc.AnnData(sps.csr_matrix(sps_x.T))  # transposed before

    sc.pp.filter_genes(adata, min_cells=configs.min_cells) 
    sc.pp.normalize_total(adata, target_sum=configs.scale_factor)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=min(adata.shape[1]-1, configs.n_hvgs), 
                                min_mean=0.0125, max_mean=3, min_disp=0.5,
                                )

    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)   # X -> array
    sc.pp.pca(adata_hvg, n_comps=npcs) # svd_solver='arpack' not accept sparse input

    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=npcs)
    sc.tl.louvain(adata_hvg, resolution=r, key_added='louvain')
    return adata_hvg.obs['louvain'].values

def hvgPipe(X, meta, scale=False, n_neighbors=15, npcs=50):
    adata = sc.AnnData(X)
    adata.obs = meta.copy()

    if scale:
        sc.pp.scale(adata, max_value=None)

    sc.pp.pca(adata, n_comps=npcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)

    return adata

def embPipe(X, meta, n_neighbors=15):
    adata = sc.AnnData(X)
    adata.obs = meta.copy()

    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata)

    return adata

