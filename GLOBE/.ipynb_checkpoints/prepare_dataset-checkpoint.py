import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from os.path import join

from GLOBE.utils import py_read_data, load_meta_txt, load_meta_txt7
from GLOBE.config import Config

configs = Config()


def prepare_MouseCellAtlas(data_root):
    data_name = 'filtered_total_batch1_seqwell_batch2_10x'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'filtered_total_sample_ext_organ_celltype_batch.txt'))
    df_meta['CellType'] = df_meta['ct']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_Pancreas(data_root):
    data_name = 'myData_pancreatic_5batches'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'mySample_pancreatic_5batches.txt'))
    df_meta['CellType'] = df_meta['celltype']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_PBMC(data_root):
    sps_x1, gene_name1, cell_name1 = py_read_data(data_root, 'b1_exprs')
    sps_x2, gene_name2, cell_name2 = py_read_data(data_root, 'b2_exprs')

    sps_x = sps.hstack([sps_x1, sps_x2])
    cell_name = np.hstack((cell_name1, cell_name2))

    assert np.all(gene_name1 == gene_name2), 'gene order not match'
    gene_name = gene_name1

    df_meta1 = load_meta_txt(join(data_root, 'b1_celltype.txt'))
    df_meta2 = load_meta_txt(join(data_root, 'b2_celltype.txt'))
    df_meta1['batchlb'] = 'Batch1'
    df_meta2['batchlb'] = 'Batch2'

    df_meta = pd.concat([df_meta1, df_meta2])

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_CellLine(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b3_exprs_filename = "b3_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"
    b3_celltype_filename = "b3_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_dir, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    expr_mat3, g3, c3 = py_read_data(data_root, b3_exprs_filename)
    metadata3 = pd.read_csv(join(data_root, b3_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'
    metadata3['batchlb'] = 'Batch_3'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2, c3])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2, metadata3])
    sps_x = sps.hstack([expr_mat1, expr_mat2, expr_mat3])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_MouseRetina(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_root, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2])
    sps_x = sps.hstack([expr_mat1, expr_mat2])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_Simulation(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'Batch'
    label_key = 'Group'

    # manually switch to counts_all.txt
    # ensure row is gene
    X = pd.read_csv(join(data_root, 'counts.txt'), sep='\t',header=0, index_col=0)  # row is cell
    X = X.T   # to gene

    metadata = pd.read_csv(join(data_root, 'cellinfo.txt'), header=0, index_col=0, sep='\t')
    metadata[configs.batch_key] = metadata[batch_key]
    metadata[configs.label_key] = metadata[label_key]

    return X, X.index.values, X.columns.values, metadata

def prepare_Lung(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'batch'
    label_key = 'cell_type'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Lung_atlas_public.h5ad'))

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmHuman(data_root):
    batch_key = 'batch'
    label_key = 'final_annotation'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Immune_ALL_human.h5ad'))

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_Muris(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_filter.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Neo(data_root):
    # batch_key = 'batch'
    label_key = 'grouping'

    import h5py
    filename = join(data_root, 'mouse_brain_merged.h5')

    with h5py.File(filename, "r") as f:
        # List all groups
        cell_name = list(map(lambda x:x.decode('utf-8'), f['cell_ids'][...]))
        gene_name = list(map(lambda x:x.decode('utf-8'), f['gene_names'][...]))
        
        X = sps.csr_matrix(f['count'][...].T)  # transpose to (genes, cells)
        types = list(map(lambda x:x.decode('utf-8'), f['grouping'][...]))

    df_meta = pd.DataFrame(types, index=cell_name, columns=[configs.label_key])
    df_meta[configs.batch_key] = 'Batch_B'
    df_meta.iloc[:10261, -1] = 'Batch_A'     
    return X, gene_name, cell_name, df_meta


def prepare_dataset(data_dir):
    dataset_name = data_dir.split('/')[-1]
    func_dict = {
                    'MouseCellAtlas': prepare_MouseCellAtlas, 
                    'Pancreas': prepare_Pancreas, 
                    'PBMC': prepare_PBMC, 
                    'CellLine': prepare_CellLine, 
                    'MouseRetina': prepare_MouseRetina, 
                    'Lung': prepare_Lung,
                    'ImmHuman': prepare_ImmHuman,
                    'Muris': prepare_Muris,
                    'Neocortex': prepare_Neo
                    # 'Simulation/*': prepare_Simulation
    }

    # dataset 3 
    return func_dict.get(dataset_name, prepare_Simulation)(data_dir)


def prepare_mnn(data_dir, fname=None):
    # dataset_name = data_dir.split('/')[-1]
    fname = 'seuratAnchors.csv' if fname is None else fname
    anchor_path = join(data_dir, fname)
    print('reading Anchors from ', anchor_path)
    df = pd.read_csv(anchor_path, sep=',', index_col=0)

    return df

    # convert cell name in metadata to global indices
    # name2idx = dict(zip(self.cname, np.arange(self.n_sample)))

    # self.anchor_metadata['cell1'] = self.anchor_metadata.name1.apply(lambda x:name2idx[x])
    # self.anchor_metadata['cell2'] = self.anchor_metadata.name2.apply(lambda x:name2idx[x])






