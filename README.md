# GLOBE
 A contrastive learning-based batch correction framework for integration of scRNA-seq datasets.

## Overview
GLOBE accomplishes batch effect removal and gene matrices integration in two steps: 
1. Mapping multiple datasets to a latent conseus space through contrastive learning. First, GLOBE employs MNNs to customize translation transformations for each cell, insuring the stability for approximating batch effects. Then, GLOBE utilizes a contrastive loss which is both hardness-aware and consistency-aware to learn translation-invariant representations. Finally, the learnt representations can achieve some invariance to real batch effects. 

2. Based on the aliged latent space, a MNN graph among datasets is built. GLOBE leverages the MNN graph to integrate datasets in the original gene expression space. 


## Installation

Firstly, clone this repository.

```
git clone https://github.com/CSUBioGroup/GLOBE.git
cd GLOBE/
```

Please make sure Pytorch is installed in your python environment (our test version: pytorch 1.6.1 and 1.7.1). Then installing the dependencies:
```
pip install -r requirements.txt
```

## Datasets
All raw datasets used in our paper can be found in:
* [dataset 1,2,3,4,6 and simulation](https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking): Mouse Cell Atlas, PBMC, Cell line, Mouse Retina, Human Pancreas and Simulation datasets.

* [dataset 5](https://drive.google.com/uc?id=17ou8nVfrTYXJhA_a-OJOEm03zfbfBgxH): Tabula Muris

* [dataset 7, 8](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968): Human Immune Cells, Human Lung Cells

* [Mouse Neocortex](https://github.com/jaydu1/VITAE)

Some of data used in our experiments can be found in [`data`](data/). Complete data can be found in [`zenodo`](https://zenodo.org/record/6395618)

## Usage
We provided some demos ([`Simulation, PBMC, Cell Line, Pancreas, Neocortex`](demo/)) to demonstrate usage of GLOBE. Following is a brief decription.

### Build Dataset and Model
```Python
    # prepare dataset
    sps_x, genes, cells, metadata = prepare_dataset('../data/Pancreas')  # loading Pancreas dataset
    anchor_data = prepare_mnn('../data/Pancreas')  # loading MNNs exported from seurat3

    # preprocessing
    sps_x, pp_genes, pp_cells, pp_metadata, pp_anchor_data = preprocess_data(sps_x, 
                                                                cnames=cells, 
                                                                gnames=genes, 
                                                                metadata=metadata, 
                                                                anchor_data=anchor_data,
                                                                select_hvg=True, 
                                                                scale=False) 


    # init model
    globe = GLOBE(
            mode='unsupervised-GLOBE',    # default
            exp_id='Pancreas_v1',   # used for creating logs
            gpu='0',              
        )

    # create dataset and loader
    globe.build_dataset(
        sps_x=sps_x, 
        gnames=pp_genes, 
        cnames=pp_cells, 
        metadata=pp_metadata, 
        anchor_data=pp_anchor_data,
        add_noise=False,         # whether to add noise during translation
        use_knn=False            # whether to use knn as positives
    )

    # initializing neural network
    globe.build_model(
        lat_dim=128,
        proj_dim=64,             # only works for header='mlp'
        header=None,             # setting 'mlp' to add projection header
        block_level=1,           # basic block
    )
```

### Training and inference
``` Python
    # to be released
    # training
    globe.train(
        temp=0.1,                       # temperature parameter
        lr=1e-5,                        # learning rate
        batch_size=256,
        epochs=80,
        plot_loss=True,                 # whether to plot loss curve
        save_freq=10,                   # interval for saving weights
        weight_decay=1e-4,
        num_workers=6,
    )

    # loading checkpoints
    globe.load_ckpt(30)

    # inference for latent representations
    lat_data, _, metadata1 = globe.evaluate(batch_size=256, num_workers=6)
    prj_data = normalize(lat_data, axis=1)      # l2-normalization for each cell

    ad_e = embPipe(prj_data, metadata1)         # GLOBE.preprocessing.embPipe
    sc.pl.umap(ad_e, color=['batchlb', 'CellType'])

    # The second step
    # integrating gene matrices
    dec_rec, dec_metas = globe.integrate_gene(rec=prj_data, knn=10, sigma=10, alpha=0.1)
    ad_g = hvgPipe(dec_rec, dec_metas, scale=False)
    sc.pl.umap(ad_g, color=['batchlb', 'CellType'])

```
