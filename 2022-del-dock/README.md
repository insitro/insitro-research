# DEL-DOCK

This repository contains code for the paper [DEL-Dock: Molecular Docking-Enabled Modeling of DNA-Encoded Libraries](https://arxiv.org/abs/2212.00136) by Kirill Shmilovich, Benson Chen, Theofanis Karaletos, and Mohammad M. Sultan.


# Dependencies

Python 3.7

Pyro (1.8.0)

PyTorch Lightning (1.6.4)

PyTorch (1.9.1)

NumPy (1.21)

rdkit (2022.03.5)

tqdm (4.64.0)

scipy (1.7.3)

wandb (0.12.21)

pandas (1.3.5)

The environment file can be found at `env.yml`


``conda env create -f env.yml``

# Getting the data

Running training and/or performing evaluation requires downloading the preprocessed CNN features of all the docked poses for molecules in the training DEL dataset and the evaluation dataset. These are referenced throughout the code as `cnn_feats_JACS_full.pt` (~2 GB) and `cnn_feats_hca_ChEMBL.pt` (~60 MB) and can be downloaded from the following links:

https://s3.us-west-2.amazonaws.com/insitro-research-2022-del-dock/cnn_feats_JACS_full.pt

https://s3.us-west-2.amazonaws.com/insitro-research-2022-del-dock/cnn_feats_hca_ChEMBL.pt

These CNN features are generated using a pretrained [GNINA](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2) model to process the docked pose configuration files. These CNN features are calculated using [gnina-torch](https://github.com/RMeli/gnina-torch) with this processing demonstrated in [get_cnn_feats.ipynb](get_cnn_feats.ipynb). Running this notebook requires all the docked pose configuration (.sdf) files for the training and evaluation datasets that are referenced in the code as directories called `docked_jacs_full/` (~1 GB, zipped) and `ChEMBLeval_docking_results_clean/` (~25 MB, zipped) and can be downloaded from the following links:

https://s3.us-west-2.amazonaws.com/insitro-research-2022-del-dock/JACS_docked.zip

https://s3.us-west-2.amazonaws.com/insitro-research-2022-del-dock/ChEMBL_docked.zip


# Loading pretrained model

Pretrianed model weights and checkpoints are provided at `ckpt.ckpt` with accompanying code demonstrating how to load the model and run evaluation available at [load_model.ipynb](load_model.ipynb).

*Ensure that all required data is downloaded from above and unzipped accordingly.*

# Training a model from scratch

Code for training models from scratch is available at [train_model.ipynb](train_model.ipynb). 

Importantly, training the model from scratch is contingent on CNN features for the docked poses and associated DEL counts for molecules in the training dataset (provided within `JACS_full.zip`). Please see **Getting the data** section above for details and download location links.

