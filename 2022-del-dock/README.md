# DEL-DOCK

Code for: https://arxiv.org/abs/2212.00136
### DEL-Dock: Molecular Docking-Enabled Modeling of DNA-Encoded Libraries ###
#### Kirill Shmilovich, Benson Chen, Theofanis Karaletos, and Mohammad M. Sultan ####

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

Environment file can be found at `env.yml`

# Loading pretrained model

Code for loading the pretrained model from the paper (`ckpt.ckpt`) is available in `load_model.ipynb`

Ensure that all required data is loaded and unzipped:

``unzip JACS_full.zip``


# Training model

Code for training models from scratch is available in `train_model.ipynb`
