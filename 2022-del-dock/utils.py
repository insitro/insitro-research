# Copyright (C) 2022 Insitro, Inc. This software and any derivative works are licensed under the 
# terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC 4.0), 
# accessible at https://creativecommons.org/licenses/by-nc/4.0/legalcode

import torch


class ExpAct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


activations = {
    "relu": torch.nn.ReLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "softplus": torch.nn.Softplus,
    "gelu": torch.nn.GELU,
    "elu": torch.nn.ELU,
    "sigmoid": torch.nn.Sigmoid,
    "identity": torch.nn.Identity,
    "exp": ExpAct,
}


def activation_factory(activation_string):
    return activations[activation_string.lower()]


class ResidualNLayerMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_act, n_layers=1, dropout=0.0):
        super().__init__()

        self.main_modules = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    hidden_act(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(hidden_dim, in_dim),
                    hidden_act(),
                    torch.nn.Dropout(p=dropout),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.main_modules:
            x = layer(x) + x
        return x
