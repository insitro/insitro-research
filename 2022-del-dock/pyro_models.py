# Copyright (C) 2022 Insitro, Inc. This software and any derivative works are licensed under the 
# terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC 4.0), 
# accessible at https://creativecommons.org/licenses/by-nc/4.0/legalcode
import pyro
import torch
from models import BaseModel
from torch import nn
from pyro.nn.module import to_pyro_module_
from utils import activation_factory


class PyroModel(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self._init_pyro_model()

    def _init_pyro_model(self):

        final_act = activation_factory(self.hparams.final_act)

        self.enrichment_head = nn.Sequential(
            nn.Linear(2 * self.hparams.hidden_dim, 1),
            final_act(),
        )
        self.matrix_head = nn.Sequential(
            nn.Linear(2 * self.hparams.hidden_dim, 1),
            final_act(),
        )
        self.zero_prob = torch.tensor(
            [0.5488], device=torch.device("cuda") if torch.cuda.is_available() else self.device, dtype=torch.float
        )
        self.zero_prob_mat = torch.tensor(
            [0.007528], device=torch.device("cuda") if torch.cuda.is_available() else self.device, dtype=torch.float
        )
        to_pyro_module_(self)
        self.svi = self._get_pyro_svi()

    def pyro_model(self, batch):
        x, target_counts, matrix_counts = self._process_batch(batch, None)
        target_effect, matrix_effect = self.forward(*x, return_matrix_effect=True)

        bs = matrix_counts.size(0)
        n_matrix_counts = matrix_counts.size(1)
        n_target_counts = target_counts.size(1)

        with pyro.plate("data", bs):
            for i in range(n_matrix_counts):
                pyro.sample(
                    "matrix_obs_%d" % i,
                    pyro.distributions.ZeroInflatedPoisson(
                        matrix_effect,
                        gate=self.zero_prob_mat,
                    ),
                    obs=matrix_counts[:, i],
                )

            for i in range(n_target_counts):
                pyro.sample(
                    "target_obs_%d" % i,
                    pyro.distributions.ZeroInflatedPoisson(
                        matrix_effect + target_effect,
                        gate=self.zero_prob,
                    ),
                    obs=target_counts[:, i],
                )

        return target_effect

    def _get_pyro_svi(self):
        opts = {
            "lr": self.hparams.learning_rate,
            "clip_norm": self.hparams.clip_norm,
            "betas": (0.95, 0.999),
            "weight_decay": self.hparams.weight_decay,
            "lrd": self.hparams.lrd_gamma ** (1 / self.hparams.lrd_num_steps),
        }

        optim = pyro.optim.ClippedAdam(opts)
        svi = pyro.infer.SVI(
            model=self.pyro_model,
            guide=pyro.infer.autoguide.AutoNormal(self.pyro_model),
            optim=optim,
            loss=pyro.infer.Trace_ELBO(),
        )
        return svi

    def training_step(self, batch, batch_idx):

        loss = self.svi.step(batch)

        loss = torch.tensor(loss).requires_grad_(True)

        bs = batch["cnn_feats"].size(0)

        self.log("train_loss", loss / bs, batch_size=bs)

        return loss / bs

    def _common_val_step(self, batch, batch_idx, stage, return_attn_scores=False):
        x, *_ = self._process_batch(batch, batch_idx)

        if return_attn_scores:
            R_pred, attn_scores = self.forward(
                *x, return_attn_scores=return_attn_scores
            )
        else:
            R_pred = self.forward(*x)

        bs = batch["cnn_feats"].size(0)

        loss = self.svi.evaluate_loss(batch)
        self.log(f"{stage}_loss", loss / bs, batch_size=bs)

        if return_attn_scores:
            return R_pred, attn_scores
        else:
            return R_pred
