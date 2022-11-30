# Copyright (C) 2022 Insitro, Inc. This software and any derivative works are licensed under the 
# terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC 4.0), 
# accessible at https://creativecommons.org/licenses/by-nc/4.0/legalcode
from abc import abstractmethod
from ema import ExponentialMovingAverage
import pytorch_lightning as pl
from torch import nn
import torch
from utils import (
    ResidualNLayerMLP,
    activation_factory,
)
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
import wandb


class LightningModule_EMABase(pl.LightningModule):
    @abstractmethod
    def configure_optimizers(self):
        if self.hparams.use_ema:
            self.ema = ExponentialMovingAverage(
                self.parameters(), decay=self.hparams.ema_decay
            )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.use_ema:
            self.ema.update(self.parameters())

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["ema"] = self.ema.state_dict()
            return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint.keys():
            self.ema = ExponentialMovingAverage(
                self.parameters(), self.hparams.ema_decay
            )
            self.ema.load_state_dict(checkpoint["ema"])

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


class EMA_module(LightningModule_EMABase):
    def validation_step(self, batch, batch_idx, return_attn_scores=False):
        with self.ema.average_parameters():
            ema_preds = self._common_val_step(
                batch, batch_idx, stage="val_EMA", return_attn_scores=return_attn_scores
            )

        preds = self._common_val_step(
            batch, batch_idx, stage="val", return_attn_scores=return_attn_scores
        )

        if return_attn_scores:
            return preds[1], ema_preds[1]  # only return attn scores
        else:
            return preds[0], ema_preds[0]  # else return raw predictions

    def test_step(self, batch, batch_idx):
        self._common_val_step(batch, batch_idx, stage="test")
        with self.ema.average_parameters():
            self._common_val_step(batch, batch_idx, stage="test_EMA")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x = self._process_batch(batch, batch_idx)[0]
        with self.ema.average_parameters():
            R_pred = self.forward(*x)
        return R_pred


class EMA_ChEMBL_module(EMA_module):
    @staticmethod
    def _get_pearson_spearman(pred, true):
        valid_idxs = torch.where(~torch.isnan(true))[0]
        if len(valid_idxs) < 2:
            return float("nan"), float("nan")
        rho = pearsonr(
            pred[valid_idxs].detach().cpu().numpy(),
            true[valid_idxs].detach().cpu().numpy(),
        )[0]
        spear = spearmanr(
            pred[valid_idxs].detach().cpu().numpy(),
            true[valid_idxs].detach().cpu().numpy(),
        )[0]
        return rho, spear

    def _common_eval_step(self, batch, batch_idx, stage):
        (
            x,
            ki,
        ) = self._process_batch(batch, batch_idx)

        R_pred = self.forward(*x).flatten()
        self._log_pearson_spearman(R_pred, ki, stage)

    def _log_pearson_spearman(self, R_pred, ki, stage):
        rho, spear = self._get_pearson_spearman(R_pred, ki)
        self.log(f"{stage}_eval_ki_pearson", rho, batch_size=len(R_pred))
        self.log(f"{stage}_eval_ki_spearman", spear, batch_size=len(R_pred))

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            super().validation_step(batch, batch_idx)
        elif dataloader_idx == 1:
            x, ki = self._process_batch(batch, batch_idx)
            with self.ema.average_parameters():
                R_pred_ema, matrix_effect = self.forward(*x, return_matrix_effect=True)
            return R_pred_ema.view(-1), ki, matrix_effect.view(-1)

    def validation_epoch_end(self, outputs):
        out = outputs[1]  # only eval dataset with pearson spearman issues
        R_pred = torch.cat([x[0] for x in out])
        ki = torch.cat([x[1] for x in out])
        self._log_pearson_spearman(R_pred, ki, stage="VAL_EMA")

        if not self.trainer.sanity_checking:
            if not hasattr(self, "_size_mask"):
                df_eval = self.trainer.datamodule.df_eval
                mws = np.array(
                    [
                        Descriptors.MolWt(Chem.MolFromSmiles(smi))
                        for smi in df_eval["smiles"]
                    ]
                )
                self._size_mask = np.logical_and(
                    mws > 417, mws < 517
                )  # 10th and 90th quartiles of training data

            rho, spear = self._get_pearson_spearman(
                R_pred[self._size_mask], ki[self._size_mask]
            )
            self.log(
                "val_EMA_eval_ki_pearson_subset",
                rho,
                batch_size=len(R_pred[self._size_mask]),
            )
            self.log(
                "val_EMA_eval_ki_spearman_subset",
                spear,
                batch_size=len(R_pred[self._size_mask]),
            )
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(
                    {
                        "ChEMBL target preds": wandb.Histogram(
                            R_pred.flatten().detach().cpu()
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )
                self.logger.experiment.log(
                    {
                        "ChEMBL target preds (subset)": wandb.Histogram(
                            R_pred[self._size_mask].flatten().detach().cpu()
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )

                matrix_effects = torch.cat([x[2] for x in out])
                self.logger.experiment.log(
                    {
                        "ChEMBL matrix preds": wandb.Histogram(
                            matrix_effects.flatten().detach().cpu()
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )
                self.logger.experiment.log(
                    {
                        "ChEMBL matrix preds (subset)": wandb.Histogram(
                            matrix_effects[self._size_mask].flatten().detach().cpu()
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )

    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            super().test_step(batch, batch_idx)
        elif dataloader_idx == 1:
            x, ki = self._process_batch(batch, batch_idx)
            with self.ema.average_parameters():
                R_pred_ema = self.forward(*x).flatten()
            return R_pred_ema, ki

    def test_epoch_end(self, outputs):
        out = outputs[1]  # only eval dataset with pearson spearman issues
        R_pred = torch.cat([x[0] for x in out])
        ki = torch.cat([x[1] for x in out])
        self._log_pearson_spearman(R_pred, ki, stage="test_EMA")


class BaseModel(EMA_ChEMBL_module):
    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        hidden_act: str = "leakyrelu",
        final_act: str = "exp",
        poses: int = 20,
        pose_reduce: str = "attn_gated",
        n_layers: int = 2,
        clip_norm=1e-1,
        lrd_gamma=0.1,
        lrd_num_steps=1250,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.setup_model(kwargs)

    def setup_model(self, kwargs):
        hidden_act = activation_factory(self.hparams.hidden_act)

        self.cnn_embed = nn.Sequential(
            nn.Linear(self.hparams.use_cnn_feats, 2 * self.hparams.hidden_dim),
            hidden_act(),
            ResidualNLayerMLP(
                in_dim=2 * self.hparams.hidden_dim,
                hidden_dim=self.hparams.hidden_dim,
                hidden_act=hidden_act,
                n_layers=self.hparams.n_layers,
                dropout=self.hparams.dropout,
            ),
        )

        self.smiles_embed = nn.Sequential(
            nn.Linear(self.hparams.use_smiles, 2 * self.hparams.hidden_dim),
            hidden_act(),
            ResidualNLayerMLP(
                in_dim=2 * self.hparams.hidden_dim,
                hidden_dim=self.hparams.hidden_dim,
                hidden_act=hidden_act,
                n_layers=self.hparams.n_layers,
                dropout=self.hparams.dropout,
            ),
        )

        self.pose_attn_tanh = nn.Sequential(
            nn.Linear(2 * self.hparams.hidden_dim, self.hparams.hidden_dim, bias=False),
            nn.Tanh(),
        )
        self.pose_attn_sig = nn.Sequential(
            nn.Linear(2 * self.hparams.hidden_dim, self.hparams.hidden_dim, bias=False),
            nn.Sigmoid(),
        )

        self.pose_attn = nn.Linear(self.hparams.hidden_dim, 1, bias=False)

        self.post_add_layer = ResidualNLayerMLP(
            in_dim=2 * self.hparams.hidden_dim,
            hidden_dim=self.hparams.hidden_dim,
            hidden_act=hidden_act,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )

    def forward(
        self,
        batch,
        return_attn_scores=False,
        return_matrix_effect=False,
    ):

        x = self.cnn_embed(batch["cnn_feats"])

        fps_embed = self.smiles_embed(
            batch["smiles"].view(-1, self.hparams.poses, batch["smiles"].size(-1))[:, 0]
        )

        x += (
            fps_embed.unsqueeze(1)
            .repeat(1, self.hparams.poses, 1)
            .view(-1, fps_embed.size(-1))
        )

        x = self.post_add_layer(x)

        if self.hparams.poses > 1:
            x = x.view(-1, self.hparams.poses, x.shape[-1])  # [B, poses, dim]

            if self.hparams.pose_reduce.lower() == "mean":
                x = x.mean(1)  # mean over the poses, [B, dim]
            elif self.hparams.pose_reduce.lower() == "attn_gated":
                a_pose = torch.softmax(
                    self.pose_attn(self.pose_attn_tanh(x) * self.pose_attn_sig(x)),
                    dim=1,
                )
                x = (x * a_pose).sum(1)

        matrix_effect = self.matrix_head(fps_embed).view(-1)
        R_pred = self.enrichment_head(x).view(-1)

        if return_matrix_effect:
            return R_pred, matrix_effect

        if return_attn_scores:
            return R_pred, a_pose

        return R_pred

    def _process_batch(self, batch, batch_idx):
        multi_pose_fixer = self._fix_multi_pose

        if "ki" in batch.keys():
            return (
                (batch,),
                multi_pose_fixer(batch["ki"]),
            )

        elif "matrix_counts" in batch.keys():
            if self.hparams.poses == 1:
                return (
                    (batch,),
                    batch["target_counts"],
                    batch["matrix_counts"],
                )
            else:
                return (
                    (batch,),
                    batch["target_counts"].view(
                        -1, self.hparams.poses, batch["target_counts"].size(-1)
                    )[:, 0],
                    batch["matrix_counts"].view(
                        -1, self.hparams.poses, batch["matrix_counts"].size(-1)
                    )[:, 0],
                )

    def _fix_multi_pose(self, data):
        if self.hparams.poses > 1:
            return data.view(-1, self.hparams.poses)[:, 0]
        return data

    def configure_optimizers(self):
        super().configure_optimizers()

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
