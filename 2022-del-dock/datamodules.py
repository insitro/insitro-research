# Copyright (C) 2022 Insitro, Inc. This software and any derivative works are licensed under the 
# terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC 4.0), 
# accessible at https://creativecommons.org/licenses/by-nc/4.0/legalcode
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        percent_train: float = 0.7,
        percent_val: float = 0.1,
        batch_size: int = 32,
        test_batch_size: int = 1028,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert (percent_train + percent_val) < 1.0
        print(
            f"train/val/test split: {percent_train:.2g}/{percent_val:.2g}/{1 - (percent_train+percent_val):.2g}"
        )

        self.n_train = round(self.num_pts * percent_train)
        self.n_val = round(self.num_pts * percent_val)
        self.n_test = self.num_pts - (self.n_train + self.n_val)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _get_dataloader(self, data, batch_size: int, shuffle: bool):
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._get_dataloader(
            data=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._get_dataloader(
            data=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self._get_dataloader(
            data=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )


class MultiEvalModule(BaseDataModule):
    def _get_eval_loader(self):
        return self._get_dataloader(
            self.eval_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        val_loader = super().val_dataloader()

        eval_loader = self._get_eval_loader()
        return [val_loader, eval_loader]

    def test_dataloader(self):
        test_loader = super().test_dataloader()

        eval_loader = self._get_eval_loader()
        return [test_loader, eval_loader]


class GraphDataMixin:
    def _get_data(
        self,
        num_pts,
        targets_dict,
        stage=None,
        cnn_feats_fname=None,
        **kwargs,
    ):
        print(f"Getting CNN feats from : {cnn_feats_fname})")

        if cnn_feats_fname is not None:
            cnn_feats = torch.load(cnn_feats_fname)

        fps = list()
        for smi in tqdm(
            targets_dict["smiles_strings"],
            desc=f"Calculating {stage} fingerprints",
            leave=False,
        ):
            m = Chem.MolFromSmiles(smi)

            arr = np.zeros((0,), dtype=np.int8)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, useChirality=True)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(torch.FloatTensor(arr.copy()).view(1, -1))
        del targets_dict["smiles_strings"]

        data = list()
        for pose_idx in tqdm(range(self.poses), desc="Getting graph poses"):
            data_pose = list()
            for ii in tqdm(range(num_pts), desc=stage):
                smiles = fps[ii]
                data_point_targets = {
                    k: v[ii] for k, v in targets_dict.items()
                }  # training targets (enrichments, ki, etc..)
                if cnn_feats_fname is not None:
                    data_point_targets.update(
                        {
                            "cnn_feats": torch.FloatTensor(
                                cnn_feats[ii][pose_idx]
                            ).view(1, -1)
                        }
                    )
                data_point = dict(
                    pose_idx=pose_idx,
                    data_idx=ii,
                    smiles=smiles,
                    **data_point_targets,
                )
                data_pose.append(data_point)
            data.append(data_pose)

        if self.poses > 1:
            return list(zip(*data))
        else:
            return data_pose


class GraphChEMBLEvalDataModule:
    def get_eval_data(
        self,
        df_eval_fname="/home/ec2-user/df_eval_data.csv",
        **kwargs,
    ):
        self.df_eval = pd.read_csv(df_eval_fname)
        print(f"Using eval data: {self.df_eval.shape[0]} samples")

        self.ki = self.df_eval["Ki (nM)"].values

        self._get_eval_dataset(**kwargs)

    def _get_eval_dataset(
        self,
        **kwargs,
    ):
        cnn_feats_eval_fname = (
            kwargs["cnn_feats_eval_fname"]
            if "cnn_feats_eval_fname" in kwargs.keys()
            else None
        )
        self.eval_dataset = self._get_data(
            num_pts=self.df_eval.shape[0],
            targets_dict={
                "ki": self.ki,
                "smiles_strings": self.df_eval["smiles"].values,
            },
            stage="Getting eval graph data",
            cnn_feats_fname=cnn_feats_eval_fname,
            **kwargs,
        )


class JACSDataMixin_counts:
    def _get_dataset_targets(
        self,
        dataset_csv_fname,
        **kwargs,
    ):
        self.dataset_csv_fname = dataset_csv_fname
        self.dataset_df = pd.read_csv(self.dataset_csv_fname)
        self.num_pts = self.dataset_df.shape[0]
        print(f"Number of datapoints: {self.num_pts}; using JACS dataset with counts")

        matrix_counts = (
            self.dataset_df[
                ["ca9_beads_r1_tpm_normalized", "ca9_beads_r2_tpm_normalized"]
            ]
            .to_numpy()
            .astype(int)
        )
        target_counts = (
            self.dataset_df[
                [
                    "ca9_exp_r1_tpm_normalized",
                    "ca9_exp_r2_tpm_normalized",
                    "ca9_exp_r3_tpm_normalized",
                    "ca9_exp_r4_tpm_normalized",
                ]
            ]
            .to_numpy()
            .astype(int)
        )

        self.matrix_counts = torch.LongTensor(matrix_counts).unsqueeze(1)
        self.target_counts = torch.LongTensor(target_counts).unsqueeze(1)
        self.targets_dict = {
            "matrix_counts": self.matrix_counts,
            "target_counts": self.target_counts,
            "smiles_strings": self.dataset_df["smiles"].values,
        }


class DataModule(
    MultiEvalModule,
    GraphChEMBLEvalDataModule,
    GraphDataMixin,
):
    def __init__(
        self,
        source_data: str = "jacs",
        source_eval: str = "CAIX",
        poses: int = 20,
        splits_fname=None,
        cnn_feats_train_fname=None,
        **kwargs,
    ):
        self.poses = poses
        self.n1 = self.n2 = None
        if source_data.lower() == "jacs_counts":
            JACSDataMixin_counts._get_dataset_targets(self, **kwargs)
        else:
            raise NotImplementedError()

        self.data = self._get_data(
            num_pts=self.num_pts,
            targets_dict=self.targets_dict,
            stage="Getting graph data",
            cnn_feats_fname=cnn_feats_train_fname,
            **kwargs,
        )

        MultiEvalModule.__init__(self, **kwargs)

        if source_eval.lower() == "caix":
            GraphChEMBLEvalDataModule.get_eval_data(self, **kwargs)
        else:
            raise NotImplementedError()

        if splits_fname is None:
            idxs = np.random.permutation(self.num_pts)
            self.train_idxs = idxs[: self.n_train]
            self.val_idxs = idxs[self.n_train : self.n_train + self.n_val]
            self.test_idxs = idxs[self.n_train + self.n_val :]
        else:
            print(f"Using splits from: {splits_fname}")
            splits = np.load(splits_fname)
            self.train_idxs = splits["train_idxs"]
            self.val_idxs = splits["val_idxs"]
            self.test_idxs = splits["test_idxs"]

    def setup(self, stage=None):
        if not hasattr(self, "train_data"):
            self.train_data = self.train_dataset = [
                self.data[ii] for ii in self.train_idxs
            ]
            self.val_data = self.val_dataset = [self.data[ii] for ii in self.val_idxs]
            self.test_data = self.test_dataset = [
                self.data[ii] for ii in self.test_idxs
            ]

    def _get_dataloader(self, data, shuffle, batch_size):
        return torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pose_collate if self.poses > 1 else None,
        )

    @staticmethod
    def pose_collate(data):
        dat = [x for xs in data for x in xs]
        for i, d in enumerate(dat):
            if i == 0:
                out = d.copy()
            elif i == 1:
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        out[k] = [out[k], v]
                    else:
                        out[k] = torch.cat((out[k], v))
            else:
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        out[k].append(v)
                    else:
                        out[k] = torch.cat((out[k], v))

        for k, v in out.items():
            if not isinstance(v, torch.Tensor):
                out[k] = torch.tensor(v)
        return out
