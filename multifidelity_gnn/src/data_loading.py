import torch
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, List, Tuple, Optional

from .chemprop_featurisation import atom_features, bond_features, get_atom_constants


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)


class GraphMoleculeDataset(TorchDataset):
    def __init__(
        self,
        csv_path: str,
        max_atom_num: int,
        smiles_column_name: str,
        label_column_name: Union[str, List[str]],
        auxiliary_data_column_name: Optional[str] = None,
        lbl_or_emb: str = "lbl",
        scaler: Optional[StandardScaler] = None,
        id_column: Optional[str] = None,
    ):
        super().__init__()
        assert lbl_or_emb in [None, "lbl", "emb"]
        self.df = pd.read_csv(csv_path)
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.atom_constants = get_atom_constants(max_atom_num)
        self.num_atom_features = (
            sum(len(choices) for choices in self.atom_constants.values()) + 2
        )
        self.num_bond_features = 13
        self.auxiliary_data_column_name = auxiliary_data_column_name
        self.lbl_or_emb = lbl_or_emb
        self.scaler = scaler
        self.id_column = id_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: Union[torch.Tensor, slice, List]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            slice_step = idx.step if idx.step else 1
            idx = list(range(idx.start, idx.stop, slice_step))
        if not isinstance(idx, list):
            idx = [idx]

        selected = self.df.iloc[idx]
        if self.id_column:
            ids = selected[self.id_column].values
        smiles = selected[self.smiles_column_name].values

        if isinstance(self.label_column_name, (list, tuple)):
            num_tasks = len(self.label_column_name)
        else:
            num_tasks = 1
        targets = selected[self.label_column_name].values

        if self.scaler is not None:
            labels = torch.Tensor(self.scaler.transform(targets.reshape(-1, num_tasks)))
        else:
            labels = torch.Tensor(targets)

        smiles = [remove_smiles_stereo(s) for s in smiles]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        aux_data = None
        if self.auxiliary_data_column_name:
            column_values = selected[self.auxiliary_data_column_name].values

            if self.lbl_or_emb and self.lbl_or_emb == "lbl":
                aux_data = torch.Tensor(column_values)

            elif self.lbl_or_emb == "emb":
                # Need to parse the NumPy array from the string stored in the DataFrame
                aux_data = torch.Tensor(
                    np.stack(
                        np.fromstring(
                            selected[self.auxiliary_data_column_name].values[0][1:-1],
                            sep=", ",
                        )
                    )
                )


        atom_feat = [
            torch.Tensor(
                [atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]
            )
            for mol in rdkit_mols
        ]

        edge_index = []
        bond_feat = []

        for mol in rdkit_mols:
            ei = torch.nonzero(
                torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))
            ).T

            bf = torch.Tensor(
                [
                    bond_features(
                        mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())
                    )
                    for i in range(ei.shape[1])
                ]
            )

            edge_index.append(ei)
            bond_feat.append(bf)

        geometric_data_points = [
            GeometricData(
                x=atom_feat[i],
                edge_attr=bond_feat[i],
                edge_index=edge_index[i],
                y=labels[i],
                aux_data=aux_data,
                iden=ids[i],
            )
            for i in range(len(atom_feat))
        ]

        for i, data_point in enumerate(geometric_data_points):
            data_point.smiles = smiles[i]
            data_point.aux_data = np.array([]) if aux_data is None else aux_data

        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points


class GeometricDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        max_atom_num: int = 80,
        split: Tuple[float, float] = (0.9, 0.05),
        train_path: Optional[str] = None,
        separate_valid_path: Optional[str] = None,
        separate_test_path: Optional[str] = None,
        id_column: Optional[str] = None,
        num_cores: Tuple[int, int, int] = (12, 0, 12),
        smiles_column_name: str = "SMILES",
        label_column_name: Union[str, List[str]] = "SD",
        train_auxiliary_data_column_name: Optional[str] = None,
        lbl_or_emb: str = "lbl",
        eval_auxiliary_data_column_name: Optional[str] = None,
        use_standard_scaler=False,
    ):
        super().__init__()
        assert lbl_or_emb in [None, "lbl", "emb"]
        self.dataset = None
        self.train_path = train_path
        self.batch_size = batch_size
        self.seed = seed
        self.max_atom_num = max_atom_num
        self.split = split
        self.num_cores = num_cores
        self.separate_valid_path = separate_valid_path
        self.separate_test_path = separate_test_path
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.train_auxiliary_data_column_name = train_auxiliary_data_column_name
        self.eval_auxiliary_data_column_name = eval_auxiliary_data_column_name
        self.lbl_or_emb = lbl_or_emb
        self.id_column = id_column

        self.use_standard_scaler = use_standard_scaler

        self.scaler = None
        if self.use_standard_scaler:
            train_df = pd.read_csv(self.train_path)
            train_data = train_df[self.label_column_name].values

            scaler = StandardScaler()
            if train_data.ndim == 1:
                scaler = scaler.fit(np.expand_dims(train_data, axis=1))
            else:
                scaler = scaler.fit(train_data)

            del train_data
            del train_df

            self.scaler = scaler

    def get_scaler(self):
        return self.scaler

    def prepare_data(self):
        self.val = None
        self.test = None
        if self.train_path:
            self.dataset = GraphMoleculeDataset(
                csv_path=self.train_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.train_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )

            self.num_atom_features = self.dataset.num_atom_features
            self.num_bond_features = self.dataset.num_bond_features

        if self.separate_valid_path:
            self.val = GraphMoleculeDataset(
                csv_path=self.separate_valid_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.eval_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )

        if self.separate_test_path:
            self.test = GraphMoleculeDataset(
                csv_path=self.separate_test_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.eval_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )

            print("Assigned test dataset to self.test")

    def setup(self, stage: str = None):
        # Called on every GPU
        # Assumes prepare_data has been called
        self.train = self.dataset

    def train_dataloader(self, shuffle=True):
        if self.train:
            return GeometricDataLoader(
                self.train,
                self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_cores[0],
                pin_memory=True,
            )
        return None

    def val_dataloader(self):
        return GeometricDataLoader(
            self.val,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0 if not self.num_cores else self.num_cores[1],
        )

    def test_dataloader(self):
        if self.test:
            return GeometricDataLoader(
                self.test,
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0 if not self.num_cores else self.num_cores[2],
            )
        return None
