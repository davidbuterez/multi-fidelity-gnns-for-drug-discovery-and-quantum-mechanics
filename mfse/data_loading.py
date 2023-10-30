import torch
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, List, Tuple, Optional

from multifidelity_gnn.src.chemprop_featurisation import atom_features, bond_features, get_atom_constants


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)


FIDELITY_TO_NUM = {
    'SD': 0,
    'DR': 1,
    'Z-Score': 0,
    'pIC50': 1
}

# DTYPE_DICT = {
#     'CID': int,
#     'SD': float,
#     'DR': float,
#     'XC50': float,
#     'Activity': str,
#     'neut-smiles': str,
#     'Largest atomic number': int,
#     '# atoms': int
# }

class GraphMoleculeDataset(TorchDataset):
    def __init__(self, csv_path: str, max_atom_num: int, smiles_column_name: str,
                 label_column_name: Union[str, List[str]], id_column: Optional[str]=None,
                 scaler: Optional[StandardScaler]=None):

        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.id_column = id_column
        self.atom_constants = get_atom_constants(max_atom_num)
        self.num_atom_features = sum(len(choices) for choices in self.atom_constants.values()) + 2
        self.num_bond_features = 13
        self.scaler = scaler


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
        ids = selected[self.id_column].values
        smiles = selected[self.smiles_column_name].values
        try:
            if self.scaler:
                if selected[self.label_column_name].values.ndim == 1:
                    labels = torch.Tensor(self.scaler.transform(
                        np.expand_dims(selected[self.label_column_name].values, axis=1)
                    ))
                else:
                    labels = torch.Tensor(self.scaler.transform(selected[self.label_column_name].values))
            else:
                labels = torch.Tensor(selected[self.label_column_name].values)
        except:
            labels = torch.Tensor([0] * len(idx))

        smiles = [remove_smiles_stereo(s) for s in smiles]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        atom_feat = [torch.Tensor(
            [atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]
        ) for mol in rdkit_mols]

        edge_index = []
        bond_feat = []

        for mol in rdkit_mols:
            ei = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
            bf = torch.Tensor(
                [bond_features(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])]
            )

            edge_index.append(ei)
            bond_feat.append(bf)

        fidelities = torch.full(size=(len(atom_feat),), fill_value=FIDELITY_TO_NUM[self.label_column_name])

        geometric_data_points = [GeometricData(x=atom_feat[i], edge_attr=bond_feat[i], edge_index=edge_index[i],
                                               y=labels[i], iden=ids[i], fidelity=fidelities[i]) for i in range(len(atom_feat))]

        for i, data_point in enumerate(geometric_data_points):
            data_point.smiles = smiles[i]

        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points


class MixedFidelityGeometricDataModule(pl.LightningDataModule):
    def __init__(
            self,
            low_fidelity_path: str,
            high_fidelity_train_path: str,
            high_fidelity_valid_path: str,
            high_fidelity_test_path: str,
            lf_batch_size: int,
            hf_batch_size: int,
            max_atom_num: int,
            id_column: str,
            smiles_column: str,
            low_fidelity_target_label: str,
            high_fidelity_target_label: str,
            use_standard_scaler: bool = False,
            num_cores: Tuple[int, int, int] = (12, 12, 12),
            just_hf: bool = False
        ):

        super().__init__()
        self.low_fidelity_path = low_fidelity_path
        self.high_fidelity_train_path = high_fidelity_train_path
        self.high_fidelity_valid_path = high_fidelity_valid_path
        self.high_fidelity_test_path = high_fidelity_test_path
        self.low_fidelity_target_label = low_fidelity_target_label
        self.high_fidelity_target_label = high_fidelity_target_label

        self.lf_batch_size = lf_batch_size
        self.hf_batch_size = hf_batch_size
        self.max_atom_num = max_atom_num
        self.num_cores = num_cores
        self.smiles_column = smiles_column
        self.id_column = id_column

        self.use_standard_scaler = use_standard_scaler
        self.just_hf = just_hf

        self.lf_scaler = None
        self.hf_scaler = None

        if self.use_standard_scaler:
            if not self.just_hf:
                # Low fidelity scaler
                lf_df = pd.read_csv(self.low_fidelity_path)
                lf_data = lf_df[self.low_fidelity_target_label].values

                scaler = StandardScaler()
                if lf_data.ndim == 1:
                    scaler = scaler.fit(np.expand_dims(lf_data, axis=1))
                else:
                    scaler = scaler.fit(lf_data)

                del lf_data
                del lf_df

                self.lf_scaler = scaler

            # High fidelity scaler
            hf_df = pd.read_csv(self.high_fidelity_train_path)
            hf_data = hf_df[self.high_fidelity_target_label].values

            scaler = StandardScaler()
            if hf_data.ndim == 1:
                scaler = scaler.fit(np.expand_dims(hf_data, axis=1))
            else:
                scaler = scaler.fit(hf_data)

            del hf_data
            del hf_df

            self.hf_scaler = scaler


    def get_lf_scaler(self):
        return self.lf_scaler
    

    def get_hf_scaler(self):
        return self.hf_scaler


    def prepare_data(self):
        if not self.just_hf:
            self.lf_dataset = GraphMoleculeDataset(
                csv_path=self.low_fidelity_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column,
                label_column_name=self.low_fidelity_target_label,
                id_column=self.id_column,
                scaler=self.lf_scaler,   
            )

            self.num_atom_features = self.lf_dataset.num_atom_features
            self.num_bond_features = self.lf_dataset.num_bond_features


        self.train_hf_dataset = GraphMoleculeDataset(
            csv_path=self.high_fidelity_train_path,
            max_atom_num=self.max_atom_num,
            smiles_column_name=self.smiles_column,
            label_column_name=self.high_fidelity_target_label,
            id_column=self.id_column,
            scaler=self.hf_scaler,
        )

        self.val_hf_dataset = GraphMoleculeDataset(
            csv_path=self.high_fidelity_valid_path,
            max_atom_num=self.max_atom_num,
            smiles_column_name=self.smiles_column,
            label_column_name=self.high_fidelity_target_label,
            id_column=self.id_column,
            scaler=self.hf_scaler,
        )

        self.test_hf_dataset = GraphMoleculeDataset(
            csv_path=self.high_fidelity_test_path,
            max_atom_num=self.max_atom_num,
            smiles_column_name=self.smiles_column,
            label_column_name=self.high_fidelity_target_label,
            id_column=self.id_column,
            scaler=self.hf_scaler,
        )

        if self.just_hf:
            self.num_atom_features = self.train_hf_dataset.num_atom_features
            self.num_bond_features = self.train_hf_dataset.num_bond_features

    def setup(self, stage: str=None):
        # Called on every GPU
        # Assumes prepare_data has been called
        pass


    def train_dataloader(self):
        if not self.just_hf:
            return CombinedLoader([
                GeometricDataLoader(self.lf_dataset, batch_size=self.lf_batch_size, shuffle=True, num_workers=self.num_cores[0]),
                GeometricDataLoader(self.train_hf_dataset, batch_size=self.hf_batch_size, shuffle=True, num_workers=self.num_cores[0])
            ], 'sequential')
        else:
            return GeometricDataLoader(self.train_hf_dataset, batch_size=self.hf_batch_size, shuffle=True, num_workers=self.num_cores[0])


    def val_dataloader(self):
        return GeometricDataLoader(self.val_hf_dataset, batch_size=self.hf_batch_size, shuffle=False, num_workers=self.num_cores[1])


    def test_dataloader(self):
        return GeometricDataLoader(self.test_hf_dataset, batch_size=self.hf_batch_size, shuffle=False, num_workers=self.num_cores[2])