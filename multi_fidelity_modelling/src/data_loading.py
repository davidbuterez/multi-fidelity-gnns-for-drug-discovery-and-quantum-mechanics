import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData, DataLoader as GeometricDataLoader
# from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, List, Tuple, Optional

from multi_fidelity_modelling.src.chemprop_featurisation import atom_features, bond_features, get_atom_constants


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return (Chem.MolToSmiles(mol))


class GraphMoleculeDataset(TorchDataset):
    def __init__(self, csv_path: str, max_atom_num: int, smiles_column_name: str, label_column_name: Union[str, List[str]], auxiliary_data_column_name: str=None, lbl_or_emb: str='lbl', scaler: Optional[StandardScaler]=None, id_column: str=None):
        super().__init__()
        assert lbl_or_emb in [None, 'lbl', 'emb']
        self.df = pd.read_csv(csv_path)
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.atom_constants = get_atom_constants(max_atom_num)
        self.num_atom_features = sum(len(choices) for choices in self.atom_constants.values()) + 2
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
        # selected = selected.rename(columns={selected.columns[0]: 'CID'})
        if self.id_column:
            # print(selected)
            ids = selected[self.id_column].values
        smiles = selected[self.smiles_column_name].values
        try:
            if self.scaler:
                if selected[self.label_column_name].values.ndim == 1:
                    labels = torch.Tensor(self.scaler.transform(np.expand_dims(selected[self.label_column_name].values, axis=1)))
                else:
                    labels = torch.Tensor(self.scaler.transform(selected[self.label_column_name].values))
            else:
                labels = torch.Tensor(selected[self.label_column_name].values)
        except:
            labels = torch.Tensor([0] * len(idx))
        smiles = [remove_smiles_stereo(s) for s in smiles]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        if self.auxiliary_data_column_name:
            column_values = selected[self.auxiliary_data_column_name].values
            if self.lbl_or_emb and self.lbl_or_emb == 'lbl':
                aux_data = torch.Tensor(column_values)
            elif self.lbl_or_emb == 'emb':
                # Need to parse np array from string stored in DataFrame
                aux_data = torch.Tensor(np.stack(np.fromstring(selected[self.auxiliary_data_column_name].values[0][1:-1], sep=', ')))
#                 aux_data = torch.Tensor(np.stack(np.fromstring(selected[self.auxiliary_data_column_name].values[0][1:-1], sep=' ')))
        else:
            aux_data = None

        atom_feat = [torch.Tensor([atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]) for mol in rdkit_mols]

        edge_index = []
        bond_feat = []

        for mol in rdkit_mols:
            ei = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
            bf = torch.Tensor([bond_features(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])])

            edge_index.append(ei)
            bond_feat.append(bf)

        geometric_data_points = [GeometricData(x=atom_feat[i], edge_attr=bond_feat[i], edge_index=edge_index[i], y=labels[i], aux_data=aux_data, iden=ids[i]) for i in range(len(atom_feat))]
        for i, data_point in enumerate(geometric_data_points):
#             data_point.rdkit_mol = rdkit_mols[i]
            data_point.smiles = smiles[i]
            data_point.aux_data = np.array([]) if aux_data is None else aux_data

        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points


class GeometricDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, seed: int, max_atom_num: int=80, split: Tuple[int, int]=(0.9, 0.05),train_path: str=None, separate_valid_path: str=None, \
                 separate_test_path :str=None, split_train: bool=False, num_cores: Tuple[int, int, int]=(2, 0, 2), smiles_column_name: str='SMILES', label_column_name: str='True Label', \
                 auxiliary_data_column_name: str=None, lbl_or_emb: str='lbl', use_standard_scaler=False, id_column: str=None):
        super().__init__()
        assert lbl_or_emb in [None, 'lbl', 'emb']
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
        self.split_train = split_train
        self.auxiliary_data_column_name = auxiliary_data_column_name
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
            print('In data_loading.prepare_data()')
            self.dataset = GraphMoleculeDataset(csv_path=self.train_path, \
                                                max_atom_num=self.max_atom_num, \
                                                smiles_column_name=self.smiles_column_name, \
                                                label_column_name=self.label_column_name, \
                                                auxiliary_data_column_name=self.auxiliary_data_column_name, \
                                                lbl_or_emb=self.lbl_or_emb, \
                                                scaler=self.scaler, id_column=self.id_column)

            print('Assigned train dataset to self.dataset')

            self.num_atom_features = self.dataset.num_atom_features
            self.num_bond_features = self.dataset.num_bond_features

        if self.separate_valid_path:
            print('In data_loading.prepare_data()')
            print('Separate validation path provided.')
            self.val = GraphMoleculeDataset(csv_path=self.separate_valid_path, \
                                            max_atom_num=self.max_atom_num, smiles_column_name=self.smiles_column_name,\
                                            label_column_name=self.label_column_name, \
                                            auxiliary_data_column_name=self.auxiliary_data_column_name,\
                                            lbl_or_emb=self.lbl_or_emb, scaler=self.scaler, id_column=self.id_column)

            print('Assigned val dataset to self.val')
        if self.separate_test_path:
            print('In data_loading.prepare_data()')
            print('Separate test path provided.')
            self.test = GraphMoleculeDataset(csv_path=self.separate_test_path, \
                                             max_atom_num=self.max_atom_num, \
                                             smiles_column_name=self.smiles_column_name, \
                                             label_column_name=self.label_column_name, \
                                             auxiliary_data_column_name=self.auxiliary_data_column_name, \
                                             lbl_or_emb=self.lbl_or_emb, scaler=self.scaler,\
                                             id_column=self.id_column)

            print('Assigned test dataset to self.test')


    def setup(self, stage: str=None):
        # Called on every GPU
        # Assumes prepare_data has been called
        if (not self.separate_valid_path) and (not self.separate_test_path) and self.split_train:
            print('In setup(), entered if')
            len_train, len_val = int(self.split[0] * len(self.dataset)), int(self.split[1] * len(self.dataset))
            len_test = len(self.dataset) - len_train - len_val
            assert len_train + len_val + len_test == len(self.dataset)

            print('In setup(), len_train = %d, len_val = %d, len_test = %d' % (len_train, len_val, len_test))

            self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [len_train, len_val, len_test], \
                                                                            generator=torch.Generator().manual_seed(self.seed))
            print('In setup(), assigned self.train, self.val, self.test with PyTorch random_split')
        elif self.train_path:
            self.train = self.dataset
            print('In setup(), assigned self.train = self.dataset')


#     drop_last=True

    def train_dataloader(self, shuffle=True):
        if self.train:
            print('Called train_dataloader() and self.train is not none')
            return GeometricDataLoader(self.train, self.batch_size, shuffle=shuffle,\
                                        num_workers=0
                                    #    num_workers=0 if not self.num_cores else self.num_cores[0],
                                       )
        print('Called train_dataloader() and self.train is NONE')
        return None


    def val_dataloader(self):
        if self.val:
            print('Called val_dataloader() and self.val is not none')
            return GeometricDataLoader(self.val, self.batch_size, shuffle=False,\
                                    #    num_workers=0 if not self.num_cores else self.num_cores[1],
                                        num_workers=0
                                       )
        print('Called val_dataloader() and self.val is NONE')
        return None


    def test_dataloader(self):
        if self.test:
            print('Called test_dataloader() and self.test is not none')
            return GeometricDataLoader(self.test, self.batch_size, shuffle=False,\
                                    #    num_workers=0 if not self.num_cores else self.num_cores[2],
                                       num_workers=0)
        print('Called test_dataloader() and self.test is NONE')
        return None

#     def train_dataloader(self, shuffle=True):
#         if self.train:
#             print('Called train_dataloader() and self.train is not none')
#             return GeometricDataLoader(self.train, self.batch_size, shuffle=shuffle, drop_last=True,\
#                                        num_workers=0 if not self.num_cores else self.num_cores[0], pin_memory=True)
#         print('Called train_dataloader() and self.train is NONE')
#         return None


#     def val_dataloader(self):
#         if self.val:
#             print('Called val_dataloader() and self.val is not none')
#             return GeometricDataLoader(self.val, self.batch_size, shuffle=False, drop_last=True,\
#                                        num_workers=0 if not self.num_cores else self.num_cores[1], pin_memory=True)
#         print('Called val_dataloader() and self.val is NONE')
#         return None


#     def test_dataloader(self):
#         if self.test:
#             print('Called test_dataloader() and self.test is not none')
#             return GeometricDataLoader(self.test, self.batch_size, shuffle=False, drop_last=True,\
#                                        num_workers=0 if not self.num_cores else self.num_cores[2], pin_memory=True)
#         print('Called test_dataloader() and self.test is NONE')
#         return None