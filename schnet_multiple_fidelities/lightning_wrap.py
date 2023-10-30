import torch
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from torch.nn import functional as F
from torch import nn

# Imports from this project
from multifidelity_gnn.src.reporting import get_metrics_qm7
from .schnet_high_fidelity import SchNet


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Estimator(pl.LightningModule):
    def __init__(
            self,
            batch_size: int=32,
            lr: float=0.001,
            readout: str='linear',
            max_num_atoms_in_mol: int=55,
            scaler=None,
            monitor_loss: str='val_total_loss',
            name: str=None,
            aux_scaler=None,
            use_layer_norm=False,
            schnet_hidden_channels=128,
            schnet_num_filters=128,
            schnet_num_interactions=6,
            atomref=None,
            is_dipole=None,
            set_transformer_hidden_dim=None,
            set_transformer_num_heads=None,
            set_transformer_num_sabs=None,
            lbl_or_emb=None,
            include=None,
            emb_type=None,
            property=None,
            aux_dim=0
        ):

        super().__init__()
        self.readout = readout
        self.lr = lr
        self.batch_size = batch_size
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.scaler = scaler
        self.aux_scaler = aux_scaler
        self.linear_output_size = 1
        self.monitor_loss = monitor_loss
        self.metric_fn = get_metrics_qm7
        self.name = name
        self.use_layer_norm = use_layer_norm

        self.schnet_hidden_channels = schnet_hidden_channels
        self.schnet_num_filters = schnet_num_filters
        self.schnet_num_interactions = schnet_num_interactions

        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs

        self.atomref = atomref
        self.is_dipole = is_dipole

        self.lbl_or_emb = lbl_or_emb
        self.include = include
        self.emb_type = emb_type
        self.property = property
        self.aux_dim = aux_dim

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.test_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        self.test_graph_embeddings = defaultdict(list)
        
        self.net = SchNet(
            hidden_channels=self.schnet_hidden_channels,
            num_filters=self.schnet_num_filters,
            num_interactions=self.schnet_num_interactions,
            num_gaussians=50,
            cutoff=10.0,
            dipole=self.is_dipole,
            atomref=self.atomref,
            readout=self.readout,
            set_transformer_hidden_dim=self.set_transformer_hidden_dim,
            set_transformer_num_heads=self.set_transformer_num_heads,
            set_transformer_num_sabs=self.set_transformer_num_sabs,
            max_num_atoms_in_mol=self.max_num_atoms_in_mol
        )

        if self.include == 'both':
            aux_mlp_dim = self.schnet_hidden_channels + 2 * self.aux_dim
        elif self.include in ['zindo', 'pbe0']:
            aux_mlp_dim = self.schnet_hidden_channels + self.aux_dim
        else:
            aux_mlp_dim = 0

        if aux_mlp_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(in_features=aux_mlp_dim, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(),
            )

        regr_nn_in_dim = 256 if self.aux_dim > 0 else self.schnet_hidden_channels
        self.regr_nn = nn.Sequential(
                nn.Linear(regr_nn_in_dim, self.schnet_hidden_channels // 2),
                ShiftedSoftplus(),
                nn.Linear(self.schnet_hidden_channels // 2, self.schnet_hidden_channels // 2),
                ShiftedSoftplus(),
                nn.Linear(self.schnet_hidden_channels // 2, 1)
            )


    def forward(self, pos, atom_z, batch_mapping, aux_zindo, aux_pbe0):
        graph_embeddings = self.net(pos=pos, z=atom_z, batch=batch_mapping)

        if aux_zindo is not None:
            aux_zindo_ = aux_zindo.view(graph_embeddings.shape[0], self.aux_dim)
            graph_embeddings = torch.cat((graph_embeddings, aux_zindo_), dim=1)

        if aux_pbe0 is not None:
            aux_pbe0_ = aux_pbe0.view(graph_embeddings.shape[0], self.aux_dim)
            graph_embeddings = torch.cat((graph_embeddings, aux_pbe0_), dim=1)

        if self.include in ['both', 'zindo', 'pbe0']:
            graph_embeddings = self.aux_mlp(graph_embeddings)

        predictions = self.regr_nn(graph_embeddings)

        return predictions, graph_embeddings


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': opt,
            'monitor': self.monitor_loss
        }


    def _batch_loss(self, pos, y, atom_z, batch_mapping, aux_zindo, aux_pbe0):
        predictions, graph_embeddings = self.forward(pos, atom_z, batch_mapping, aux_zindo, aux_pbe0)

        loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y).float())

        return loss, predictions, graph_embeddings


    def _step(self, batch, step_type: str):
        assert step_type in ['train', 'valid', 'test']

        pos, y, atom_z, batch_mapping = batch.pos, batch.y, batch.z, batch.batch
        homo_zindo_sum_emb = batch.homo_zindo_sum_emb
        homo_zindo_st_emb = batch.homo_zindo_st_emb
        homo_pbe0_sum_emb = batch.homo_pbe0_sum_emb
        homo_pbe0_st_emb = batch.homo_pbe0_st_emb

        lumo_zindo_sum_emb = batch.lumo_zindo_sum_emb
        lumo_zindo_st_emb = batch.lumo_zindo_st_emb
        lumo_pbe0_sum_emb = batch.lumo_pbe0_sum_emb
        lumo_pbe0_st_emb = batch.lumo_pbe0_st_emb

        homo_zindo_label = batch.homo_zindo_label
        homo_pbe0_label = batch.homo_pbe0_label
        homo_gw_label = batch.homo_gw_label

        lumo_zindo_label = batch.lumo_zindo_label
        lumo_pbe0_label = batch.lumo_pbe0_label
        lumo_gw_label = batch.lumo_gw_label

        aux_zindo = None
        aux_pbe0 = None
        
        if self.lbl_or_emb == 'lbl':
            if self.property == 'homo':
                if self.include == 'zindo':
                    aux_zindo = homo_zindo_label
                elif self.include == 'pbe0':
                    aux_pbe0 = homo_pbe0_label
                elif self.include == 'both':
                    aux_zindo = homo_zindo_label
                    aux_pbe0 = homo_pbe0_label
            elif self.property == 'lumo':
                if self.include == 'zindo':
                    aux_zindo = lumo_zindo_label
                elif self.include == 'pbe0':
                    aux_pbe0 = lumo_pbe0_label
                elif self.include == 'both':
                    aux_zindo = lumo_zindo_label
                    aux_pbe0 = lumo_pbe0_label
        elif self.lbl_or_emb == 'emb':
            if self.property == 'homo':
                if self.include == 'zindo':
                    if self.emb_type == 'sum':
                        aux_zindo = homo_zindo_sum_emb
                    elif self.emb_type == 'st':
                        aux_zindo = homo_zindo_st_emb
                elif self.include == 'pbe0':
                    if self.emb_type == 'sum':
                        aux_pbe0 = homo_pbe0_sum_emb
                    elif self.emb_type == 'st':
                        aux_pbe0 = homo_pbe0_st_emb
                elif self.include == 'both':
                    if self.emb_type == 'sum':
                        aux_zindo = homo_zindo_sum_emb
                        aux_pbe0 = homo_pbe0_sum_emb
                    elif self.emb_type == 'st':
                        aux_zindo = homo_zindo_st_emb
                        aux_pbe0 = homo_pbe0_st_emb
            elif self.property == 'lumo':
                if self.include == 'zindo':
                    if self.emb_type == 'sum':
                        aux_zindo = lumo_zindo_sum_emb
                    elif self.emb_type == 'st':
                        aux_zindo = lumo_zindo_st_emb
                elif self.include == 'pbe0':
                    if self.emb_type == 'sum':
                        aux_pbe0 = lumo_pbe0_sum_emb
                    elif self.emb_type == 'st':
                        aux_pbe0 = lumo_pbe0_st_emb
                elif self.include == 'both':
                    if self.emb_type == 'sum':
                        aux_zindo = lumo_zindo_sum_emb
                        aux_pbe0 = lumo_pbe0_sum_emb
                    elif self.emb_type == 'st':
                        aux_zindo = lumo_zindo_st_emb
                        aux_pbe0 = lumo_pbe0_st_emb
            

        total_loss, predictions, graph_embeddings = self._batch_loss(pos, y, atom_z, batch_mapping, aux_zindo, aux_pbe0)

        output = (torch.flatten(predictions), torch.flatten(y))

        if step_type == 'train':
            self.train_output[self.current_epoch].append(output)

        elif step_type == 'valid':
            self.val_output[self.current_epoch].append(output)

        elif step_type == 'test':
            self.test_output[self.num_called_test].append(output)
            self.test_graph_embeddings[self.num_called_test].append(graph_embeddings)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_total_loss = self._step(batch, 'train')

        self.log('train_total_loss', train_total_loss, batch_size=self.batch_size)

        return train_total_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        val_total_loss = self._step(batch, 'valid')
        self.log('val_total_loss', val_total_loss, batch_size=self.batch_size)

        return val_total_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_total_loss = self._step(batch, 'test')

        self.log('test_total_loss', test_total_loss, batch_size=self.batch_size)

        return test_total_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        def flatten_list_of_tensors(lst):
            return np.array([item.item() for sublist in lst for item in sublist])

        y_pred = flatten_list_of_tensors([item[0] for item in epoch_outputs])
        y_true = flatten_list_of_tensors([item[1] for item in epoch_outputs])
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)

        if self.scaler:
            y_pred = self.scaler.inverse_transform(y_pred).squeeze()
            y_true = self.scaler.inverse_transform(y_true).squeeze()

        metrics = self.metric_fn(y_true, y_pred)
        self.log(f'{epoch_type} MAE', metrics[0], batch_size=self.batch_size)
        self.log(f'{epoch_type} RMSE', metrics[1], batch_size=self.batch_size)
        self.log(f'{epoch_type} R2', metrics[-1], batch_size=self.batch_size)

        return metrics, y_pred, y_true


    ### Do not save any training outputs
    def on_train_epoch_end(self):
        train_metrics, y_pred, y_true = self._epoch_end_report(self.train_output[self.current_epoch], epoch_type='Train')

        self.train_metrics[self.current_epoch] = train_metrics

        del self.train_output[self.current_epoch]


    ### Do not save any validation outputs
    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            val_metrics, y_pred, y_true = self._epoch_end_report(self.val_output[self.current_epoch], epoch_type='Validation')

            self.val_metrics[self.current_epoch] = val_metrics

            del self.val_output[self.current_epoch]
        

    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        metrics, y_pred, y_true = self._epoch_end_report(test_outputs_per_epoch, epoch_type='Test')

        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true
        self.test_metrics[self.num_called_test] = metrics
        self.test_graph_embeddings = torch.cat(self.test_graph_embeddings[self.num_called_test], dim=0).detach().cpu().numpy()

        self.num_called_test += 1
