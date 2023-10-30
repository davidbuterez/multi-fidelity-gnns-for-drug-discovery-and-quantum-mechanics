import torch
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from typing import Optional

import sys
sys.path.append('..')

from .fidelity_message_passing import FidelityGNN
from multifidelity_gnn.src.reporting import get_metrics_pt, get_cls_metrics

torch.set_num_threads(1)


class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        edge_dim: int,
        node_latent_dim: int,
        edge_latent_dim: int,
        fidelity_embedding_dim: int,
        just_hf: bool = False,
        lf_batch_size: int = 512,
        hf_batch_size: int = 32,
        use_set2set: bool = True,
        lr: float = 0.001,
        linear_output_size: int = 1,
        lf_scaler=None,
        hf_scaler=None,
        monitor_loss: str = "val_loss",
        num_layers: Optional[int] = None,
        use_batch_norm: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        assert task_type in ["classification", "regression"]

        if use_batch_norm:
            print("Using batch normalisation for all layers.")
        else:
            print("NOT using batch normalisation.")

        self.num_features = num_features
        self.edge_dim = edge_dim
        self.node_latent_dim = node_latent_dim
        self.edge_latent_dim = edge_latent_dim
        self.task_type = task_type
        self.lr = lr
        self.lf_batch_size = lf_batch_size
        self.hf_batch_size = hf_batch_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.lf_scaler = lf_scaler
        self.hf_scaler = hf_scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss = monitor_loss
        self.name = name
        self.fidelity_embedding_dim = fidelity_embedding_dim
        self.use_set2set = use_set2set
        self.just_hf = just_hf

        # Store model outputs per epoch; used to compute the reporting metrics
        self.train_lf_output = defaultdict(list)
        self.train_hf_output = defaultdict(list)
        self.train_lf_metrics = {}
        self.train_hf_metrics = {}

        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        self.val_metrics = {}
        self.test_metrics = {}

        # Keep track of how many times we called test
        self.num_called_test = 1

        self.gnn_model = FidelityGNN(
            node_in_channels=self.num_features,
            edge_in_channels=self.edge_dim,
            fidelity_embedding_dim=self.fidelity_embedding_dim,
            edge_out_channels=self.edge_latent_dim,
            node_out_channels=self.node_latent_dim,
            set2set_processing_steps=2,
            use_bn=self.use_batch_norm,
            use_set2set=self.use_set2set,
        )

        if self.use_set2set:
            graph_embeddings_dim = (
                2 * self.edge_latent_dim
                + 2 * self.node_latent_dim
                + self.fidelity_embedding_dim
            )
        else:
            graph_embeddings_dim = (
                3 * self.node_latent_dim
                + 3 * self.edge_latent_dim
                + self.fidelity_embedding_dim
            )

        self.linear_output1 = nn.Linear(graph_embeddings_dim, 256)

        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm1d(256)

        self.linear_output2 = nn.Linear(256, self.linear_output_size)

    def forward(self, x, edge_index, edge_attr, fidelity_indicator, batch_mapping):
        x, edge_attr, fidelity_emb = self.gnn_model(
            x, edge_index, edge_attr, fidelity_indicator, batch_mapping
        )

        graph_embeddings = torch.cat((x, edge_attr, fidelity_emb), dim=1)

        if self.use_batch_norm:
            predictions = self.bn3(self.linear_output1(graph_embeddings)).relu()
        else:
            predictions = self.linear_output1(graph_embeddings).relu()

        predictions = torch.flatten(self.linear_output2(predictions))

        return graph_embeddings, predictions

    # Reduce learning rate when a metric has stopped improving
    # The ReduceLROnPlateau scheduler requires a monitor
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.75, patience=15
            ),
            "monitor": self.monitor_loss,
        }

    def _batch_loss(
        self, x, edge_index, edge_attr, fidelity_indicator, y, batch_mapping
    ):
        # Forward pass
        graph_embeddings, predictions = self.forward(
            x, edge_index, edge_attr, fidelity_indicator, batch_mapping
        )

        if self.task_type == "classification":
            predictions = predictions.reshape(-1, self.linear_output_size)
            task_loss = F.binary_cross_entropy_with_logits(
                predictions.squeeze(), y.squeeze().float()
            )

        else:
            task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))

        return task_loss, graph_embeddings, predictions

    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "valid", "test"]

        if not self.just_hf:
            # If training, batch will come from CombinedLoader and we must extract the actual batch
            if step_type == "train":
                batch, _, _ = batch

        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        fidelity_indicator, y, batch_mapping = batch.fidelity, batch.y, batch.batch

        loss, graph_embeddings, predictions = self._batch_loss(
            x, edge_index, edge_attr, fidelity_indicator, y, batch_mapping
        )

        output = (torch.flatten(predictions), torch.flatten(y))

        if step_type == "train":
            if 0 in fidelity_indicator:
                self.train_lf_output[self.current_epoch].append(output)
            elif 1 in fidelity_indicator:
                self.train_hf_output[self.current_epoch].append(output)

        elif step_type == "valid":
            self.val_output[self.current_epoch].append(output)

        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)
            # self.test_graph_embeddings[self.num_called_test].append(graph_embeddings)

        return loss, fidelity_indicator

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_loss, fidelity_indicator = self._step(batch, "train")

        if 0 in fidelity_indicator:
            self.log(
                "sd_train_loss",
                train_loss,
                batch_size=self.lf_batch_size,
                prog_bar=True,
            )
        elif 1 in fidelity_indicator:
            self.log("dr_train_loss", train_loss, batch_size=self.hf_batch_size)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        val_loss, _ = self._step(batch, "valid")

        self.log("val_loss", val_loss, batch_size=self.hf_batch_size)

        return val_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_loss, _ = self._step(batch, "test")

        self.log("test_loss", test_loss, batch_size=self.hf_batch_size)

        return test_loss

    def _epoch_end_report(self, epoch_outputs, epoch_type, is_lf):
        y_pred = (torch.flatten(torch.cat([item[0] for item in epoch_outputs], dim=0)).detach().cpu().numpy())
        y_true = (torch.flatten(torch.cat([item[1] for item in epoch_outputs], dim=0)).detach().cpu().numpy())

        if is_lf:
            fidelity_name = "SD"
            batch_size = self.lf_batch_size
        else:
            fidelity_name = "DR"
            batch_size = self.hf_batch_size

        if self.lf_scaler is not None or self.hf_scaler is not None:
            if is_lf:
                y_pred = self.lf_scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.lf_scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.hf_scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.hf_scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))

        if self.task_type == "classification":
            y_pred = np.where(y_pred >= 0.5, 1, 0).astype(int).flatten()
            y_true = y_true.astype(int).flatten()

            metrics = get_cls_metrics(y_true, y_pred)

            self.log(f"{fidelity_name} {epoch_type} AUROC", metrics[1], batch_size=batch_size)
            self.log(f"{fidelity_name} {epoch_type} MCC", metrics[-1], batch_size=batch_size)
        else:
            y_pred = torch.from_numpy(y_pred).flatten()
            y_true = torch.from_numpy(y_true).flatten()

            metrics = get_metrics_pt(y_true, y_pred)
            for metric_name, metric_value in metrics.items():
                self.log(f"{fidelity_name} {epoch_type} {metric_name}", metric_value, batch_size=batch_size)

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

        return metrics, y_pred, y_true

    def on_train_epoch_end(self):
        if not self.just_hf:
            train_lf_metrics, y_pred, y_true = self._epoch_end_report(
                self.train_lf_output[self.current_epoch], epoch_type="Train", is_lf=not self.just_hf
            )

            self.train_lf_metrics[self.current_epoch] = train_lf_metrics

            del self.train_lf_output[self.current_epoch]

        train_hf_metrics, y_pred, y_true = self._epoch_end_report(
            self.train_hf_output[self.current_epoch], epoch_type="Train", is_lf=not self.just_hf
        )

        self.train_hf_metrics[self.current_epoch] = train_hf_metrics

        del y_pred
        del y_true
        del self.train_hf_output[self.current_epoch]

    def on_validation_epoch_end(self):
        val_outputs_per_epoch = self.val_output[self.current_epoch]
        self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            val_outputs_per_epoch, epoch_type="Validation", is_lf=not self.just_hf
        )

        del y_pred
        del y_true
        del self.val_output[self.current_epoch]

    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch,
            epoch_type="Test",
            is_lf=not self.just_hf
        )

        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true
        self.num_called_test += 1
