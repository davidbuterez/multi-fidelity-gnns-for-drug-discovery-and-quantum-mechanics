import torch
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU
import torch_geometric
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    VGAE,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GINConv,
    GINEConv,
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import to_dense_batch, degree
from tqdm.auto import tqdm
from typing import Optional

from .set_transformer_models import SetTransformer
from .reporting import get_metrics_pt, get_cls_metrics

torch.set_num_threads(1)


def get_degrees(train_dataset_as_list):
    deg = torch.zeros(10, dtype=torch.long)
    print("Computing degrees for PNA...")
    for data in tqdm(train_dataset_as_list):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


# ############# Variational encoders ##############


# Taken and adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
class VariationalGCNEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        name: str = None,
    ):
        super(VariationalGCNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (
                        GCNConv(in_channels, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )
            else:
                modules.append(
                    (
                        GCNConv(intermediate_dim, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

            if self.use_batch_norm:
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True))

        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        self.conv_mu = GCNConv(intermediate_dim, out_channels, cached=False)
        self.conv_logstd = GCNConv(intermediate_dim, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.convs(x, edge_index)

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VariationalGINEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        name: str = None,
    ):
        super(VariationalGINEncoder, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )
            else:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

            if self.use_batch_norm:
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        nn_mu = nn.Sequential(
            Linear(intermediate_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )
        if self.edge_dim:
            self.conv_mu = GINEConv(nn_mu, edge_dim=self.edge_dim)
        else:
            self.conv_mu = GINConv(nn_mu)

        nn_sigma = nn.Sequential(
            Linear(intermediate_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )
        if self.edge_dim:
            self.conv_logstd = GINEConv(nn_sigma, edge_dim=self.edge_dim)
        else:
            self.conv_logstd = GINConv(nn_sigma)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            x = self.convs(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.convs(x, edge_index)

        if self.edge_dim:
            mu = self.conv_mu(x, edge_index, edge_attr=edge_attr)
            sigma = self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            mu = self.conv_mu(x, edge_index)
            sigma = self.conv_logstd(x, edge_index)
        return mu, sigma


class VariationalPNAEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        train_dataset,
        edge_dim: int = None,
        name: str = None,
    ):
        super(VariationalPNAEncoder, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = get_degrees(train_dataset)

        pna_num_towers = 5

        pna_common_args = dict(
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=None,
            towers=pna_num_towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        if self.edge_dim:
            pna_common_args = pna_common_args | dict(edge_dim=edge_dim)

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )
            else:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

            if self.use_batch_norm:
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        self.conv_mu = PNAConv(
            in_channels=intermediate_dim, out_channels=out_channels, **pna_common_args
        )
        self.conv_logstd = PNAConv(
            in_channels=intermediate_dim, out_channels=out_channels, **pna_common_args
        )

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            x = self.convs(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.convs(x, edge_index)

        if self.edge_dim:
            mu = self.conv_mu(x, edge_index, edge_attr=edge_attr)
            sigma = self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            mu = self.conv_mu(x, edge_index)
            sigma = self.conv_logstd(x, edge_index)
        return mu, sigma


# ############# Variational encoders ##############


# ############# Non-variational GNN ##############


class GCN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        name: str = None,
    ):
        super(GCN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (
                        GCNConv(in_channels, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                modules.append(
                    (
                        GCNConv(intermediate_dim, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                modules.append(
                    (
                        GCNConv(intermediate_dim, out_channels, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index):
        return self.convs(x, edge_index)


class GIN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        name: str = None,
    ):
        super(GIN, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, out_channels),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, out_channels),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            return self.convs(x, edge_index, edge_attr=edge_attr)
        return self.convs(x, edge_index)


class PNA(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        train_dataset,
        edge_dim: int = None,
        name: str = None,
    ):
        super(PNA, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = get_degrees(train_dataset)

        pna_num_towers = 5

        pna_common_args = dict(
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=None,
            towers=pna_num_towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        if self.edge_dim:
            pna_common_args = pna_common_args | dict(edge_dim=edge_dim)

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=out_channels,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=out_channels,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            return self.convs(x, edge_index, edge_attr=edge_attr)
        return self.convs(x, edge_index)


# ############# Non-variational GNN ##############


class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        gnn_intermediate_dim: int,
        node_latent_dim: int,
        graph_latent_dim: Optional[int] = None,
        train_dataset=None,
        batch_size: int = 32,
        lr: float = 0.001,
        linear_output_size: int = 1,
        auxiliary_dim: int = 0,
        output_intermediate_dim: int = 768,
        scaler=None,
        readout: str = "linear",
        max_num_atoms_in_mol: int = 55,
        monitor_loss: str = "val_total_loss",
        num_layers: Optional[int] = None,
        use_batch_norm: bool = False,
        name: Optional[str] = None,
        set_transformer_hidden_dim: Optional[int] = None,
        set_transformer_num_heads: Optional[int] = None,
        set_transformer_num_sabs: Optional[int] = None,
        set_transformer_dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        use_vgae: bool = True,
        linear_interim_dim: int = 64,
        linear_dropout_p: float = 0.2,
        conv_type: str = "GCN",
        only_train: bool = False,
    ):
        super().__init__()
        assert task_type in ["classification", "regression"]
        assert conv_type in ["GCN", "GIN", "PNA"]
        assert readout in [
            "linear",
            "global_mean_pool",
            "global_add_pool",
            "global_max_pool",
            "set_transformer",
        ]

        print(
            "%s task with %d %s layers and %s readout."
            % (task_type.capitalize(), num_layers, conv_type, readout)
        )

        if use_batch_norm:
            print("Using batch normalisation for all layers.")
        else:
            print("NOT using batch normalisation.")

        self.use_vgae = use_vgae
        self.edge_dim = edge_dim
        self.only_train = only_train
        self.graph_latent_dim = graph_latent_dim if self.only_train else node_latent_dim
        self.task_type = task_type
        self.global_pool_fn = (
            global_mean_pool
            if readout == "global_mean_pool"
            else (
                global_add_pool
                if readout == "global_add_pool"
                else (global_max_pool if readout == "global_max_pool" else None)
            )
        )

        if self.use_vgae:
            print("Using the VGAE framework.")
        else:
            print("Using a non-variational GNN model.")

        if self.global_pool_fn:
            print("Using %s, graph_latent_dim not used." % (readout))
            print("Using %d latent node features." % node_latent_dim)
        else:
            print(
                "Using %d latent node features and %d latent graph features."
                % (node_latent_dim, self.graph_latent_dim)
            )

        self.auxiliary_dim = auxiliary_dim if auxiliary_dim else 0
        if self.auxiliary_dim > 0:
            print(
                "Using auxiliary data with dimension %d, total with GNN/VGAE embeddings: %d."
                % (self.auxiliary_dim, self.graph_latent_dim + self.auxiliary_dim)
            )

        if self.edge_dim:
            print("Using edge (bond) features of dimension %d." % (self.edge_dim))
        else:
            print("NOT using edge (bond) features.")

        self.readout = readout
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.conv_type = conv_type
        self.node_latent_dim = node_latent_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.output_intermediate_dim = output_intermediate_dim
        self.linear_interim_dim = linear_interim_dim
        self.linear_dropout_p = linear_dropout_p
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss = monitor_loss
        self.name = name

        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs
        self.set_transformer_dropout = set_transformer_dropout

        # Store model outputs per epoch; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.train_metrics = {}

        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        self.val_metrics = {}
        self.test_metrics = {}

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Holds final graphs embeddings
        self.train_graph_embeddings = defaultdict(list)
        self.test_graph_embeddings = defaultdict(list)

        gnn_args = dict(
            in_channels=num_features,
            out_channels=node_latent_dim,
            num_layers=self.num_layers,
            intermediate_dim=self.gnn_intermediate_dim,
            use_batch_norm=self.use_batch_norm,
            name=self.name,
        )

        if self.edge_dim:
            gnn_args = gnn_args | dict(edge_dim=self.edge_dim)
        if self.conv_type == "PNA":
            gnn_args = gnn_args | dict(train_dataset=train_dataset)

        if self.conv_type == "GCN":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalGCNEncoder(**gnn_args))
            else:
                self.gnn_model = GCN(**gnn_args)
        elif self.conv_type == "GIN":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalGINEncoder(**gnn_args))
            else:
                self.gnn_model = GIN(**gnn_args)
        elif self.conv_type == "PNA":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalPNAEncoder(**gnn_args))
            else:
                self.gnn_model = PNA(**gnn_args)

        if self.readout == "linear":
            self.linear_readout1 = nn.Linear(
                self.max_num_atoms_in_mol * node_latent_dim, self.linear_interim_dim
            )
            self.linear_readout2 = nn.Linear(
                self.linear_interim_dim, self.graph_latent_dim
            )
            if self.use_batch_norm:
                self.bn1 = nn.BatchNorm1d(self.linear_interim_dim)
                self.bn2 = nn.BatchNorm1d(self.graph_latent_dim)

            if self.linear_dropout_p > 0:
                self.linear_dropout = nn.Dropout1d(p=self.linear_dropout_p)

        elif self.readout == "set_transformer":
            self.st = SetTransformer(
                dim_input=node_latent_dim,
                num_outputs=32,
                dim_output=self.graph_latent_dim,
                num_inds=None,
                ln=True,
                dim_hidden=self.set_transformer_hidden_dim,
                num_heads=self.set_transformer_num_heads,
                num_sabs=self.set_transformer_num_sabs,
                dropout=self.set_transformer_dropout,
            )

        if self.only_train:
            self.linear_output1 = nn.Linear(
                self.graph_latent_dim + self.auxiliary_dim, 256
            )
        else:
            self.linear_output1 = nn.Linear(
                self.node_latent_dim * 3 + self.auxiliary_dim, 256
            )

        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm1d(256)

        self.linear_output2 = nn.Linear(256, self.linear_output_size)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        aux_data: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        # 1. Obtain node embeddings
        if self.use_vgae:
            if self.edge_dim:
                z = self.gnn_model.encode(x, edge_index, edge_attr=edge_attr)
            else:
                z = self.gnn_model.encode(x, edge_index)
        else:
            if self.edge_dim:
                z = self.gnn_model.forward(x, edge_index, edge_attr=edge_attr)
            else:
                z = self.gnn_model.forward(x, edge_index)

        # 2. Readout layer
        # Due to batching in PyTorch Geometric, the node embeddings must be regrouped into their original graphs
        # Details: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

        graph_embeddings_to_return = None

        # Simple global pooling of node features
        if self.only_train and self.global_pool_fn:
            graph_embeddings = self.global_pool_fn(z, batch)
            graph_embeddings_to_return = graph_embeddings

        if self.only_train and not self.global_pool_fn and self.readout == "linear":
            graph_embeddings, _ = to_dense_batch(
                z, batch, fill_value=0, max_num_nodes=self.max_num_atoms_in_mol
            )

            # Reshape to (current_batch_shape, flattened_node_features)
            graph_embeddings = graph_embeddings.reshape(
                graph_embeddings.shape[0],
                self.max_num_atoms_in_mol * self.node_latent_dim,
            )

            # Apply the dense layers to get a graph-level representation
            if self.use_batch_norm:
                graph_embeddings = self.bn1(
                    self.linear_readout1(graph_embeddings)
                ).relu()
                graph_embeddings_without_relu = self.bn2(
                    self.linear_readout2(graph_embeddings)
                )
            else:
                graph_embeddings = self.linear_readout1(graph_embeddings).relu()
                graph_embeddings_without_relu = self.linear_readout2(graph_embeddings)

            graph_embeddings_to_return = graph_embeddings_without_relu
            graph_embeddings = graph_embeddings_without_relu.relu()

            if self.linear_dropout_p > 0:
                graph_embeddings = self.linear_dropout(graph_embeddings)

        elif (
            self.only_train
            and not self.global_pool_fn
            and self.readout == "set_transformer"
        ):
            graph_embeddings, _ = to_dense_batch(
                z, batch, fill_value=0, max_num_nodes=self.max_num_atoms_in_mol
            )
            graph_embeddings = self.st(graph_embeddings)
            graph_embeddings = graph_embeddings.mean(dim=1)
            graph_embeddings_to_return = graph_embeddings

        if not self.only_train:
            graph_embeddings_sum = global_add_pool(z, batch)
            graph_embeddings_mean = global_mean_pool(z, batch)
            graph_embeddings_max = global_max_pool(z, batch)
            graph_embeddings = torch.cat(
                (graph_embeddings_sum, graph_embeddings_mean, graph_embeddings_max),
                dim=-1,
            )
            graph_embeddings_to_return = graph_embeddings

        # 2.1. Concatenate auxiliary data (labels or embeddings) as additional columns, when available
        if self.auxiliary_dim > 0:
            assert len(aux_data.shape) == 1
            if self.auxiliary_dim == 1:
                # Here we assume the auxiliary data are just additional labels
                # (a column with single values in the DataFrame), with resulting shape (batch_size, 1)
                aux_data = aux_data.unsqueeze(dim=1)
            elif self.auxiliary_dim > 1:
                # Here we assume the individual auxiliary data points are numpy arrays,
                # so a batch of aux data would have shape (batch_size, length_of_np_arr)
                aux_data = aux_data.reshape(
                    (graph_embeddings.shape[0], self.auxiliary_dim)
                )

            # Actual concatenation
            graph_embeddings = torch.cat((graph_embeddings, aux_data), dim=1)

        # 3. Apply a final classifier

        if self.use_batch_norm:
            predictions = self.bn3(self.linear_output1(graph_embeddings)).relu()
        else:
            predictions = self.linear_output1(graph_embeddings).relu()

        predictions = torch.flatten(self.linear_output2(predictions))

        return z, graph_embeddings_to_return, predictions

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
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        batch_mapping: Optional[torch.Tensor] = None,
        aux_data: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        # Number of nodes in graph
        num_nodes = x.shape[0]

        # Forward pass
        if not self.edge_dim:
            z, graph_embeddings, predictions = self.forward(
                x, edge_index, batch_mapping, aux_data
            )
        else:
            z, graph_embeddings, predictions = self.forward(
                x, edge_index, batch_mapping, aux_data, edge_attr
            )

        # VGAE loss from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
        if self.use_vgae:
            vgae_loss = self.gnn_model.recon_loss(z, edge_index)
            vgae_loss = vgae_loss + (1 / num_nodes) * self.gnn_model.kl_loss()

        if self.task_type == "classification":
            predictions = predictions.reshape(-1, self.linear_output_size)
            task_loss = F.binary_cross_entropy_with_logits(
                predictions.squeeze(), y.squeeze().float()
            )

        else:
            task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))

        if self.use_vgae:
            total_loss = vgae_loss + task_loss
            return total_loss, vgae_loss, task_loss, z, graph_embeddings, predictions
        else:
            total_loss = task_loss
            return total_loss, 0.0, 0.0, z, graph_embeddings, predictions

    def _step(self, batch: torch.Tensor, step_type: str):
        # assert step_type in ['train', 'valid', 'test']

        x, edge_index, edge_attr, y, batch_mapping = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y,
            batch.batch,
        )
        aux_data = batch.aux_data

        (
            total_loss,
            vgae_loss,
            task_loss,
            _,
            graph_embeddings,
            predictions,
        ) = self._batch_loss(x, edge_index, y, batch_mapping, aux_data, edge_attr)

        output = (torch.flatten(predictions), torch.flatten(y))

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif not self.only_train and step_type == "valid":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)
            self.test_graph_embeddings[self.num_called_test].append(graph_embeddings)

        return total_loss, vgae_loss, task_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # If not using VGAE, vgae_loss is set to 0 and train_total_loss = task_loss
        train_total_loss, vgae_loss, task_loss = self._step(batch, "train")

        if self.use_vgae:
            self.log("train_total_loss", train_total_loss, batch_size=self.batch_size)
            self.log("train_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("train_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("train_total_loss", train_total_loss, batch_size=self.batch_size)

        return train_total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # edge_attr not used so far

        # If not using VGAE, vgae_loss is set to 0 and val_total_loss = task_loss
        val_total_loss, vgae_loss, task_loss = self._step(batch, "valid")

        if self.use_vgae:
            self.log("val_total_loss", val_total_loss, batch_size=self.batch_size)
            self.log("val_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("val_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("val_total_loss", val_total_loss, batch_size=self.batch_size)

        return val_total_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        # edge_attr not used so far

        # If not using VGAE, vgae_loss is set to 0 and test_total_loss = task_loss
        test_total_loss, vgae_loss, task_loss = self._step(batch, "test")

        if self.use_vgae:
            self.log("test_total_loss", test_total_loss, batch_size=self.batch_size)
            self.log("test_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("test_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("test_total_loss", test_total_loss, batch_size=self.batch_size)

        return test_total_loss

    def _epoch_end_report(self, epoch_outputs, epoch_type):
        y_pred = (
            torch.flatten(torch.cat([item[0] for item in epoch_outputs], dim=0))
            .detach()
            .cpu()
            .numpy()
        )
        y_true = (
            torch.flatten(torch.cat([item[1] for item in epoch_outputs], dim=0))
            .detach()
            .cpu()
            .numpy()
        )

        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(
                    y_pred.reshape(-1, self.linear_output_size)
                )
                y_true = self.scaler.inverse_transform(
                    y_true.reshape(-1, self.linear_output_size)
                )
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        if self.task_type == "classification":
            y_pred = np.where(y_pred >= 0.5, 1, 0).astype(int).flatten()
            y_true = y_true.astype(int).flatten()

            metrics = get_cls_metrics(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[-1], batch_size=self.batch_size)
        else:
            y_pred = torch.from_numpy(y_pred).flatten()
            y_true = torch.from_numpy(y_true).flatten()

            metrics = get_metrics_pt(y_true, y_pred)
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{epoch_type} {metric_name}",
                    metric_value,
                    batch_size=self.batch_size,
                )

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

        return metrics, y_pred, y_true

    def on_train_epoch_end(self):
        if self.only_train:
            (
                self.train_metrics[self.current_epoch],
                y_pred,
                y_true,
            ) = self._epoch_end_report(
                self.train_output[self.current_epoch], epoch_type="Train"
            )

            del y_pred
            del y_true
            del self.train_output[self.current_epoch]

    def on_validation_epoch_end(self):
        if not self.only_train:
            val_outputs_per_epoch = self.val_output[self.current_epoch]
            (
                self.val_metrics[self.current_epoch],
                y_pred,
                y_true,
            ) = self._epoch_end_report(val_outputs_per_epoch, epoch_type="Validation")

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        (
            self.test_metrics[self.num_called_test],
            y_pred,
            y_true,
        ) = self._epoch_end_report(test_outputs_per_epoch, epoch_type="Test")

        self.test_graph_embeddings[self.num_called_test] = (
            torch.cat(self.test_graph_embeddings[self.num_called_test])
            .detach()
            .cpu()
            .numpy()
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true
        self.num_called_test += 1
