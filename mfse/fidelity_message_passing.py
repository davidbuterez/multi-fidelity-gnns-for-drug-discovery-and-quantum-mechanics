import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.utils import scatter
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin



def get_mlp(input_dim, interim_dim, out_dim, use_bn=False):
    return Seq(
        Lin(input_dim, interim_dim),
        nn.BatchNorm1d(interim_dim) if use_bn else nn.Identity(),
        nn.ReLU(),
        Lin(interim_dim, out_dim),
        nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
    )


class MFMPN(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, node_out_channels, edge_out_channels, fidelity_embedding_dim, use_bn=False):
        super(MFMPN, self).__init__(aggr=None)  # No aggregation scheme set here.
        self.phi_e = get_mlp(
            2 * node_in_channels + edge_in_channels + fidelity_embedding_dim, 128, edge_out_channels, use_bn=use_bn
        )
        self.phi_v = get_mlp(
            edge_out_channels + node_in_channels + fidelity_embedding_dim, 128, node_out_channels, use_bn=use_bn
        )
        self.phi_u = get_mlp(
            edge_out_channels + node_out_channels + fidelity_embedding_dim, 128, fidelity_embedding_dim, use_bn=use_bn
        )
    
    
    def forward(self, x, edge_index, edge_attr, fidelity_emb, batch_mapping=None):
        # Generate edge-to-batch mapping
        edge_batch_mapping = batch_mapping.index_select(dim=0, index=edge_index[0, :])

        # Edge Update:
        # Get node attributes/features corresponding to the source and target nodes for each edge
        src_node_features = x.index_select(dim=0, index=edge_index[0, :])
        target_node_features = x.index_select(dim=0, index=edge_index[1, :])
        
        # Repeat the fidelity embedding for each edge
        fidelity_repeated = self.repeat_for_edges(fidelity_emb, edge_index, batch_mapping)

        # Update edge attributes/features
        updated_edge_attr = self.phi_e(
            torch.cat([src_node_features, target_node_features, edge_attr, fidelity_repeated], dim=-1)
        )

        # Node Update:
        # Compute the average of edge attributes/features for each node
        averaged_edge_features_per_node = scatter(src=updated_edge_attr, index=edge_index[1], dim=0, reduce='mean')

        fidelity_repeated_for_nodes = fidelity_emb.index_select(dim=0, index=batch_mapping)

        # Update node attributes/features
        updated_x = self.phi_v(torch.cat([averaged_edge_features_per_node, x, fidelity_repeated_for_nodes], dim=-1))

        # State (Fidelity) Update
        # Compute global average of edge and node attributes/features for each graph in the batch
        averaged_edge_features = scatter(src=updated_edge_attr, index=edge_batch_mapping, dim=0, reduce='mean')
        averaged_node_features = scatter(src=updated_x, index=batch_mapping, dim=0, reduce='mean')

        # Update the global state/fidelity embedding
        fidelity_emb = self.phi_u(torch.cat([averaged_node_features, averaged_edge_features, fidelity_emb], dim=-1))

        return updated_x, updated_edge_attr, fidelity_emb


    def repeat_for_edges(self, fidelity_emb, edge_index, batch_mapping):
        # Fetch the batch identifier for the source node of each edge.
        edge_batches = batch_mapping.index_select(dim=0, index=edge_index[0, :])
        # Fetch the corresponding fidelity embedding using the batch identifiers.
        return fidelity_emb.index_select(dim=0, index=edge_batches)
    
    

class FidelityGNN(nn.Module):
    def __init__(
            self,
            node_in_channels,
            edge_in_channels,
            fidelity_embedding_dim,
            edge_out_channels,
            node_out_channels,
            set2set_processing_steps=2,
            use_bn=False,
            use_set2set=True
        ):
        super(FidelityGNN, self).__init__()

        self.use_bn = use_bn
        self.use_set2set = use_set2set
        self.fidelity_embedding = nn.Embedding(num_embeddings=2, embedding_dim=fidelity_embedding_dim)

        self.mfmp_0 = MFMPN(node_in_channels, edge_in_channels, node_out_channels, edge_out_channels, fidelity_embedding_dim, use_bn=self.use_bn)
        self.mfmp_1 = MFMPN(node_out_channels, edge_out_channels, node_out_channels, edge_out_channels, fidelity_embedding_dim, use_bn=self.use_bn)
        self.mfmp_2 = MFMPN(node_out_channels, edge_out_channels, node_out_channels, edge_out_channels, fidelity_embedding_dim, use_bn=self.use_bn)

        if self.use_set2set:
            self.node_set2set = Set2Set(node_out_channels, processing_steps=set2set_processing_steps)
            self.edge_set2set = Set2Set(edge_out_channels, processing_steps=set2set_processing_steps)


    def forward(self, x, edge_index, edge_attr, fidelity_indicator, batch_mapping):
        fidelity_emb = self.fidelity_embedding(fidelity_indicator)

        x, edge_attr, fidelity_emb = self.mfmp_0(x, edge_index, edge_attr, fidelity_emb, batch_mapping)
        x, edge_attr, fidelity_emb = F.relu(x), F.relu(edge_attr), F.relu(fidelity_emb)

        x, edge_attr, fidelity_emb = self.mfmp_1(x, edge_index, edge_attr, fidelity_emb, batch_mapping)
        x, edge_attr, fidelity_emb = F.relu(x), F.relu(edge_attr), F.relu(fidelity_emb)

        x, edge_attr, fidelity_emb = self.mfmp_2(x, edge_index, edge_attr, fidelity_emb, batch_mapping)
        x, edge_attr, fidelity_emb = F.relu(x), F.relu(edge_attr), F.relu(fidelity_emb)

        edge_batch_index = batch_mapping.index_select(dim=0, index=edge_index[0, :])

        if self.use_set2set:
            x_batched = self.node_set2set(x, batch_mapping)
            edge_attr_batched = self.edge_set2set(edge_attr, edge_batch_index)
        else:
            x_batched = torch.cat((
                global_add_pool(x, batch_mapping),
                global_mean_pool(x, batch_mapping),
                global_max_pool(x, batch_mapping)
            ), dim=1)

            edge_attr_batched = torch.cat((
                global_add_pool(edge_attr, edge_batch_index),
                global_mean_pool(edge_attr, edge_batch_index),
                global_max_pool(edge_attr, edge_batch_index)
            ), dim=1)

        return x_batched, edge_attr_batched, fidelity_emb