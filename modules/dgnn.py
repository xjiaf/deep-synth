import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init, Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import NeighborSampler


class DGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.35):
        super(DGNNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.delta = Parameter(torch.tensor(1.0))
        self.edge_weight_norm = nn.BatchNorm1d(1, affine=True)  # Added for edge weights normalization
        self.w_self = Linear(in_channels, out_channels)
        self.w_hist = Linear(in_channels, out_channels)
        self.fc = nn.Sequential(nn.BatchNorm1d(out_channels),
                                nn.ReLU(),  # Changed to nn.ReLU
                                Linear(out_channels, out_channels),
                                nn.Dropout(dropout)
                                )

        # Initialize parameters
        init.xavier_normal_(self.delta)
        init.xavier_normal_(self.w_self.weight)
        init.xavier_normal_(self.w_hist.weight)

    def forward(self, x, edge_index, edge_weights, edge_times, node_time):
        return self.propagate(edge_index, x=x, edge_weights=edge_weights,
                              edge_times=edge_times, node_time=node_time)

    def message(self, x_j, edge_weights, edge_times, node_time):
        # Calculate z-score for edge_weights using nn.BatchNorm1d
        z_scored_weights = self.edge_weight_norm(edge_weights.unsqueeze(1)).squeeze(1)  # Modified for nn.BatchNorm1d

        # Calculate normalized kappa using softmax
        time_diffs = node_time.unsqueeze(-1) - edge_times
        normalized_kappa = F.softmax(-self.delta * time_diffs, dim=1)

        return z_scored_weights * normalized_kappa * x_j

    def update(self, aggr_out, x):
        out = self.w_self(x) + self.w_hist(aggr_out)
        return self.fc(out)


class DGNN(nn.Module):
    def __init__(self, params, num_node_features, neighbor_sampler_sizes=[-1, -1]):
        super(DGNN, self).__init__()
        self.params = params
        self.neighbor_sampler_sizes = neighbor_sampler_sizes
        self.conv1 = DGNNConv(num_node_features, params['dgnn_hid_dim'], dropout=params['dropout'])
        self.conv2 = DGNNConv(params['dgnn_hid_dim'], params['dgnn_out_dim'], dropout=params['dropout'])

    def forward(self, graph, node_ids, node_time):
        x, edge_index, edge_weights, edge_times = graph.x, graph.edge_index, graph.edge_weight, graph.edge_time

        # Check if node_time is a scalar tensor or a simple scalar value (like float or int).
        if torch.is_tensor(node_time) and node_time.dim() == 0 or not torch.is_tensor(node_time):
            node_time = torch.full(node_ids.size(), float(node_time))

        # Use NeighborSampler to sample a two-hop subgraph for the given node_ids
        sampler = NeighborSampler(edge_index, sizes=self.neighbor_sampler_sizes, num_nodes=graph.num_nodes, node_idx=node_ids, batch_size=len(node_ids))

        outs = []
        for _, n_id, adjs in sampler:
            # For each node, filter edges based on its node_time
            node_specific_time = node_time[n_id[0]]
            valid_edges_mask = edge_times <= node_specific_time
            local_edge_weights = edge_weights[valid_edges_mask]
            local_edge_times = edge_times[valid_edges_mask]

            two_hop_indices, two_hop_e_id = adjs[1]
            one_hop_indices, one_hop_e_id = adjs[0]

            # First message passing (2-hop to 1-hop)
            inter_result = self.conv1(x[n_id], two_hop_indices, local_edge_weights[two_hop_e_id],
                                      local_edge_times[two_hop_e_id], node_specific_time)
            x[n_id] += inter_result

            # Second message passing (1-hop to target node)
            out = self.conv2(x[n_id], one_hop_indices, local_edge_weights[one_hop_e_id],
                             local_edge_times[one_hop_e_id], node_specific_time)
            outs.append(out)

        return torch.cat(outs, dim=0)
