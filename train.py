import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import DGDCN
from datasets.temporal_graph import TemporalGraph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(params: dict, train_loader, device):
    # Load graph
    origin_edge_path = params['processed_data_path'] / params['dataset'] / params['origin_edge_file']
    x_path = params['processed_data_path'] / params['dataset'] / params['x_file']
    graph = TemporalGraph(params, origin_edge_path, x_path).to(device)

    # Initialize
    model = init_model(params=params, graph=graph).to(device)
    optimizer = optim.Adam([{'params': model.parameters()}], lr=params['lr'], weight_decay=params['weight_decay']).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    model = torch.compile(model)  # pytorch compile to accelerate

    # TODO: dataloader
    for inputs, labels in train_loader:
        outputs = model(graph, item_ids, item_time, origin_emb)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=5, norm_type=2.0)
        optimizer.step()


def init_model(params: dict, graph: TemporalGraph):
    """Initialize the model"""
    if params['model'] == 'dgdcn':
        # Assuming linear_feature_columns and dnn_feature_columns are already defined
        linear_feature_columns = params['linear_feature_columns']
        dnn_feature_columns = params['dnn_feature_columns']
        model = DGDCN(params, graph.num_node_features, linear_feature_columns, dnn_feature_columns)

    return model


def test_model(params: dict, model: nn.Module, test_loader):
    model.eval()
    with torch.no_grad():
        pass


def infoNCE(pos_sim, neg_sim, temperature=1.0):
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature

    # max 1000
    pos_sim = pos_sim.clamp(max=1e4)
    neg_sim = neg_sim.clamp(max=1e4)

    # numerator = torch.exp(pos_sim)
    denominator = torch.exp(pos_sim) + torch.exp(neg_sim).sum(dim=-1)
    # print("numerator {0} and denominator {1}".format(numerator, denominator))
    loss = torch.log(denominator) - pos_sim
    return loss.mean()
