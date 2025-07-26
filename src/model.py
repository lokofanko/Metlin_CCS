import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, GATv2Conv
import torch_geometric.nn as geom_nn

class GCNWithCategoricalFeature(torch.nn.Module):
    def __init__(self, num_graph_features, num_cat_features, hidden_dim=128, fc_hidden_dim=600):
        super().__init__()
        
        # Графовые сверточные слои
        self.conv1 = GATConv(num_graph_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(hidden_dim + num_cat_features, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.05)
        self.dropout_conv = nn.Dropout(p=0.05)

        # Batch Normalization (закомментированы, как у тебя)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # ...

    def forward(self, data):
        graph_data, cat_features = data[0], data[1]
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout_conv(x)
        
        x3 = F.relu(self.conv3(x, edge_index))
        x3 = self.dropout_conv(x3)
        x = x3 + x  # Skip connection

        x = F.relu(self.conv4(x, edge_index))
        x = self.dropout_conv(x)
        
        x = geom_nn.global_add_pool(x, batch)
        x = torch.cat([x, cat_features], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x