import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设备选择 'CPU' or 'GPU'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Message_Passing(nn.Module):
    def forward(self, x, adjacency_matrix):
        neighbor_nodes = torch.bmm(adjacency_matrix, x)

        return neighbor_nodes


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()

        self.graph_modules = nn.Sequential(OrderedDict([
            ('dense_0', nn.Linear(args.atom_attr_dim, args.latent_dim1)),
            ('batch_normalization_0', nn.BatchNorm1d(args.max_node_num)),
            ('activation_0', nn.PReLU()),
            ('message_passing_0', Message_Passing()),
            ('dense_1', nn.Linear(args.latent_dim1, args.latent_dim2)),
            ('batch_normalization_1', nn.BatchNorm1d(args.max_node_num)),
            ('activation_1', nn.PReLU()),
            ('message_passing_1', Message_Passing()),
            ('dense_2', nn.Linear(args.latent_dim2, args.latent_dim3)),
            ('batch_normalization_2', nn.BatchNorm1d(args.max_node_num)),
            ('activation_2', nn.PReLU()),
            ('message_passing_2', Message_Passing())
        ]))

        self.fully_connected = nn.Sequential(
            nn.Linear(args.max_node_num * args.latent_dim3, 512),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, args.gnn_out_dim)
        )

    def forward(self, adjacency_matrix, node_attr_matrix):
        adjacency_matrix = adjacency_matrix.float()
        node_attr_matrix = node_attr_matrix.float()

        # Extract only the 5th feature from the original node attributes for weighting
        node_attr_matrix_5th_feature = node_attr_matrix[:, :, 4]  # Shape: (batch_size, num_nodes)

        x = node_attr_matrix

        for (name, module) in self.graph_modules.named_children():
            if 'message_passing' in name:
                x = module(x, adjacency_matrix)
            else:
                x = module(x)

        # Reshape x for the final fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all nodes' features, shape: (batch_size, max_node_num * latent_dim3)

        # Pass through the fully connected layers for the final output
        x = self.fully_connected(x)

        return nn.functional.normalize(x, p=2, dim=1)


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        self.fc1_in_rnn = nn.Sequential(
            nn.Linear(10, args.strain_dim)
        )

        self.rnn = nn.LSTM(input_size=args.input_size,
                           hidden_size=args.hid_size,
                           num_layers=args.num_layers,
                           batch_first=True,
                           dropout=args.lstm_dropout)

        self.fc1_out_rnn = nn.Sequential(
            nn.Linear(args.hid_size, 1)
        )

    def forward(self, gnn_outputs, strain_matrix, hidden=None):
        strain_matrix = self.fc1_in_rnn(strain_matrix)

        x = gnn_outputs
        x = x.unsqueeze(1).repeat(1, strain_matrix.shape[1], 1)
        x = torch.cat((x, strain_matrix), dim=-1)

        x, _ = self.rnn(x)
        x = self.fc1_out_rnn(x)

        return x