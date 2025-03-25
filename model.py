import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

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
            ('batch_normalization_0', nn.BatchNorm1d(args.latent_dim1)),  # 使用LayerNorm代替BatchNorm1d
            ('activation_0', nn.PReLU()),
            ('dropout_0', nn.Dropout(p=args.gnn_dropout)),  # 添加Dropout层
            ('message_passing_0', Message_Passing()),
            ('dense_1', nn.Linear(args.latent_dim1, args.latent_dim2)),
            ('batch_normalization_1', nn.BatchNorm1d(args.latent_dim2)),
            ('activation_1', nn.PReLU()),
            ('dropout_1', nn.Dropout(p=args.gnn_dropout)),  # 添加Dropout层
            ('message_passing_1', Message_Passing()),
            ('dense_2', nn.Linear(args.latent_dim2, args.latent_dim3)),
            ('batch_normalization_2', nn.BatchNorm1d(args.latent_dim3)),
            ('activation_2', nn.PReLU()),
            ('dropout_2', nn.Dropout(p=args.gnn_dropout)),  # 添加Dropout层
            ('message_passing_2', Message_Passing()),
        ]))

        self.fully_connected = nn.Sequential(
            nn.Linear(args.max_node_num * args.latent_dim3, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.gnn_out_dim)
        )

        # Apply He initialization to the linear layers
        self._apply_he_init()

    def _apply_he_init(self):
        # Iterate through each module in the graph_modules and fully_connected
        for module in self.graph_modules:
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

        # Apply He initialization to the fully connected layers
        for module in self.fully_connected:
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, adjacency_matrix, node_attr_matrix):
        adjacency_matrix = adjacency_matrix.float()
        node_attr_matrix = node_attr_matrix.float()

        x = node_attr_matrix
        # x1 = x

        for (name, module) in self.graph_modules.named_children():
            if 'batch_normalization' in name:
                # 调整形状，以符合 nn.BatchNorm1d 的输入要求
                x = x.view(-1, x.size(-1))  # (batch_size * num_nodes, num_features)
                x = module(x)  # 进行批归一化
                x = x.view(x.size(0) // adjacency_matrix.size(1), adjacency_matrix.size(1), -1)  # 恢复为 (batch_size, num_nodes, num_features)
            elif 'message_passing' in name:
                x = module(x, adjacency_matrix)
            else:
                x = module(x)

        # Reshape x for the final fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all nodes' features, shape: (batch_size, max_node_num * latent_dim3)

        # Pass through the fully connected layers for the final output
        x = self.fully_connected(x)

        final = x

        return final
        # return nn.functional.normalize(final, p=2, dim=1)


# class RNN(nn.Module):#Transformer
#     def __init__(self, args):
#         super(RNN, self).__init__()
#
#         self.fc1_in_transformer = nn.Sequential(
#             nn.Linear(10, args.strain_dim)
#         )
#
#         # Transformer encoder层
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.input_size, nhead=4, dropout=args.lstm_dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_layers)
#
#         self.fc1_out_transformer = nn.Sequential(
#             nn.Linear(args.input_size, 1)
#         )
#
#     def forward(self, gnn_outputs, strain_matrix):
#         strain_matrix = self.fc1_in_transformer(strain_matrix)
#
#         x = gnn_outputs
#         x = x.unsqueeze(1).repeat(1, strain_matrix.shape[1], 1)
#         x = torch.cat((x, strain_matrix), dim=-1)
#
#         # 将输入形状调整为 (seq_len, batch_size, feature_dim) 适应 Transformer
#         x = x.permute(1, 0, 2)  # (seq_len, batch_size, feature_dim) -> (seq_len, batch_size, feature_dim)
#
#         # 传入 Transformer 编码器
#         x = self.transformer_encoder(x)
#
#         # 将输出恢复为 (batch_size, seq_len, hidden_size)
#         x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
#
#         # 对 Transformer 输出应用全连接层
#         x = self.fc1_out_transformer(x)
#
#         return x

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        self.fc1_in_rnn = nn.Sequential(
            nn.Linear(10, args.strain_dim)
        )

        self.rnn = nn.LSTM(input_size = args.input_size,
                           hidden_size = args.hid_size,
                           num_layers = args.num_layers,
                           batch_first=True,
                           dropout = args.lstm_dropout)

        self.fc1_out_rnn = nn.Sequential(
            nn.Linear(args.hid_size , 1)
        )

    def forward(self, gnn_outputs, strain_matrix):
        strain_matrix = self.fc1_in_rnn(strain_matrix)

        x = gnn_outputs
        x = x.unsqueeze(1).repeat(1, strain_matrix.shape[1], 1)
        x = torch.cat((x, strain_matrix), dim=-1)

        x, _ = self.rnn(x)
        x = self.fc1_out_rnn(x)

        return x