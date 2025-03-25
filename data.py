# load the data into the dataset
from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader, Subset

class GraphDataSet(Dataset):
    def __init__(self, csv_path, max_node_num, atom_attr_dim):
        # Load the dataset
        data = pd.read_csv(csv_path)

        # Extract relevant sections of the data
        self.label_matrix = data.iloc[:, 0:10].values  # Columns 1-10
        self.strain_matrix = data.iloc[:, 10:20].values  # Columns 11-20
        self.strain_matrix = manipulate_strain(self.strain_matrix)

        features = data.iloc[:, 20:20 + (max_node_num * atom_attr_dim)].values  # Columns for features
        adjacency = data.iloc[:, 20 + (max_node_num * atom_attr_dim):].values  # Columns for adjacency

        # Reshape features and adjacency
        self.node_attr_matrix = features.reshape(-1, max_node_num, atom_attr_dim)  # (num_data, max_node_num, atom_attr_dim)
        self.adjacency_matrix = adjacency.reshape(-1, max_node_num, max_node_num)  # (num_data, max_node_num, max_node_num)

        self.adjacency_matrix = torch.tensor(self.adjacency_matrix, dtype=torch.float)

        # Normalize strain and label matrices
        self.strain_matrix, self.label_matrix = normalize_strain_label(self.strain_matrix, self.label_matrix)

        print('--------------------')
        print('total Data:')
        print('adjacency matrix:\t', self.adjacency_matrix.shape)
        print('node attribute matrix:\t', self.node_attr_matrix.shape)
        print('strain matrix:\t\t', self.strain_matrix.shape)
        print('label matrix:\t\t', self.label_matrix.shape)
        print('--------------------')

    def __len__(self):
        return len(self.adjacency_matrix)

    def __getitem__(self, idx):
        adjacency_matrix = self.adjacency_matrix[idx]
        node_attr_matrix = self.node_attr_matrix[idx]
        strain_matrix = self.strain_matrix[idx]
        label_matrix = self.label_matrix[idx]

        node_attr_matrix = torch.from_numpy(node_attr_matrix)
        strain_matrix = torch.from_numpy(strain_matrix)
        label_matrix = torch.from_numpy(label_matrix)

        return adjacency_matrix, node_attr_matrix, strain_matrix, label_matrix

def manipulate_strain(data):
    """
    Apply a mask to the input data, converting each row into a lower triangular form.
    Args:
        data (np.ndarray): A 2D array with shape (num_samples, seq_length).
    Returns:
        np.ndarray: A 3D array with shape (num_samples, seq_length, seq_length), where
                    each row of the input is transformed into a lower triangular matrix.
    """
    num_samples, seq_length = data.shape
    modified_strain = np.zeros((num_samples, seq_length, seq_length))

    for sample_idx in range(num_samples):
        for i in range(seq_length):
            for j in range(seq_length):
                if j <= i:
                    modified_strain[sample_idx, i, j] = data[sample_idx, j]

    return modified_strain


def normalize_strain_label(strain_matrix, label_matrix):
    strain_max = np.max(strain_matrix)
    strain_min = np.min(strain_matrix)
    strain_matrix = (strain_matrix - strain_min) / (strain_max - strain_min)
    
    label_max = np.max(label_matrix)
    label_min = np.min(label_matrix)
    label_matrix = (label_matrix - label_min) / (label_max - label_min)

    norm = np.array([strain_max, strain_min, label_max, label_min])
    np.savez_compressed('norm.npz', norm=norm)

    return strain_matrix, label_matrix

def get_data(args):
    batch_size = args.batch_size
    validation_index = args.validation_index
    test_index = args.test_index
    num_data = args.num_data
    max_node_num = args.max_node_num
    atom_attr_dim = args.atom_attr_dim

    # Initialize dataset
    # dataset = GraphDataSet('data/augmented_dataset_64_5_2n2d_16295.csv', max_node_num, atom_attr_dim)
    dataset = GraphDataSet('../aug_data/data/new_64_1/augmented_dataset_17745.csv', max_node_num, atom_attr_dim)
    # dataset = GraphDataSet('data/augmented_dataset_64_5_6156_6756.csv', max_node_num, atom_attr_dim)
    # dataset = GraphDataSet('data/sorted_dataset.csv', max_node_num, atom_attr_dim)


    train_idx, validation_idx, test_idx = np.arange(0, validation_index), np.arange(validation_index, test_index), np.arange(test_index, num_data)

    train_set, validation_set, test_set = Subset(dataset, train_idx), Subset(dataset, validation_idx), Subset(dataset, test_idx)

    # Create data loaders
    train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_data, validation_data, test_data