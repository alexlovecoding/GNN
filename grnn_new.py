"""
This program is for GRNN

@author: ljq at hust 2024.3.25
@modifier: Lyh at hust 2024.9.6
"""
from __future__ import print_function
import argparse
import time
from datetime import datetime
import os
import numpy as np
import torch
import random
from util import *
from data import *
from model import GNN, RNN
from calculation import train, test

import threading

import warnings
warnings.filterwarnings("ignore")

# 获取命令行参数
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--manualseed', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--folder_name', default='output/', type=str)
    parser.add_argument('--patience', default=40, type=int)
    #
    parser.add_argument('--max_node_num', default=65, type=int)
    parser.add_argument('--atom_attr_dim', default=8, type=int)
    parser.add_argument('--validation_index', default=5400, type=int)
    parser.add_argument('--test_index', default=5700, type=int)
    parser.add_argument('--num_data', default=6000, type=int)
    # set parameters for GNN
    parser.add_argument('--latent_dim1', default=100, type=int)
    parser.add_argument('--latent_dim2', default=5, type=int)
    parser.add_argument('--latent_dim3', default=3, type=int)
    parser.add_argument('--gnn_out_dim', default=64, type=int)
    parser.add_argument('--gnn_dropout', default=0.3, type=float)
    # set parameters for LSTM
    parser.add_argument('--strain_dim', default=32, type=int)
    parser.add_argument('--input_size', default=96, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--hid_size', default=10, type=int)
    parser.add_argument('--lstm_dropout', default=0.5, type=float, help='dropout of LSTM')

    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_start', default=100, type=int, help='set to -1 if no learning rate decay is adopted')
    parser.add_argument('--lr_decay_every', default=10, type=int, help='the learning rate times the learning rate decay rate every lr_decay_every epochs')
    parser.add_argument('--lr_decay_rate', default=0.867, type=float, help='the learning rate times lr_decay_rate every lr_decay_every epochs')
    #
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    return parser.parse_args()


# 设置随机种子
def set_seed(args):
    random.seed(args.manualseed)
    os.environ['PYTHONHASHSEED'] = str(args.manualseed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(args.manualseed)
    torch.manual_seed(args.manualseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.manualseed)
        torch.cuda.manual_seed_all(args.manualseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

# 训练模型的主函数
def train_model(args, train_dataloader, validation_dataloader, test_dataloader):
    set_seed(args)

    # 设备选择 'CPU' or 'GPU'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                     
    '''
    print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU device name: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.1f} MB")

    '''

    # 创建输出文件夹
    start_time = datetime.now()
    # 获取当前线程 ID
    thread_id = threading.get_ident()

    # 在 folder_name 中添加线程 ID
    args.folder_name = f"{args.folder_name}_{start_time.strftime('%Y-%m-%d-%H-%M-%S')}_{thread_id}"

    # # 尝试创建文件夹，如果文件夹已存在，递增后缀数字
    # # 通过这个方式可以实现多线程并行
    # counter = 1
    # while os.path.exists(args.folder_name):
    #     args.folder_name = f"output/{args.folder_name}_{counter}"
    #     counter += 1

    if not os.path.exists(args.folder_name):
        os.makedirs(args.folder_name)

    checkpoint_dir = args.folder_name + '/checkpoint_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 定义模型
    net_gnn = GNN(args)
    net_rnn = RNN(args)


    # 训练模型
    train_start_time = time.time()
    train(net_gnn, net_rnn, train_dataloader, validation_dataloader, args)
    train_end_time = time.time()

    # 保存模型
    torch.save(net_gnn.state_dict(), f'{checkpoint_dir}/net_gnn.pth')
    torch.save(net_rnn.state_dict(), f'{checkpoint_dir}/net_rnn.pth')

    # 测试模型
    train_rel, train_mse = test(net_gnn, net_rnn, train_dataloader, 'Train', True, args)
    validation_rel, validation_mse = test(net_gnn, net_rnn, validation_dataloader, 'Validation', True, args)
    test_rel, test_mse = test(net_gnn, net_rnn, test_dataloader, 'Test', True, args)

    print('--------------------')
    print(f"Training Time: {train_end_time - train_start_time}")
    print(f"Train Relative Error: {100 * train_rel:.3f}%")
    print(f"Validation Relative Error: {100 * validation_rel:.3f}%")
    print(f"Test Relative Error: {100 * test_rel:.3f}%")
    print(f"Train Loss: {train_mse}")
    print(f"Validation Loss: {validation_mse}")
    print(f"Test Loss: {test_mse}")


    # 输出参数和结果到文件
    filename = f'{args.folder_name}/message.txt'
    with open(filename, "w") as output:
        print('--------------------', file=output, flush=True)
        print('Details of args', file=output, flush=True)
        print('--------------------', file=output, flush=True)
        for k, v in vars(args).items():
            print(f"{k}: {v}", file=output, flush=True)
        print('--------------------', file=output, flush=True)
        print('Evaluation of the current model', file=output, flush=True)
        print('--------------------', file=output, flush=True)
        print(f"Train Relative Error: {100 * train_rel:.3f}%", file=output, flush=True)
        print(f"Validation Relative Error: {100 * validation_rel:.3f}%", file=output, flush=True)
        print(f"Test Relative Error: {100 * test_rel:.3f}%", file=output, flush=True)

    return validation_mse

# 如果作为独立脚本运行，执行训练流程
if __name__ == "__main__":
    args = get_args()
    # 加载数据
    train_dataloader, validation_dataloader, test_dataloader = get_data(args)

    train_model(args, train_dataloader, validation_dataloader, test_dataloader)
