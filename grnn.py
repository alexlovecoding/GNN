"""
This program is for GRNN

@author: ljq at hust 2024.3.25
"""
from __future__ import print_function

import os
import random
import time
# 忽略警告
import warnings
from datetime import datetime

from calculation import train, test
from data import *
from model0 import GNN, RNN

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Parse input arguments
# ---------------------------------------------------
parser = argparse.ArgumentParser()
# 
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--manualseed', default=3407, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--folder_name', default='output/', type=str)
parser.add_argument('--patience', default=50, type=int)
# 
parser.add_argument('--max_node_num', default=64, type=int)
parser.add_argument('--atom_attr_dim', default=5, type=int)
parser.add_argument('--validation_index', default=17145, type=int)
parser.add_argument('--test_index', default=17445, type=int)
parser.add_argument('--num_data', default=17745, type=int)

# set parameters for GNN
parser.add_argument('--latent_dim1', default=100, type=int)
parser.add_argument('--latent_dim2', default=5, type=int)
parser.add_argument('--latent_dim3', default=3, type=int)
parser.add_argument('--gnn_out_dim', default=64, type=int)
parser.add_argument('--gnn_dropout', default=0.0, type=float)
# set parameters for LSTM
parser.add_argument('--strain_dim', default=32, type=int)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--hid_size', default=10, type=int)
parser.add_argument('--lstm_dropout', default=0.5, type=float, help='dropout of LSTM')
# 
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--lr_decay_start', default=100, type=int, help='set to -1 if no learning rate decay is adopted')
parser.add_argument('--lr_decay_every', default=10, type=int, help='the learning rate times the learning rate decay rate every lr_decay_every epochs')
parser.add_argument('--lr_decay_rate', default=0.867, type=float, help='the learning rate times lr_decay_rate every lr_decay_every epochs')
# 
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

args = parser.parse_args()

# This fuction sets all the random seeds to make the results are reproducible
def set_seed():
    random.seed(args.manualseed)
    os.environ['PYTHONHASHSEED'] = str(args.manualseed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    np.random.seed(args.manualseed)
    torch.manual_seed(args.manualseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.manualseed)  
        torch.cuda.manual_seed_all(args.manualseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

# 设备选择 'CPU' or 'GPU'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    set_seed()

    # 输出设备信息
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.1f} MB")

    start_time = datetime.now()
    args.folder_name = args.folder_name + str(start_time.strftime("%Y-%m-%d-%H-%M-%S"))

    if not os.path.exists(args.folder_name):
        os.makedirs(args.folder_name)

    checkpoint_dir = args.folder_name + '/checkpoint_dir'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Define the model
    net_gnn = GNN(args)
    net_rnn = RNN(args)

    # get the data
    train_dataloader, validation_dataloader, test_dataloader = get_data(args)

    # train the model
    train_start_time = time.time()
    train(net_gnn, net_rnn, train_dataloader, validation_dataloader, args)
    train_end_time = time.time()

    torch.save(net_gnn.state_dict(), '{}/net_gnn.pth'.format(checkpoint_dir))
    torch.save(net_rnn.state_dict(), '{}/net_rnn.pth'.format(checkpoint_dir))
    
    # predictions on the entire training and test datasets
    train_rel, train_mse= test(net_gnn, net_rnn, train_dataloader, 'Train', True, args)
    validation_rel, validation_mse=test(net_gnn, net_rnn, validation_dataloader, 'Validation', True, args)
    test_start_time = time.time()
    test_rel, test_mse= test(net_gnn, net_rnn, test_dataloader, 'Test', True, args)
    test_end_time = time.time()

    print('--------------------')
    print("training_time : {}".format(train_end_time-train_start_time))
    print("testing_time : {}".format(test_end_time-test_start_time))
    print("Train Relative Error: {:.3f}%".format(100 * train_rel))
    print("Validation Relative Error: {:.3f}%".format(100 * validation_rel))
    print("Test Relative Error: {:.3f}%".format(100 * test_rel))
    print("Train loss : {}".format(train_mse))
    print("Validation loss : {}".format(validation_mse))
    print("Test loss: {}".format(test_mse))  

    filename='{}/message.txt'.format(args.folder_name)
    output=open(filename, "w")
    print('--------------------', file=output, flush = True)
    print('Details of args', file=output, flush = True)
    print('--------------------', file=output, flush = True)
    print('', file=output, flush = True)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]), file=output, flush = True)
    print('', file=output, flush = True)
    print('----------------------------------', file=output, flush = True)
    print('Evaluation of the current model', file=output, flush = True)
    print('----------------------------------', file=output, flush = True)
    print('', file=output, flush = True)
    print("Train Relative Error: {:.3f}%".format(100 * train_rel), file=output, flush = True)
    print("Validation Relative Error: {:.3f}%".format(100 * validation_rel), file=output, flush = True)
    print("Test Relative Error: {:.3f}%".format(100 * test_rel), file=output, flush = True)