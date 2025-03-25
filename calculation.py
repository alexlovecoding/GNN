from __future__ import print_function
import time
import threading
import torch.nn as nn
from draw_figure import Draw_comparison_figure, Draw_loss_figure
from util import *

# 设备选择 'CPU' or 'GPU'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 获取预测值和真实值的长度
        length = y_pred.size(-1)
        # 生成权重系数，前面的权重大，后面的权重小
        weights = torch.linspace(1, 0.1, length, device=y_pred.device)
        # 对权重进行归一化处理，使权重之和为 1
        weights = weights / torch.sum(weights)
        # 计算预测值和真实值之差的平方
        squared_diff = (y_pred - y_true) ** 2
        # 乘以权重系数
        weighted_squared_diff = squared_diff * weights
        # 取均值得到最终的损失值
        loss = 500 * torch.mean(weighted_squared_diff)
        return loss

def adjust_lr(optimizer, epoch, args):
    frac = (epoch - args.lr_decay_start) // args.lr_decay_every
    decay_factor = args.lr_decay_rate ** frac
    current_lr = args.learning_rate * decay_factor

    for group in optimizer.param_groups:
        group['lr'] = current_lr


class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = None
        self.path = path

    def __call__(self, val_loss, model_gnn, model_rnn):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model_gnn, model_rnn)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model_gnn, model_rnn)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_gnn, model_rnn):
        '''保存最佳模型'''
        torch.save({
            'model_gnn_state_dict': model_gnn.state_dict(),
            'model_rnn_state_dict': model_rnn.state_dict(),
        }, self.path)
        self.best_loss = val_loss


def train(net_gnn, net_rnn, train_dataloader, validation_dataloader, args):
    print()
    print("*** Training started! ***")
    print()

    filename = '{}/learning_Output.txt'.format(args.folder_name)
    output = open(filename, "w")
    print('Epoch Training_time Training_MSE Validation_MSE', file=output, flush=True)

    # loss_func = nn.SmoothL1Loss()
    # loss_func = nn.L1Loss()
    #loss_func = SMAPELoss()
    loss_func = WeightedMSELoss()
    params = list(net_gnn.parameters()) + list(net_rnn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999),
                                 eps=1e-08)

    net_gnn.to(device)
    net_rnn.to(device)

    training_loss = []
    validation_loss = []

    # 获取当前线程 ID
    thread_id = threading.get_ident()
    # 在 path 中添加线程 ID
    save_path = f'best_model_{thread_id}.pt'
    early_stopping = EarlyStopping(patience=args.patience, delta=0.000001, path=save_path)
    # early_stopping = EarlyStopping(patience=30, delta=0.000001, path='best_model.pt')  # 实例化 EarlyStopping 类

    for epoch in range(args.epochs):

        net_gnn.train()
        net_rnn.train()

        train_start_time = time.time()

        if epoch >= args.lr_decay_start:
            adjust_lr(optimizer, epoch, args)

        for batch_id, (adjacency_matrix, node_attr_matrix, strain_matrix, label_matrix) in enumerate(train_dataloader):
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            strain_matrix = tensor_to_variable(strain_matrix)
            label_matrix = tensor_to_variable(label_matrix)

            # print(adjacency_matrix)

            optimizer.zero_grad()

            gnn_outputs = net_gnn(adjacency_matrix, node_attr_matrix)
            y_pred = net_rnn(gnn_outputs, strain_matrix)

            loss = loss_func(y_pred.squeeze(), label_matrix.squeeze())
            loss.backward()
            optimizer.step()

        train_end_time = time.time()

        _, training_loss_epoch = test(net_gnn, net_rnn, train_dataloader, 'train', False, args)
        _, validation_loss_epoch = test(net_gnn, net_rnn, validation_dataloader, 'validation', False, args)

        training_loss.append(training_loss_epoch)
        validation_loss.append(validation_loss_epoch)

        print('%d %e %e' % (epoch, training_loss_epoch, validation_loss_epoch), flush=True)
        print('%d %.3f %e %e' % (epoch, train_end_time - train_start_time, training_loss_epoch, validation_loss_epoch),
              file=output, flush=True)

        early_stopping(validation_loss_epoch, net_gnn, net_rnn)  # 调用 EarlyStopping

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print()
    print("*** Training finished! ***")
    print()

    # 加载最佳模型权重
    checkpoint = torch.load(save_path)
    net_gnn.load_state_dict(checkpoint['model_gnn_state_dict'])
    net_rnn.load_state_dict(checkpoint['model_rnn_state_dict'])

    Draw_loss_figure(len(training_loss), training_loss, validation_loss, args.folder_name)



def test(net_gnn, net_rnn, data_loader, tra_val_test, printcond, args):
    net_gnn.to(device)
    net_rnn.to(device)
    net_gnn.eval()
    net_rnn.eval()

    if data_loader is None:
        return None, None

    strain_list, y_label_list, y_pred_list, rel_err, total_mse =[], [], [], 0, 0

    # loss_func = nn.L1Loss()
    #loss_func = SMAPELoss()
    loss_func = WeightedMSELoss()

    with torch.no_grad():
        for batch_id, (adjacency_matrix, node_attr_matrix, strain_matrix, label_matrix) in enumerate(data_loader): 
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            strain_matrix = tensor_to_variable(strain_matrix)
            label_matrix = tensor_to_variable(label_matrix)

            gnn_outputs = net_gnn(adjacency_matrix, node_attr_matrix)
            
            # y_pred = gnn_outputs
            y_pred = net_rnn(gnn_outputs, strain_matrix)

            strain_list.extend(variable_to_numpy(strain_matrix))
            y_pred_list.extend(variable_to_numpy(y_pred.squeeze()))
            y_label_list.extend(variable_to_numpy(label_matrix))
    
    norm = np.load('norm.npz', allow_pickle=True)['norm']
    strain_max, strain_min, label_max, label_min = norm[0], norm[1], norm[2], norm[3]
    strain_list = np.array(strain_list) * (strain_max - strain_min) + strain_min
    y_pred_list = np.array(y_pred_list) * (label_max - label_min) + label_min
    y_label_list = np.array(y_label_list) * (label_max - label_min) + label_min
    
    rel_err = avg_rel_err(y_pred_list, y_label_list)
    total_mse = loss_func(torch.from_numpy(y_pred_list), torch.from_numpy(y_label_list)).item()

    length, w = np.shape(y_label_list)
    if printcond:
        filename = '{}/{}_Output.txt'.format(args.folder_name, tra_val_test)
        output = open(filename, 'w')
        #print()
        print('{} Set Predictions: '.format(tra_val_test), file = output, flush = True)
        for i in range(0, length):
            print('sample-%d' % (i+1), file=output, flush = True)
            print('--------------------------------------', file=output, flush = True)
            print('Predicted_value:', file=output, flush = True)
            for j in range(0, w):
                print('%f, ' % (y_pred_list[i, j]), file=output, flush = True, end="")
            print('\nTrue_value:', file=output, flush = True)
            for j in range(0, w):
                print('%f, ' % (y_label_list[i, j]), file=output, flush = True, end="")
            print('\n--------------------------------------', file=output, flush = True)

    if tra_val_test == 'Validation':
        save_fig = '{}/{}_figure/'.format(args.folder_name, tra_val_test)
        Draw_comparison_figure(strain_list, y_pred_list, y_label_list, save_fig)

    if tra_val_test == 'Test':
        save_fig = '{}/{}_figure/'.format(args.folder_name, tra_val_test)
        Draw_comparison_figure(strain_list, y_pred_list, y_label_list, save_fig)

    return rel_err, total_mse