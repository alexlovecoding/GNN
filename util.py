import torch
import numpy as np
from torch.autograd import Variable


def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def avg_rel_err(Y_pred, Y):
    if type(Y_pred) is np.ndarray:
        return np.mean(np.abs(Y - Y_pred) / np.abs(Y))
    return torch.mean(torch.abs(Y - Y_pred) / torch.abs(Y))