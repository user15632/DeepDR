"""
DNN CNN NULL
input: exp, gsva or cnv [batch_size, in_dim]
output: cell_ft [batch_size, cell_dim]
"""

import torch
from torch import nn
import torch.nn.functional as F

_in_dim = 6163
_cell_dim = 512
_dropout = 0.3

_kernel_size = 3
_padding = 1


class DNN(nn.Module):
    def __init__(self, in_dim: int = _in_dim, ft_dim: int = _cell_dim, dropout: float = _dropout,
                 hid_dim: int = 1024, num_layers: int = 3):
        """hid_dim, num_layers"""
        super(DNN, self).__init__()
        assert num_layers >= 2
        dim_ls = [in_dim] + [hid_dim] * (num_layers - 1) + [ft_dim]
        self.encode_dnn = nn.ModuleList([nn.Linear(dim_ls[i], dim_ls[i + 1]) for i in range(num_layers - 1)])
        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers - 1)])
        self.output = nn.Linear(dim_ls[-2], dim_ls[-1])

    def forward(self, f):
        for i in range(len(self.encode_dnn)):
            f = F.relu(self.encode_dnn[i](f))
            f = self.dropout[i](f)
        f = self.output(f)
        return f


class CNN(nn.Module):
    def __init__(self, in_dim: int = _in_dim, ft_dim: int = _cell_dim, kernel_size: int = _kernel_size,
                 padding: int = _padding, hid_dim: int = 64, num_layers: int = 3):
        """hid_dim, num_layers"""
        super(CNN, self).__init__()
        assert num_layers >= 1
        dim_ls = [1] + [hid_dim] * num_layers
        self.input = nn.Linear(in_dim, ft_dim)
        self.encode_conv = nn.ModuleList([nn.Conv1d(in_channels=dim_ls[i], out_channels=dim_ls[i + 1],
                                                    kernel_size=kernel_size, padding=padding) for i in range(num_layers)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(dim_ls[i + 1]) for i in range(num_layers - 1)])

    def forward(self, f):
        f = F.relu(self.input(f))
        f = torch.unsqueeze(f, dim=1)
        for i in range(len(self.encode_conv) - 1):
            f = F.relu(self.encode_conv[i](f))
            f = self.batch_norm[i](f)
        f = self.encode_conv[-1](f)
        f_mean = torch.mean(f, dim=1)
        f_max, _ = torch.max(f, dim=1)
        f_mix, _ = torch.min(f, dim=1)
        f = f_mean + f_max + f_mix
        return f


"""
NULL
input == output
"""


class NULL(nn.Module):
    def __init__(self):
        """"""
        super(NULL, self).__init__()

    def forward(self, f):
        return f
