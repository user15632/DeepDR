"""
DNN, MHA_DNN
input: cell [batch_size, cell_dim] + drug [batch_size, drug_dim]
        or [batch_size, drug_dim, seq_len] or (encoded_graph.x + graph)
output: response [batch_size, 1] + cell_ft [batch_size, cell_dim] + drug_ft [batch_size, drug_dim]
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from ._MPG_mha import MultiHeadAttentionLayer

_cell_dim = 512
_drug_dim = 768
_dropout = 0.3
_num_heads = 8


class _MHA(nn.Module):
    def __init__(self, cell_dim: int = _cell_dim, drug_dim: int = _drug_dim, dropout: float = _dropout,
                 num_heads: int = _num_heads):
        """"""
        super(_MHA, self).__init__()
        self.input_query = nn.Linear(cell_dim, drug_dim)
        self.input_key_value = nn.Linear(drug_dim, drug_dim)
        self.attention = MultiHeadAttentionLayer(hid_dim=drug_dim, n_heads=num_heads, dropout=dropout)

    def forward(self, f, x):
        if type(x) == torch.Tensor:
            f = F.relu(self.input_query(f))
            query = torch.unsqueeze(f, 1)
            x = x.permute(0, 2, 1).contiguous()
            key_value = F.relu(self.input_key_value(x))
            f = self.attention(query, key_value, key_value)
        else:
            x, g = x
            f = F.relu(self.input_query(f))
            query = torch.unsqueeze(f, 1)
            x = to_dense_batch(x, g.batch)
            key_value = F.relu(self.input_key_value(x[0]))
            mask = torch.unsqueeze(torch.unsqueeze(x[1], 1), 1)
            f = self.attention(query, key_value, key_value, mask)
        f = torch.squeeze(f[0])
        return f


class DNN(nn.Module):
    def __init__(self, cell_dim: int = _cell_dim, drug_dim: int = _drug_dim, dropout: float = _dropout,
                 num_heads: int = _num_heads, pool: str = 'mean', concat: bool = False, hid_dim_ls: list = None):
        """hid_dim_ls"""
        super(DNN, self).__init__()
        assert pool in ['attention', 'mean', 'max', 'add', 'mix']
        if hid_dim_ls is not None:
            assert len(hid_dim_ls) >= 1
        else:
            hid_dim_ls = [512, 256, 128]
        self.pool = pool
        self.concat = concat
        if self.pool in ['attention', 'mix']:
            self.attention = _MHA(cell_dim=cell_dim, drug_dim=drug_dim, dropout=dropout, num_heads=num_heads)
        if not self.concat:
            dim_ls = [drug_dim] + hid_dim_ls + [1]
            self.input_cell = nn.Linear(cell_dim, dim_ls[0])
            self.input_drug = nn.Linear(drug_dim, dim_ls[0])
        else:
            dim_ls = [cell_dim + drug_dim] + hid_dim_ls + [1]
            self.input_cell = nn.Linear(cell_dim, cell_dim)
            self.input_drug = nn.Linear(drug_dim, drug_dim)
        self.encode_dnn = nn.ModuleList([nn.Linear(dim_ls[i], dim_ls[i + 1]) for i in range(len(dim_ls) - 2)])
        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(len(dim_ls) - 2)])
        self.output = nn.Linear(dim_ls[-2], dim_ls[-1])

    def forward(self, f, x):
        if type(x) == torch.Tensor:
            if len(x.shape) == 3:
                if self.pool == 'mean':
                    x = torch.mean(x, dim=2)
                elif self.pool == 'max':
                    x, _ = torch.max(x, dim=2)
                elif self.pool == 'add':
                    x = torch.sum(x, dim=2)
                elif self.pool == 'attention':
                    x = self.attention(f, x)
                else:
                    x_m, _ = torch.max(x, dim=2)
                    x = torch.mean(x, dim=2) + x_m + self.attention(f, x)
        else:
            x, g = x
            if self.pool == 'mean':
                x = global_mean_pool(x, g.batch)
            elif self.pool == 'max':
                x = global_max_pool(x, g.batch)
            elif self.pool == 'add':
                x = global_add_pool(x, g.batch)
            elif self.pool == 'attention':
                x = self.attention(f, (x, g))
            else:
                x = global_mean_pool(x, g.batch) + global_max_pool(x, g.batch) + self.attention(f, (x, g))

        cell_ft, drug_ft = f, x
        f = F.relu(self.input_cell(f))
        x = F.relu(self.input_drug(x))
        if not self.concat:
            f = f + x
        else:
            f = torch.cat((f, x), dim=1)
        for i in range(len(self.encode_dnn)):
            f = F.relu(self.encode_dnn[i](f))
            f = self.dropout[i](f)
        f = self.output(f)
        return f, cell_ft, drug_ft


class MHA_DNN(nn.Module):
    def __init__(self, cell_dim: int = _cell_dim, drug_dim: int = _drug_dim, dropout: float = _dropout,
                 num_heads: int = _num_heads, pool: bool = True, concat: bool = False, hid_dim_ls: list = None):
        """hid_dim_ls"""
        super(MHA_DNN, self).__init__()
        pool = 'mix' if pool else 'attention'
        self.encode_dnn = DNN(cell_dim, drug_dim, dropout, num_heads, pool, concat, hid_dim_ls)

    def forward(self, f, x):
        f, cell_ft, drug_ft = self.encode_dnn(f, x)
        return f, cell_ft, drug_ft
