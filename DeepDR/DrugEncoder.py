import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
import torch_geometric.nn.models as models
from torch_geometric.nn.conv import GCNConv, GATConv

from ._TrimNet_model import Block

_in_dim = 512
_drug_dim = 768
_dropout = 0.3

_num_embedding = 37
_kernel_size = 3
_padding = 1
_bidirectional = True

_x_num_embedding = 178
_edge_num_embedding = 18
_num_heads = 4
_MPG_dim = 768

"""
DNN
input: [batch_size, in_dim]
output: [batch_size, ft_dim]
"""


class DNN(nn.Module):
    def __init__(self, in_dim: int = _in_dim, ft_dim: int = _drug_dim, dropout: float = _dropout, hid_dim: int = 256,
                 num_layers: int = 2):
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


"""
CNN GRU LSTM
input: preprocessed_SMILES [batch_size, seq_len]
output: encoded_SMILES [batch_size, ft_dim, seq_len]
"""


class CNN(nn.Module):
    def __init__(self, num_embedding: int = _num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                 kernel_size: int = _kernel_size, padding: int = _padding, hid_dim: int = 256, num_layers: int = 2):
        """hid_dim, num_layers"""
        super(CNN, self).__init__()
        assert num_layers >= 1
        dim_ls = [embedding_dim] + [hid_dim] * (num_layers - 1) + [ft_dim]
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.encode_conv = nn.ModuleList([nn.Conv1d(in_channels=dim_ls[i], out_channels=dim_ls[i + 1],
                                                    kernel_size=kernel_size, padding=padding) for i in
                                          range(num_layers)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(dim_ls[i + 1]) for i in range(num_layers - 1)])

    def forward(self, f):
        f = self.embedding(f)
        f = f.permute(0, 2, 1).contiguous()
        for i in range(len(self.encode_conv) - 1):
            f = F.relu(self.encode_conv[i](f))
            f = self.batch_norm[i](f)
        f = self.encode_conv[-1](f)
        return f


class GRU(nn.Module):
    def __init__(self, num_embedding: int = _num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                 dropout: float = _dropout, bidirectional: bool = _bidirectional, num_layers: int = 2):
        """num_layers"""
        super(GRU, self).__init__()
        assert num_layers >= 1
        assert ft_dim % 2 == 0
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.encode_gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=ft_dim // (2 if bidirectional else 1),
                                       num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                                       batch_first=True)

    def forward(self, f):
        f = self.embedding(f)
        f, _ = self.encode_gru(f)
        f = f.permute(0, 2, 1).contiguous()
        return f


class LSTM(nn.Module):
    def __init__(self, num_embedding: int = _num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                 dropout: float = _dropout, bidirectional: bool = _bidirectional, num_layers: int = 2):
        """num_layers"""
        super(LSTM, self).__init__()
        assert num_layers >= 1
        assert ft_dim % 2 == 0
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.encode_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=ft_dim // (2 if bidirectional else 1),
                                         num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                                         batch_first=True)

    def forward(self, f):
        f = self.embedding(f)
        f, _ = self.encode_lstm(f)
        f = f.permute(0, 2, 1).contiguous()
        return f


"""
GCN GAT MPG
input: preprocessed_Graph
output: encoded_Graph.x + preprocessed_Graph
"""


class GCN(nn.Module):
    def __init__(self, x_num_embedding: int = _x_num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                 hid_dim: int = 384, num_layers: int = 2):
        """hid_dim, num_layers"""
        super(GCN, self).__init__()
        assert num_layers >= 2
        self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
        self.reset_parameters()
        self.input = GCNConv(embedding_dim, hid_dim)
        self.encode_gcn = nn.ModuleList([GCNConv(hid_dim, hid_dim) for _ in range(num_layers - 2)])
        self.output = GCNConv(hid_dim, ft_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

    def forward(self, g):
        x, edge_index = g.x, g.edge_index
        x = self.x_embedding(x).sum(1)
        x = F.relu(self.input(x, edge_index))
        for layer in self.encode_gcn:
            x = x + F.relu(layer(x, edge_index))
        x = self.output(x, edge_index)
        return x, g


class GAT(nn.Module):
    def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                 embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim, num_heads: int = _num_heads,
                 dropout: float = _dropout, hid_dim: int = 384, num_layers: int = 2):
        """hid_dim num_layers"""
        super(GAT, self).__init__()
        assert num_layers >= 2
        self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
        self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
        self.reset_parameters()
        self.input = GATConv(embedding_dim, hid_dim, heads=num_heads, concat=False, edge_dim=embedding_dim,
                             dropout=dropout)
        self.encode_gat = nn.ModuleList(
            [GATConv(hid_dim, hid_dim, heads=num_heads, concat=False, edge_dim=embedding_dim,
                     dropout=dropout) for _ in range(num_layers - 2)])
        self.output = GATConv(hid_dim, ft_dim, heads=num_heads, concat=False, edge_dim=embedding_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, g):
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
        x = self.x_embedding(x).sum(1)
        edge_attr = self.edge_embedding(edge_attr).sum(1)
        x = F.relu(self.input(x, edge_index, edge_attr))
        for layer in self.encode_gat:
            x = x + F.relu(layer(x, edge_index, edge_attr))
        x = self.output(x, edge_index, edge_attr)
        return x, g


class MPG(nn.Module):
    def __init__(self, MPG_dim: int = _MPG_dim, ft_dim: int = _drug_dim):
        """"""
        super(MPG, self).__init__()
        self.output = GCNConv(MPG_dim, ft_dim)

    def forward(self, g):
        x, edge_index = g.x, g.edge_index
        x = self.output(x, edge_index)
        return x, g


"""
TrimNet AttentiveFP
input: preprocessed_Graph
output: drug_ft [batch_size, ft_dim]
"""


class TrimNet(nn.Module):
    def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                 embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim, num_heads: int = _num_heads,
                 dropout: float = _dropout, hid_dim: int = 64, depth: int = 2):
        """hid_dim, depth"""
        super(TrimNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
        self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
        self.reset_parameters()
        self.lin0 = nn.Linear(embedding_dim, hid_dim)
        self.convs = nn.ModuleList([Block(hid_dim, embedding_dim, num_heads) for _ in range(depth)])
        self.set2set = Set2Set(hid_dim, processing_steps=3)
        self.out = nn.Sequential(nn.Linear(2 * hid_dim, 512),
                                 nn.LayerNorm(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=self.dropout),
                                 nn.Linear(512, ft_dim))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, g):
        x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
        x = self.x_embedding(x).sum(1)
        edge_attr = self.edge_embedding(edge_attr).sum(1)
        x = F.celu(self.lin0(x))
        for conv in self.convs:
            x = x + F.dropout(conv(x, edge_index, edge_attr), p=self.dropout, training=self.training)
        x = self.set2set(x, batch)
        x = self.out(F.dropout(x, p=self.dropout, training=self.training))
        return x


class AttentiveFP(nn.Module):
    def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                 embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim, dropout: float = _dropout, hid_dim: int = 384,
                 num_layers: int = 2, num_steps: int = 3):
        """hid_dim, num_layers, num_steps"""
        super(AttentiveFP, self).__init__()
        assert num_layers >= 1
        self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
        self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
        self.reset_parameters()
        self.encode_AttentiveFP = models.AttentiveFP(in_channels=embedding_dim, hidden_channels=hid_dim,
                                                     out_channels=ft_dim, edge_dim=embedding_dim, num_layers=num_layers,
                                                     num_timesteps=num_steps, dropout=dropout)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, g):
        x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
        x = self.x_embedding(x).sum(1)
        edge_attr = self.edge_embedding(edge_attr).sum(1)
        x = self.encode_AttentiveFP(x, edge_index, edge_attr, batch)
        return x


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
