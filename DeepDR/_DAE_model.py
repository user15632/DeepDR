import torch
import random
import numpy as np
from abc import ABC
import torch.nn.functional as F
from torch.utils.data import Dataset
# from optuna.samplers import TPESampler
import torch.nn as nn
from copy import deepcopy

ldim = 512
hdim = [2048, 1536, 1024]


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=ldim, h_dims=None, drop_out_rate=0.3):
        super(Encoder, self).__init__()
        if h_dims is None:
            h_dims = hdim
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(drop_out_rate))
            )
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input):
        result = self.encoder(input)
        embedding = F.relu(self.bottleneck(result))
        return embedding


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim=ldim, h_dims=None, drop_out_rate=0.3):
        super(Decoder, self).__init__()
        if h_dims is None:
            h_dims = hdim
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(drop_out_rate))
            )
        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    def forward(self, embedding):
        result = self.decoder_input(embedding)
        result = self.decoder(result)
        output = self.decoder_output(result)
        return output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # sampler = TPESampler(seed=seed)
    # return sampler


class CollateFn:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        batch_data = torch.stack(batch)
        return batch_data


class MyDataSet(Dataset, ABC):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        data = self._data[idx]
        return data

    def __len__(self):
        return len(self._data)
