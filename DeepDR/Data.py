import os
import time
import torch
import random
import joblib
import pandas as pd
from abc import ABC
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .DrugPreprocess import PreEcfp, PreSmiles, PreGraph


def NormalizeName(string: str):
    """"""
    lt = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    string = string.upper()
    std_string = ''
    for char in string:
        if char in lt:
            std_string += char
    return std_string


def _Clean(pair_list: list, cell_dict: str or dict, drug_dict: dict = None):
    """"""
    if type(cell_dict) == str:
        assert cell_dict in ['GDSC_EXP', 'CCLE_EXP', 'GDSC_GSVA', 'CCLE_GSVA', 'GDSC_CNV', 'CCLE_CNV']
        cell_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/' + cell_dict + '_dict.pkl'))
    if drug_dict is None:
        drug_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/SMILES_dict.pkl'))
    pair_list_cleaned = []
    for each in pair_list:
        if each[0] in cell_dict and each[1] in drug_dict:
            pair_list_cleaned.append(each)
    print('Number of original pairs: ' + str(len(pair_list)))
    print('Number of efficient pairs: ' + str(len(pair_list_cleaned)))
    return pair_list_cleaned


def Read(pair_list: str = None, pair_list_csv_path: str = None, no_tag: bool = False, header=0, sep=',',
         index: list = None, clean: bool = False, cell_dict: str or dict or list = None, drug_dict: dict = None):
    """"""
    assert pair_list in ['CCLE_ActArea', 'CCLE_IC50', 'GDSC1_AUC', 'GDSC1_IC50', 'GDSC2_AUC', 'GDSC2_IC50', None]
    assert (pair_list is None and pair_list_csv_path is None) is False
    if pair_list is not None and pair_list_csv_path is None:
        pair_list_csv_path = os.path.join(os.path.split(__file__)[0], 'DefaultData/' + pair_list + '.csv')
    if index is None:
        index = [0, 1] if no_tag else [0, 1, 2]
    if no_tag:
        assert len(index) == 2
    else:
        assert len(index) == 3

    print('Start reading!')
    csv = pd.read_csv(pair_list_csv_path, header=header, sep=sep, dtype=str)
    Cell = [NormalizeName(each) for each in list(csv.iloc[:, index[0]])]
    Drug = list(csv.iloc[:, index[1]])
    if no_tag:
        Tag = [0.0] * len(Cell)
    else:
        Tag = [float(_) for _ in list(csv.iloc[:, index[2]])]
    pair_list = [(Cell[i], Drug[i], Tag[i]) for i in range(len(Cell))]
    if clean:
        if type(cell_dict) == list:
            for each in cell_dict:
                pair_list = _Clean(pair_list, each, drug_dict)
        else:
            pair_list = _Clean(pair_list, cell_dict, drug_dict)
    print('Reading completed!')
    return pair_list


def Split(pair_list: list, mode: str = 'default', k: int = 1, seed: int = 1, ratio: list = None, save: bool = True,
          save_path: str = None):
    """"""
    if k > 1:
        train_pair_k_folds, val_pair_k_folds, test_pair = _SplitKFolds(pair_list, mode, k, seed, ratio, save, save_path)
        return train_pair_k_folds, val_pair_k_folds, test_pair
    else:
        assert mode in ['default', 'cell_strict_split', 'drug_strict_split']
        if ratio is None:
            ratio = [0.8, 0.1, 0.1]
        assert (sum(ratio) - 1) < 1e-5
        assert len(ratio) == 3
        if save is True and save_path is None:
            t = time.localtime()
            save_path = 'SplitPair_list_{}_{}_{}_{}_{}_{}.pkl'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

        print('Start splitting!')
        if mode == 'default':
            random.seed(seed)
            random.shuffle(pair_list)
            train_pair = pair_list[:int(len(pair_list) * ratio[0])]
            val_pair = pair_list[int(len(pair_list) * ratio[0]): int(len(pair_list) * (ratio[0] + ratio[1]))]
            test_pair = pair_list[int(len(pair_list) * (ratio[0] + ratio[1])):]
        elif mode == 'cell_strict_split':
            cell_list = sorted(list(set([each[0] for each in pair_list])))
            random.seed(seed)
            random.shuffle(cell_list)
            train_cell = cell_list[:int(len(cell_list) * ratio[0])]
            val_cell = cell_list[int(len(cell_list) * ratio[0]): int(len(cell_list) * (ratio[0] + ratio[1]))]
            train_pair = []
            val_pair = []
            test_pair = []
            for each in pair_list:
                if each[0] in train_cell:
                    train_pair.append(each)
                elif each[0] in val_cell:
                    val_pair.append(each)
                else:
                    test_pair.append(each)
        else:
            drug_list = sorted(list(set([each[1] for each in pair_list])))
            random.seed(seed)
            random.shuffle(drug_list)
            train_drug = drug_list[:int(len(drug_list) * ratio[0])]
            val_drug = drug_list[int(len(drug_list) * ratio[0]): int(len(drug_list) * (ratio[0] + ratio[1]))]
            train_pair = []
            val_pair = []
            test_pair = []
            for each in pair_list:
                if each[1] in train_drug:
                    train_pair.append(each)
                elif each[1] in val_drug:
                    val_pair.append(each)
                else:
                    test_pair.append(each)
        if save:
            joblib.dump((train_pair, val_pair, test_pair), save_path)
        print('Splitting completed!')
        return train_pair, val_pair, test_pair


def _SplitKFolds(pair_list: list, mode: str = 'default', k: int = 5, seed: int = 1, ratio: list = None,
                 save: bool = True, save_path: str = None):
    """"""
    assert mode in ['default', 'cell_strict_split', 'drug_strict_split']
    if ratio is None:
        ratio = [0.85, 0.15]
    assert (sum(ratio) - 1) < 1e-5
    assert len(ratio) == 2
    if save is True and save_path is None:
        t = time.localtime()
        save_path = 'SplitKFoldsPair_list_{}_{}_{}_{}_{}_{}.pkl'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

    def get_k_folds(ls):
        train = ls[:int(len(ls) * ratio[0])]
        test = ls[int(len(ls) * ratio[0]):]
        train_k_folds = []
        val_k_folds = []
        for i in range(k):
            train_k = train[:i * len(train) // k] + train[(i + 1) * len(train) // k:]
            val_k = train[i * len(train) // k: (i + 1) * len(train) // k]
            train_k_folds.append(train_k)
            val_k_folds.append(val_k)
        return train_k_folds, val_k_folds, test

    def get_pair(ls, index):
        pair = []
        for _each in pair_list:
            if _each[index] in ls:
                pair.append(_each)
        return pair

    print('Start splitting!')
    if mode == 'default':
        random.seed(seed)
        random.shuffle(pair_list)
        train_pair_k_folds, val_pair_k_folds, test_pair = get_k_folds(pair_list)
    elif mode == 'cell_strict_split':
        cell_list = sorted(list(set([each[0] for each in pair_list])))
        random.seed(seed)
        random.shuffle(cell_list)
        train_cell_k_folds, val_cell_k_folds, test_cell = get_k_folds(cell_list)
        train_pair_k_folds = [get_pair(each, 0) for each in train_cell_k_folds]
        val_pair_k_folds = [get_pair(each, 0) for each in val_cell_k_folds]
        test_pair = get_pair(test_cell, 0)
    else:
        drug_list = sorted(list(set([each[1] for each in pair_list])))
        random.seed(seed)
        random.shuffle(drug_list)
        train_drug_k_folds, val_drug_k_folds, test_drug = get_k_folds(drug_list)
        train_pair_k_folds = [get_pair(each, 1) for each in train_drug_k_folds]
        val_pair_k_folds = [get_pair(each, 1) for each in val_drug_k_folds]
        test_pair = get_pair(test_drug, 1)
    if save:
        joblib.dump((train_pair_k_folds, val_pair_k_folds, test_pair), save_path)
    print('Splitting completed!')
    return train_pair_k_folds, val_pair_k_folds, test_pair


class DrDataset(Dataset, ABC):
    """"""
    def __init__(self, pair_list: list, drug_encoding: str, cell_dict: str or dict, drug_dict: dict = None,
                 radius: int = 2, nBits: int = 512, max_len: int = None, char_dict: dict = None, right: bool = True,
                 MPG_dict: dict = None):
        super().__init__()
        self._pair_list = pair_list
        self._drug_encoding = drug_encoding
        self._cell_dict = cell_dict
        self._drug_dict = drug_dict
        self._radius = radius
        self._nBits = nBits
        self._max_len = max_len
        self._char_dict = char_dict
        self._right = right
        self._MPG_dict = MPG_dict
        self._data = DrDataset.preprocess(self)

    def __getitem__(self, idx):
        data = self._data[idx]
        return data

    def __len__(self):
        return len(self._data)

    def preprocess(self):
        assert self._drug_encoding in ['ECFP', 'SMILESVec', 'MPGVec', 'SMILES', 'Graph', 'MPGGraph']
        if type(self._cell_dict) == str:
            assert self._cell_dict in ['GDSC_EXP', 'CCLE_EXP', 'GDSC_GSVA', 'CCLE_GSVA', 'GDSC_CNV', 'CCLE_CNV']
            self._cell_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/' + self._cell_dict + '_dict.pkl'))
        if self._drug_dict is None:
            if self._drug_encoding == 'SMILESVec':
                self._drug_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/SMILESVec_dict.pkl'))
            elif self._drug_encoding == 'MPGVec':
                self._drug_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/MPGVec_dict.pkl'))
            else:
                self._drug_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/SMILES_dict.pkl'))
        if self._drug_encoding == 'MPGGraph' and self._MPG_dict is None:
            self._MPG_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/MPG_dict.pkl'))
        self._pair_list = _Clean(self._pair_list, self._cell_dict, self._drug_dict)

        data = []
        for each_pair in self._pair_list:
            if type(self._cell_dict[each_pair[0]]) == torch.Tensor:
                cell_ft = self._cell_dict[each_pair[0]]
            else:
                cell_ft = torch.tensor(self._cell_dict[each_pair[0]], dtype=torch.float32)
            cell_ft = (cell_ft - cell_ft.mean()) / cell_ft.std(dim=0)
            if self._drug_encoding == 'ECFP':
                drug_ft = PreEcfp(self._drug_dict[each_pair[1]], self._radius, self._nBits)
            elif self._drug_encoding == 'SMILESVec' or self._drug_encoding == 'MPGVec':
                if type(self._drug_dict[each_pair[1]]) == torch.Tensor:
                    drug_ft = self._drug_dict[each_pair[1]]
                else:
                    drug_ft = torch.tensor(self._drug_dict[each_pair[1]], dtype=torch.float32)
            elif self._drug_encoding == 'SMILES':
                drug_ft = PreSmiles(self._drug_dict[each_pair[1]], self._max_len, self._char_dict, self._right)
            elif self._drug_encoding == 'Graph':
                drug_ft = PreGraph(self._drug_dict[each_pair[1]])
            else:
                g = PreGraph(self._drug_dict[each_pair[1]])
                x, edge_index, edge_attr = self._MPG_dict[each_pair[1]], g.edge_index, g.edge_attr
                drug_ft = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            data.append(Data(cell_ft=cell_ft, drug_ft=drug_ft,
                             response=torch.tensor([each_pair[2]], dtype=torch.float32),
                             cell_name=each_pair[0], drug_name=each_pair[1]))
        return data


class DrCollate:
    """"""
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        cell_ft = torch.stack([g.cell_ft for g in batch])
        if type(batch[0].drug_ft) == torch.Tensor:
            drug_ft = torch.stack([g.drug_ft for g in batch])
        else:
            drug_ft = Batch.from_data_list([g.drug_ft for g in batch], self.follow_batch, self.exclude_keys)
        response = torch.stack([g.response for g in batch])
        return cell_ft, drug_ft, response, [g.cell_name for g in batch], [g.drug_name for g in batch]


def DrDataLoader(dataset, batch_size: int = 128, shuffle: bool = True, follow_batch=None, exclude_keys=None):
    """"""
    collate = DrCollate(follow_batch, exclude_keys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader
