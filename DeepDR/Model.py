import os
import torch
from torch import nn
import torch.nn.functional as F

from .CellEncoder import DNN as DNN_CE
from .CellEncoder import CNN as CNN_CE
from ._DAE_model import Encoder as DAE_CE
from .DrugEncoder import DNN, CNN, LSTM, GRU, GCN, GAT, MPG, AttentiveFP, TrimNet
from .FusionModule import DNN as DNN_FM
from .FusionModule import MHA_DNN as MHA_DNN_FM

from ._Training import device, Train, SetSeed
from ._Prediction import Predict

_cell_dim = 512
_dropout = 0.3


class _Model(nn.Module):
    def __init__(self, cell_encoder, drug_encoder, fusion_module):
        """"""
        super(_Model, self).__init__()

        if type(cell_encoder) == str:
            assert cell_encoder in ['DNN_EXP', 'CNN_EXP', 'DNN_GSVA', 'CNN_GSVA', 'DNN_CNV', 'CNN_CNV']
            if cell_encoder == 'DNN_EXP' or cell_encoder == 'DNN_CNV':
                self.Cell_Encoder = DNN_CE()
            elif cell_encoder == 'CNN_EXP' or cell_encoder == 'CNN_CNV':
                self.Cell_Encoder = CNN_CE()
            elif cell_encoder == 'DNN_GSVA':
                self.Cell_Encoder = DNN_CE(in_dim=1329)
            else:
                self.Cell_Encoder = CNN_CE(in_dim=1329)
        else:
            self.Cell_Encoder = cell_encoder

        if type(drug_encoder) == str:
            assert drug_encoder in ['DNN_ECFP', 'DNN_SMILESVec', 'CNN_SMILES', 'LSTM_SMILES', 'GRU_SMILES',
                                    'GCN_Graph', 'GAT_Graph', 'MPG_Graph', 'AttentiveFP_Graph', 'TrimNet_Graph']
            if drug_encoder == 'DNN_ECFP':
                self.Drug_Encoder = DNN()
            elif drug_encoder == 'DNN_SMILESVec':
                self.Drug_Encoder = DNN(in_dim=100)
            elif drug_encoder == 'CNN_SMILES':
                self.Drug_Encoder = CNN()
            elif drug_encoder == 'LSTM_SMILES':
                self.Drug_Encoder = LSTM()
            elif drug_encoder == 'GRU_SMILES':
                self.Drug_Encoder = GRU()
            elif drug_encoder == 'GCN_Graph':
                self.Drug_Encoder = GCN()
            elif drug_encoder == 'GAT_Graph':
                self.Drug_Encoder = GAT()
            elif drug_encoder == 'MPG_Graph':
                self.Drug_Encoder = MPG()
            elif drug_encoder == 'AttentiveFP_Graph':
                self.Drug_Encoder = AttentiveFP()
            else:
                self.Drug_Encoder = TrimNet()
        else:
            self.Drug_Encoder = drug_encoder

        if type(fusion_module) == str:
            if drug_encoder in ['DNN_ECFP', 'DNN_SMILESVec', 'AttentiveFP_Graph', 'TrimNet_Graph']:
                assert fusion_module == 'DNN'
                self.Fusion_Module = DNN_FM()
            else:
                assert fusion_module in ['DNN', 'MHA_DNN']
                if fusion_module == 'DNN':
                    self.Fusion_Module = DNN_FM()
                else:
                    self.Fusion_Module = MHA_DNN_FM()
        else:
            self.Fusion_Module = fusion_module

    def forward(self, cell_ft, drug_ft):
        cell_ft = self.Cell_Encoder(cell_ft)
        drug_ft = self.Drug_Encoder(drug_ft)
        res, cell_ft, drug_ft = self.Fusion_Module(cell_ft, drug_ft)
        return res, cell_ft, drug_ft


def _ModelTP(cell_encoder, drug_encoder, fusion_module, cell_encoder_path: str = None):
    """"""
    if cell_encoder_path is None:
        cell_encoder_path = os.path.join(os.path.split(__file__)[0], 'DefaultData/EncoderDAE.pt')

    if type(cell_encoder) == str:
        assert cell_encoder in ['DNN_EXP', 'CNN_EXP', 'DAE_EXP', 'DNN_GSVA', 'CNN_GSVA', 'DNN_CNV', 'CNN_CNV']
        if cell_encoder == 'DNN_EXP' or cell_encoder == 'DNN_CNV':
            cell_encoder = DNN_CE()
        elif cell_encoder == 'CNN_EXP' or cell_encoder == 'CNN_CNV':
            cell_encoder = CNN_CE()
        elif cell_encoder == 'DAE_EXP':
            cell_encoder = DAE_CE(input_dim=6163)
            cell_encoder.load_state_dict(torch.load(cell_encoder_path))
        elif cell_encoder == 'DNN_GSVA':
            cell_encoder = DNN_CE(in_dim=1329)
        else:
            cell_encoder = CNN_CE(in_dim=1329)

    if type(drug_encoder) == str:
        assert drug_encoder in ['DNN_ECFP', 'DNN_SMILESVec', 'CNN_SMILES', 'LSTM_SMILES', 'GRU_SMILES',
                                'GCN_Graph', 'GAT_Graph', 'MPG_Graph', 'AttentiveFP_Graph', 'TrimNet_Graph']
        if drug_encoder == 'DNN_ECFP':
            drug_encoder = DNN()
        elif drug_encoder == 'DNN_SMILESVec':
            drug_encoder = DNN(in_dim=100)
        elif drug_encoder == 'CNN_SMILES':
            drug_encoder = CNN()
        elif drug_encoder == 'LSTM_SMILES':
            drug_encoder = LSTM()
        elif drug_encoder == 'GRU_SMILES':
            drug_encoder = GRU()
        elif drug_encoder == 'GCN_Graph':
            drug_encoder = GCN()
        elif drug_encoder == 'GAT_Graph':
            drug_encoder = GAT()
        elif drug_encoder == 'MPG_Graph':
            drug_encoder = MPG()
        elif drug_encoder == 'AttentiveFP_Graph':
            drug_encoder = AttentiveFP()
        else:
            drug_encoder = TrimNet()

    if type(fusion_module) == str:
        if drug_encoder in ['DNN_ECFP', 'DNN_SMILESVec', 'AttentiveFP_Graph', 'TrimNet_Graph']:
            assert fusion_module == 'DNN'
            fusion_module = DNN_FM()
        else:
            assert fusion_module in ['DNN', 'MHA_DNN']
            if fusion_module == 'DNN':
                fusion_module = DNN_FM()
            else:
                fusion_module = MHA_DNN_FM()

    return cell_encoder, drug_encoder, fusion_module


def MDL(cell_encoder, drug_encoder, fusion_module, integrate: bool = False, cell_encoder_path: str = None):
    if integrate:
        model = _Model(cell_encoder, drug_encoder, fusion_module)
    else:
        model = _ModelTP(cell_encoder, drug_encoder, fusion_module, cell_encoder_path)
    return model


class SDL(nn.Module):
    def __init__(self, cell_encoder, cell_dim: int = _cell_dim, dropout: float = _dropout, hid_dim_ls: list = None):
        """hid_dim_ls"""
        super(SDL, self).__init__()

        if type(cell_encoder) == str:
            assert cell_encoder in ['DNN_EXP', 'CNN_EXP', 'DNN_GSVA', 'CNN_GSVA', 'DNN_CNV', 'CNN_CNV']
            if cell_encoder == 'DNN_EXP' or cell_encoder == 'DNN_CNV':
                self.Cell_Encoder = DNN_CE()
            elif cell_encoder == 'CNN_EXP' or cell_encoder == 'CNN_CNV':
                self.Cell_Encoder = CNN_CE()
            elif cell_encoder == 'DNN_GSVA':
                self.Cell_Encoder = DNN_CE(in_dim=1329)
            else:
                self.Cell_Encoder = CNN_CE(in_dim=1329)
        else:
            self.Cell_Encoder = cell_encoder

        if hid_dim_ls is not None:
            assert len(hid_dim_ls) >= 1
        else:
            hid_dim_ls = [512, 256, 128]

        dim_ls = [cell_dim] + hid_dim_ls + [1]
        self.encode_dnn = nn.ModuleList([nn.Linear(dim_ls[i], dim_ls[i + 1]) for i in range(len(dim_ls) - 2)])
        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(len(dim_ls) - 2)])
        self.output = nn.Linear(dim_ls[-2], dim_ls[-1])

    def forward(self, cell_ft):
        cell_ft = self.Cell_Encoder(cell_ft)
        f = cell_ft
        for i in range(len(self.encode_dnn)):
            f = F.relu(self.encode_dnn[i](f))
            f = self.dropout[i](f)
        res = self.output(f)
        return res, cell_ft
