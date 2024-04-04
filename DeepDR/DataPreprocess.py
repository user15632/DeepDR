import os
import time
import torch
import joblib
import pubchempy as pcp

from .DrugPreprocess import PreGraph
from ._MPG_model import MolGNet


def GetSMILESDict(pair_list: list, save: bool = True, save_path_SMILES_dict: str = None):
    """"""
    if save_path_SMILES_dict is None:
        t = time.localtime()
        save_path_SMILES_dict = 'SMILES_dict_{}_{}_{}_{}_{}_{}.pkl'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    print('Retrieving SMILES strings...')
    drug_list = sorted(list(set([each[1] for each in pair_list])))
    SMILES_dict_0 = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/SMILES_dict.pkl'))
    SMILES_dict = dict()
    SMILES_not_found = []
    for each in drug_list:
        if each in SMILES_dict_0:
            SMILES_dict[each] = SMILES_dict_0[each]
        else:
            try:
                _ = pcp.get_compounds(each, 'name')
                SMILES_dict[each] = _[0].isomeric_smiles
            except:
                SMILES_not_found.append(each)
    if save:
        joblib.dump(SMILES_dict, save_path_SMILES_dict)
    print('Total: {}  Successful: {}'.format(len(drug_list), len(drug_list) - len(SMILES_not_found)))
    return SMILES_dict


def GetMPGDict(SMILES_dict: dict, save: bool = True, save_path_MPG_dict: str = None):
    """"""
    if save_path_MPG_dict is None:
        t = time.localtime()
        save_path_MPG_dict = 'MPG_dict_{}_{}_{}_{}_{}_{}.pkl'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
    MPG_dict_0 = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/MPG_dict.pkl'))
    MPG_dict = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0)
    gnn.load_state_dict(torch.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/MolGNet.pt')))
    gnn = gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        for each in SMILES_dict:
            if each in MPG_dict_0:
                MPG_dict[each] = MPG_dict_0[each]
            else:
                graph = PreGraph(SMILES_dict[each]).to(device)
                MPG_dict[each] = gnn(graph).cpu()
    if save:
        joblib.dump(MPG_dict, save_path_MPG_dict)
    return MPG_dict


def GetGeneList():
    """"""
    f = open(os.path.join(os.path.split(__file__)[0], 'DefaultData/key.genes.txt'), encoding='gbk')
    Gene_list = []
    for each_row in f:
        Gene_list.append(each_row.strip())
    return Gene_list
