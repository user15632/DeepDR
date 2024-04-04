import torch
import numpy as np
from rdkit.Chem import AllChem

from ._MPG_util import Self_loop, Add_seg_id
from ._MPG_loader import mol_to_graph_data_obj_complex

_char_ls = ["7", "6", "o", "]", "3", "s", "(", "-", "S", "/", "B", "4", "[", ")", "#", "I", "l", "O", "H", "c", "t", "1", "@",
            "=", "n", "P", "8", "C", "2", "F", "5", "r", "N", "+", "\\", ".", " "]
_max_len = 230

_Self_loop = Self_loop()
_Add_seg_id = Add_seg_id()


def _GetEcfp(smiles: str, radius: int = 2, nBits: int = 512):
    """"""
    mol = AllChem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    ECFP = np.zeros((nBits,), dtype=int)
    on_bits = list(fp.GetOnBits())
    ECFP[on_bits] = 1
    return ECFP.tolist()


def PreEcfp(smiles: str, radius: int = 2, nBits: int = 512):
    """"""
    smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
    return torch.tensor(_GetEcfp(smiles, radius, nBits), dtype=torch.float32)


def _PadSmiles(smiles: str, max_len: int = None, right: bool = True):
    """"""
    if max_len is None:
        max_len = _max_len
    assert max_len >= len(smiles)
    if right:
        return smiles + " " * (max_len - len(smiles))
    else:
        return " " * (max_len - len(smiles)) + smiles


def PreSmiles(smiles: str, max_len: int = None, char_dict: dict = None, right: bool = True):
    """"""
    if char_dict is None:
        char_dict = dict(zip(_char_ls, [i for i in range(len(_char_ls))]))
    smiles = _PadSmiles(AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True), max_len, right)
    return torch.tensor([char_dict[c] for c in smiles], dtype=torch.int)


def PreGraph(smiles: str):
    """"""
    smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
    return _Add_seg_id(_Self_loop(mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(smiles))))
