DeepDR.Data.DrDataset
===========================



.. code-block:: python

    class DrDataset(pair_list: list, drug_encoding: str, cell_dict: str or dict, drug_dict: dict = None,
    radius: int = 2, nBits: int = 512, max_len: int = None, char_dict: dict = None, right: bool = True,
    MPG_dict: dict = None)


**PARAMETERS:**

* **pair_list** *(list)* - The list of cell line-drug pairs.
    Each element in the list is in the format [cell_name: str, drug_name: str, response: float].

* **drug_encoding** *(str)* - The drug encoding mode.
    The value could be 'ECFP', 'SMILESVec', 'MPGVec', 'SMILES', 'Graph' or 'MPGGraph'.

* **cell_dict** *(str or dict)* - The dict of cell line expression profiles.
    The value could be 'CCLE', 'GDSC' or your own dict.
    The key of the dict is the cell line name and the value is the gene expression profile.

* **drug_dict** *(dict, optional)* - The dict of drug SMILES. *(default: None)*
    The value could be None or your own dict created by DeepDR.DataPreprocess.GetSMILESDict.
    The default dict is used when the value is None.
    The key of the dict is the drug name and the value is SMILES.

* **radius** *(int, optional)* - The radius of ECFP. *(default: 2)*
* **nBits** *(int, optional)* - The nBits of ECFP. *(default: 512)*

* **max_len** *(int, optional)* - The max length of the sequence. *(default: None)*
    The default value is used when it is None.
* **char_dict** *(dict, optional)* - The character-integer mapping dict of the sequence. *(default: None)*
    The value could be None or your own dict.
    The default dict is used when the value is None.
    The key of the dict is the character and the value is the corresponding integer.
* **right** *(bool, optional)* - The padding direction of the sequence. *(default: True)*
    If the value is True, padding to the right.
    If the value is False, padding to the left.

* **MPG_dict** *(dict, optional)* - The dict of drug MPG features. *(default: None)*
    The value could be None or your own dict created by DeepDR.DataPreprocess.GetMPGDict.
    The default dict is used when the value is None.
    The key of the dict is the drug name and the value is the MPG feature.
