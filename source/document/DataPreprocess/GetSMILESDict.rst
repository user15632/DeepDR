DeepDR.DataPreprocess.GetSMILESDict
===========================



.. code-block:: python

    def GetSMILESDict(pair_list: list, save: bool = True, save_path_SMILES_dict: str = None)


It can be used to create SMILES_dict.

**PARAMETERS:**

* **pair_list** *(list)* - The list of cell line-drug pairs.
    Each element in the list is in the format [cell_name: str, drug_name: str, response: float].

* **save** *(bool, optional)* - Whether to save SMILES_dict. *(default: True)*

* **save_path_SMILES_dict** *(str, optional)* - The save path of SMILES_dict. *(default: None)*
    The default path is used when the value is None.

**OUTPUTS:**

* **SMILES_dict** *(dict)* - The dict of drug SMILES.
    The key of the dict is the drug name and the value is SMILES.
