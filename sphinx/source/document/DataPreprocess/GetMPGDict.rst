DeepDR.DataPreprocess.GetMPGDict
===========================



.. code-block:: python

    def GetMPGDict(SMILES_dict: dict, save: bool = True, save_path_MPG_dict: str = None)


It can be used to create MPG_dict.

**PARAMETERS:**

* **SMILES_dict** *(dict)* - The dict of drug SMILES.
    The key of the dict is the drug name and the value is SMILES.

* **save** *(bool, optional)* - Whether to save MPG_dict. *(default: True)*

* **save_path_SMILES_dict** *(str, optional)* - The save path of MPG_dict. *(default: None)*
    The default path is used when the value is None.

**OUTPUTS:**

* **MPG_dict** *(dict)* - The dict of drug MPG features.
    The key of the dict is the drug name and the value is the MPG feature.
