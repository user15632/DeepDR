DeepDR.Data.Read
===========================



.. code-block:: python

    def Read(dataset: str, response: str, cell_ft: str or dict, drug_ft: str or dict,
    clean: bool = False, cell_dict_for_clean: dict or list = None)


It can be used to read cell-drug pairs from the csv file.

**PARAMETERS:**

* **dataset** *(str)* - 'CCLE', 'GDSC1', or 'GDSC2'.
* **response** *(str)* - 'ActArea', 'AUC', or 'IC50'.
* **cell_ft** *(str or dict)* - 'EXP', 'PES', 'MUT', 'CNV', or other dicts.
* **drug_ft** *(str or dict)* - 'ECFP', 'SMILESVec', 'SMILES', 'Graph', 'Image', or other dicts.

* **clean** *(bool, optional)* - Clear pairs that lack cell or drug features. *(default: False)*
* **cell_dict_for_clean** *(dict or list, optional)* - Basis for clearance. *(default: None)*


**OUTPUTS:**

* **dr_data** *(DrData)*
