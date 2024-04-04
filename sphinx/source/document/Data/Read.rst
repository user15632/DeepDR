DeepDR.Data.Read
===========================



.. code-block:: python

    def Read(pair_list: str = None, pair_list_csv_path: str = None, no_tag: bool = False, header=0,
    sep=',', index: list = None, clean: bool = False, cell_dict: str or dict or list = None,
    drug_dict: dict = None)


It can be used to read cell line-drug pairs from the csv file.

**PARAMETERS:**

* **pair_list** *(str, optional)* - The list of cell line-drug pairs. *(default: None)*
    The value could be 'CCLE_ActArea', 'CCLE_IC50', 'GDSC1_AUC', 'GDSC1_IC50', 'GDSC2_AUC', 'GDSC2_IC50' or None.

* **pair_list_csv_path** *(str, optional)* - The path of the csv file contains the cell line-drug pairs. *(default: None)*
    Parameter pair_list and pair_list_csv_path cannot be None at the same time.
    If their values are both not None, the latter has a higher priority when reading data.

* **no_tag** *(bool, optional)* - Whether the csv file contains no tags. *(default: False)*

* **header** *(optional)* - Parameter header of pandas.read_csv. *(default: None)*
* **sep** *(optional)* - Parameter sep of pandas.read_csv. *(default: ',')*

* **index** *(list, optional)* - The column index of cell line, drug and response in the csv file. *(default: None)*
    The default list is used when the value is None.

* **clean** *(bool, optional)* - Whether to clean up invalid data when reading data. *(default: False)*

* **cell_dict** *(str or dict or list)* - The dict of cell line features.
    The value could be 'CCLE', 'GDSC', your own dict or a list containing preceding ones.
    The key of the dict is the cell line name and the value is the cell line feature.

* **drug_dict** *(dict, optional)* - The dict of drug SMILES. *(default: None)*
    The value could be None or your own dict created by DeepDR.DataPreprocess.GetSMILESDict.
    The default dict is used when the value is None.
    The key of the dict is the drug name and the value is SMILES.

**OUTPUTS:**

* **pair_list** *(list)* - The list of cell line-drug pairs.
    Each element in the list is in the format [cell_name: str, drug_name: str, response: float].
