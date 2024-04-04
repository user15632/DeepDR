DeepDR.Data.Split
===========================



.. code-block:: python

    def Split(pair_list: list, mode: str = 'default', k: int = 1, seed: int = 1, ratio: list = None,
    save: bool = True, save_path: str = None)


**PARAMETERS:**

* **pair_list** *(list)* - The list of cell line-drug pairs.
    Each element in the list is in the format [cell_name: str, drug_name: str, response: float].

* **mode** *(str)* - The split mode.
    The value could be 'default', 'cell_strict_split' or 'drug_strict_split'.

* **k** *(int)* - The number of folds of cross-validation. *(default: 1)*

* **seed** *(int)* - The random seed. *(default: 1)*

* **ratio** *(list)* - The ratio of training set, validation set and test set. *(default: None)*
    [0.8, 0.1, 0.1] is used when it is None and k == 1.
    [0.85, 0.15] is used when it is None and k >= 1.

* **save** *(bool)* - Save or not. *(default: True)*

* **save_path** *(int)* - Save path. *(default: None)*
    The default dict is used when the value is None.


**OUTPUTS:**

* **train_pair** *(list)* - The cell line-drug pairs in the training set.
* **val_pair** *(list)* - The cell line-drug pairs in the validation set.
* **test_pair** *(list)* - The cell line-drug pairs in the test set.
