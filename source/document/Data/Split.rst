DeepDR.Data.Split
===========================



.. code-block:: python

    def Split(dr_data: DrData, mode: str, ratio: list, seed: int = 1, save: bool = True,
    save_path: str = None)


**PARAMETERS:**

* **dr_data** *(DrData)* - The drug response data.

* **mode** *(str)* - The split mode.
    The value could be 'common', 'cell_out', 'drug_out', or 'strict'.

* **ratio** *(list)* - The ratio of training set, validation set (and test set).

* **seed** *(int)* - The random seed. *(default: 1)*

* **save** *(bool)* - Save or not. *(default: True)*

* **save_path** *(int)* - Save path. *(default: None)*
    The default dict is used when the value is None.


**OUTPUTS:**

    When the length of ratio list is 2:
* **train_dr_data** *(DrData)*
* **val_dr_data** *(DrData)*
    When the length of ratio list is 3:
* **train_dr_data** *(DrData)*
* **val_dr_data** *(DrData)*
* **test_dr_data** *(DrData)*
