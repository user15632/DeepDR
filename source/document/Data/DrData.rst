Data.DrData
===========================

`Click here </document/Data/DrDataCode.html>`_ to view source code.

.. code-block:: python

    class DrData(pair_ls: list,
                 cell_ft: str or dict,
                 drug_ft: str or dict,
                 smiles_dict: dict = None,
                 mpg_dict: dict = None):


It stores all the data needed for drug response prediction.
You can remove pairs lacking cell or drug data,
and split the response data into training, validation, and test set
using ``.clean`` and ``.split``.


**PARAMETERS:**

* **pair_ls** *(list)* - The cell-drug pairs. Each element in the list is a sub-list that contains three elements, which are the cell name, drug name, and drug response. You can build it yourself or get it through ``Data.DrRead.PairCSV`` or ``Data.DrRead.PairDef``.

* **cell_ft** *(str or dict)* - ``"EXP"``, ``"PES"``, ``"MUT"``, ``"CNV"``, **CellFeat** got by ``Data.DrRead.FeatCell``, or your own dict, where the key is the cell name and the value is the feature vector, e.g. `VAE_dict.pkl <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/VAE_dict.pkl>`_.
* **drug_ft** *(str or dict)* - ``"ECFP"``, ``"SMILES"``, ``"Graph"``, ``"Image"``, or your own dict, where the key is the drug name and the value is the feature vector, e.g. `SMILESVec_dict.pkl <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/SMILESVec_dict.pkl>`_.

* **smiles_dict** *(dict, optional)* - **SMILES_dict** got by ``Data.DrRead.FeatDrug``. *(default: None)*
* **mpg_dict** *(dict, optional)* - **MPG_dict** got by ``Data.DrRead.FeatDrug``. *(default: None)*


self.clean
--------

.. code-block:: python

    def clean(self, cell_ft_ls: list = None):


It can be used to remove pairs lacking cell or drug data.

* **cell_ft_ls** *(list, optional)* - Each element should have the same form as **cell_ft**. *(default: None)*


self.split
--------


.. code-block:: python

    def split(self, mode: str,
              fold: int,
              ratio: list,
              seed: int,
              save: bool = True,
              save_path: str = None):


It can be used to split the response data into training, validation, and test set.

**PARAMETERS:**

* **mode** *(str)* - The splitting mode. ``"common"``, ``"cell_out"``, ``"drug_out"``, ``"strict"`` are available.
* **fold** *(int)* - The number of folds for k-fold cross-validation. It should greater or equal to 1. Setting ``fold=1`` will not use k-fold cross-validation.

* **ratio** *(list)* - The splitting ratio. If ``fold=1``, it should be a list containing 3 floats, respectively correspond to the ratio of training set, validation set, and test set. If ``fold>1``, it should be a list containing 2 floats, respectively correspond to the ratio of non-test set and test set.

* **seed** *(int)* - The random seed.

* **save** *(bool, optional)* - Whether to save the return value. *(default: True)*

* **save_path** *(str, optional)* - Save path for the return value. It is required to end in ``".pkl"``. If it is set to None, the default path will be used. *(default: None)*


**OUTPUTS:**

* **train_dr_data_ls** *(list)* - The segmented training sets.
* **val_dr_data_ls** *(list)* - The segmented validation sets.
* **test_dr_data** *(DrData)* - The segmented test set.
