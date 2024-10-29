Data.DrRead
===========================

`Click here </en/latest/document/Data/DrReadCode.html>`_ to view source code.


.. code-block:: python

    class DrRead:

It contains functions ``DrRead.PairCSV``, ``DrRead.PairDef``, ``DrRead.FeatCell``, and ``DrRead.FeatDrug``.
They are respectively used for reading the response data from csv file, reading the response data integrated in the library,
preparing the cell feature, and preparing the drug feature.

DrRead.PairCSV
--------

.. code-block:: python

    def PairCSV(csv_path: str):


It can be used to read the response data from csv file.
First you need to follow the tutorial to prepare the required response data,
`click here </en/latest/document/Tutorials2.html#prepare-your-response-data>`_ for details.


**PARAMETERS:**

* **csv_path** *(str)* - The path of the response data. It is required to end in ``".csv"``.

**OUTPUTS:**

* **pair_ls** *(list)* - The cell-drug pairs. Each element in the list is a sub-list that contains three elements, which are the cell name, drug name, and drug response.


DrRead.PairDef
--------


.. code-block:: python

    def PairDef(dataset: str,
                response: str):


It can be used to read the response data integrated in the library.

For **dataset** *(str)* and **response** *(str)*, the following settings are available:

+-----------------------------------------+-------------------------+
| Parameter setting                       | Corresponding file      |
+=========================================+=========================+
| dataset="CCLE", response="ActArea"      | CCLE_ActArea.csv        |
+-----------------------------------------+-------------------------+
| dataset="CCLE", response="IC50"         | CCLE_IC50.csv           |
+-----------------------------------------+-------------------------+
| dataset="GDSC1", response="AUC"         | GDSC1_AUC.csv           |
+-----------------------------------------+-------------------------+
| dataset="GDSC1", response="IC50"        | GDSC1_IC50.csv          |
+-----------------------------------------+-------------------------+
| dataset="GDSC2", response="AUC"         | GDSC2_AUC.csv           |
+-----------------------------------------+-------------------------+
| dataset="GDSC2", response="IC50"        | GDSC2_IC50.csv          |
+-----------------------------------------+-------------------------+

**PARAMETERS:**

* **dataset** *(str)* - ``"CCLE"``, ``"GDSC1"``, or ``"GDSC2"``.
* **response** *(str)* - ``"ActArea"``, ``"AUC"``, or ``"IC50"``.

**OUTPUTS:**

* **pair_ls** *(list)* - The cell-drug pairs. Each element in the list is a sub-list that contains three elements, which are the cell name, drug name, and drug response.



DrRead.FeatCell
--------

.. code-block:: python

    def FeatCell(csv_path: str,
                 subset: bool,
                 subset_path: str = None,
                 save_feat_path: str = None,
                 save_gene_path: str = None):


It can be used to prepare the cell feature.
Each cell feature will be z-score standardized.
First you need to follow the tutorials to prepare the required gene subset and cell data,
`click here </en/latest/document/Tutorials2.html#prepare-your-gene-subset>`_ for details.

.. note::

    If there are nan values or missing genes in the cell data, the average of the non-nan values of the cell data will be filled in.


If you want to get genome-wide feature, use ``FeatCell(csv_path="feat.example.csv", subset=False)``.

If you want to use the default gene subset (containing 6,163 genes) to screen for cell feature,
use ``FeatCell(csv_path="feat.example.csv", subset=True)``.

**PARAMETERS:**

* **csv_path** *(str)* - The path of the cell data. It is required to end in ``".csv"``.

* **subset** *(bool)* - Whether to use the gene subset.
* **subset_path** *(str, optional)* - The path of the gene subset. It is required to end in ``".txt"``. If it is set to None, the default path will be used. *(default: None)*

* **save_feat_path** *(str, optional)* - Save path for **CellFeat**. It is required to end in ``".pkl"``. If it is set to None, the default path will be used. *(default: None)*
* **save_gene_path** *(str, optional)* - Save path for **GeneList**. It is required to end in ``".pkl"``. If it is set to None, the default path will be used. *(default: None)*


**OUTPUTS:**

* **CellFeat** *(dict)* - The key is the cell name and the value is the z-score standardized cell feature.
* **GeneList** *(list)* - Each element is a gene name, which corresponds to the cell feature.


DrRead.FeatDrug
--------


.. code-block:: python

    def FeatDrug(csv_path: str,
                 MPG_path: str,
                 save_SMILES_path: str = None,
                 save_MPG_path: str = None):


It can be used to prepare the drug feature.
First you need to follow the tutorials to prepare the required response data,
`click here </en/latest/document/Tutorials2.html#prepare-your-response-data>`_ for details.

If you don't need MPG feature, use ``FeatDrug(csv_path="pair.example.csv", MPG_path=None)``.

If you want to get MPG feature, use ``FeatDrug(csv_path="pair.example.csv", MPG_path="MolGNet.pt")``.
`Click here <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/MolGNet.pt>`_ to download ``MolGNet.pt``.

**PARAMETERS:**

* **csv_path** *(str)* - The path of the response data. It is required to end in ``".csv"``.
* **MPG_path** *(str)* - The path of the gene subset. It is required to end in ``".pt"``. If it is set to None, **MPG_dict** will be None.

* **save_SMILES_path** *(str, optional)* - Save path for **SMILES_dict**. It is required to end in ``".pkl"``. If it is set to None, the default path will be used. *(default: None)*
* **save_MPG_path** *(str, optional)* - Save path for **MPG_dict**. It is required to end in ``".pkl"``. If it is set to None, the default path will be used. *(default: None)*


**OUTPUTS:**

* **SMILES_dict** *(dict)* - The key is the drug name and the value is the SMILES string.
* **MPG_dict** *(dict)* - The key is the drug name and the value is the MPG feature. The value is None if **MPG_path** is set to None.
