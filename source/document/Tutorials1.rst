How to build and clean data
==================================

The following are detailed tutorials on how to build and clean data.


--------

Use integrated response data
--------

.. note::

   If you use your own response data, ignore this section.


Use ``Data.DrData`` as follows:

.. code-block:: python

    DrData(pair_ls=...,
           cell_ft=...,
           drug_ft=...)



~~~~~~~~

**Section 1.1: set pair_ls**

For **pair_ls**, you just need to set it to the return value of ``Data.DrRead.PairDef``,
`click here </document/Data/DrRead.html#drread-pairdef>`_ for details.

~~~~~~~~

**Section 1.2: set cell_ft**


For **cell_ft** *(str or dict)*, the following settings are available:

+------------------------+-------------------+
| Parameter setting      | Corresponding file|
+========================+===================+
| cell_ft="EXP"          | GDSC_EXP.pkl      |
+------------------------+-------------------+
| cell_ft="PES"          | GDSC_PES.pkl      |
+------------------------+-------------------+
| cell_ft="MUT"          | GDSC_MUT.pkl      |
+------------------------+-------------------+
| cell_ft="CNV"          | GDSC_CNV.pkl      |
+------------------------+-------------------+
| cell_ft=cell_dict      | None              |
+------------------------+-------------------+

In the last row of the table, ``cell_dict`` could be **CellFeat** got by ``Data.DrRead.FeatCell``, or your own dict, where the key is the cell name and the value is the feature vector, e.g. `VAE_dict.pkl <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/VAE_dict.pkl>`_.

If you set **cell_ft** to ``"EXP"``, ``"PES"``, ``"MUT"``, ``"CNV"``,
pairs lacking cell data will be removed based on the corresponding integrated file when you use ``.clean``.
If you set **cell_ft** to ``cell_dict``,
pairs lacking cell data will be removed based on the dict you set when you use ``.clean``.


~~~~~~~~

**Section 1.3: set drug_ft**

For **drug_ft** *(str or dict)*, the following settings are available:

+-------------------+-------------------+
| Parameter setting | Corresponding file|
+===================+===================+
| drug_ft="ECFP"    | SMILES_dict.pkl   |
+-------------------+-------------------+
| drug_ft="SMILES"  | SMILES_dict.pkl   |
+-------------------+-------------------+
| drug_ft="Graph"   | SMILES_dict.pkl   |
+-------------------+-------------------+
| drug_ft="Image"   | SMILES_dict.pkl   |
+-------------------+-------------------+
| drug_ft=drug_dict | None              |
+-------------------+-------------------+

In the last row of the table, ``drug_dict`` could be your own dict, where the key is the drug name and the value is the feature vector, e.g. `SMILESVec_dict.pkl <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/SMILESVec_dict.pkl>`_.

If you set **drug_ft** to ``"ECFP"``, ``"SMILES"``, ``"Graph"``, ``"Image"``,
pairs lacking drug data will be removed based on the corresponding integrated file when you use ``.clean``.
If you set **drug_ft** to ``drug_dict``,
pairs lacking drug data will be removed based on the dict you set when you use ``.clean``.


--------

Use your own response data
--------


.. note::

   If you use integrated response data, ignore this section.


Use ``Data.DrData`` as follows:

.. code-block:: python

    DrData(pair_ls=...,
           cell_ft=...,
           drug_ft=...,
           smiles_dict=...,
           mpg_dict=...)



~~~~~~~~

**Section 2.1: set pair_ls**

For **pair_ls**, you just need to set it to the return value of ``Data.DrRead.PairCSV``,
`click here </document/Data/DrRead.html#drread-paircsv>`_ for details.


~~~~~~~~

**Section 2.2: set cell_ft**


The setting of this parameter is the same as that in section 1.2.


~~~~~~~~

**Section 2.3: set drug_ft, smiles_dict, mpg_dict**


* **Use integrated drug feature**


For **drug_ft** *(str)*, ``"ECFP"``, ``"SMILES"``, ``"Graph"``, ``"Image"`` are available.

For **smiles_dict** *(dict)*, it should be **SMILES_dict** got by ``Data.DrRead.FeatDrug``.

For **mpg_dict** *(dict or None, optional, default: None)*,
if you want to use MPG (frozen) as the drug encoder, it should be **MPG_dict** got by ``Data.DrRead.FeatDrug``,
and if you want to use other drug encoders, you just need to use the default value.

Pairs lacking drug data will be removed based on the **smiles_dict** you set when you use ``.clean``.

* **Use your own drug feature**


For **drug_ft** *(dict)*, it should be your own dict ``drug_dict``, where the key is the drug name and the value is the feature vector, e.g. `SMILESVec_dict.pkl <https://huggingface.co/spaces/user15632/DeepDR/blob/main/additional/SMILESVec_dict.pkl>`_.

For **smiles_dict** *(dict or None, optional, default: None)*, you just need to use the default value.

For **mpg_dict** *(dict or None, optional, default: None)*, you just need to use the default value.

Pairs lacking drug data will be removed based on the **drug_ft** you set when you use ``.clean``.

--------

Clean the response data
--------

To clean response data, use ``.clean`` as follows, where ``data`` is the instantiated ``Data.DrData``:

.. code-block:: python

    data.clean(cell_ft_ls=...)


For **cell_ft_ls** *(list or None, optional, default: None)*,
usually you just need to use the default value, which will remove pairs lacking cell or drug data based on the setting of **cell_ft** and **drug_ft**.
For detailed removal rules, see **Sections 1.2 and 1.3** or **Sections 2.2 and 2.3** above.

If you want to build the benchmark, **cell_ft_ls** needs to be set as a list, and each element in the list has the same form as **cell_ft**, which will additionally remove pairs lacking cell data based on each element in the list.
The detailed removal rules are the same as the **cell_ft** based removal rules in **Section 1.2** or **Section 2.2** above.
