DeepDR.Data.DrDataset
===========================



.. code-block:: python

    class DrDataset(dr_data: DrData, radius: int = 2, nBits: int = 512, max_len: int = 230,
    char_dict: dict = None, right: bool = True, MPG_dict: dict = None, MPG: bool = True)


**PARAMETERS:**

* **dr_data** *(DrData)* - The drug response data.

* **radius** *(int, optional)* - The radius of ECFP. *(default: 2)*
* **nBits** *(int, optional)* - The nBits of ECFP. *(default: 512)*

* **max_len** *(int, optional)* - The max length of the sequence. *(default: 230)*
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

* **MPG_dict** *(bool, optional)* - Load MPG feature or not. *(default: True)*
