Data.DrDataset
===========================

`Click here </document/Data/DrDatasetCode.html>`_ to view source code.


.. code-block:: python

    class DrDataset(dr_data: DrData,
                    radius: int = 2,
                    nBits: int = 512,
                    max_len: int = 230,
                    char_dict: dict = None,
                    right: bool = True,
                    mpg: bool = True):


It can be used to build drug response dataset.

**PARAMETERS:**

* **dr_data** *(DrData)* - The drug response data.

* **radius** *(int, optional)* - The radius of ECFP. *(default: 2)*
* **nBits** *(int, optional)* - The nBits of ECFP. *(default: 512)*

* **max_len** *(int, optional)* - The max length of the sequence. *(default: 230)*

* **char_dict** *(dict, optional)* - The character-integer mapping dict of the sequence. The key of the dict is the character and the value is the corresponding integer. The default dict is used when the value is set to ``None``. *(default: None)*

* **right** *(bool, optional)* - The padding direction of the sequence. If the value is set to ``True``, padding to the right. If the value is set to ``False``, padding to the left. *(default: True)*

* **mpg** *(bool, optional)* - Load MPG feature or not. *(default: True)*
