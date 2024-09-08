DeepDR.DataPreprocess.PreSmiles
===========================



.. code-block:: python

    def PreSmiles(smiles: str, max_len: int = None, char_dict: dict = None, right: bool = True)

**PARAMETERS:**

* **smiles** *(str)* - SMILES of the drug.

* **max_len** *(int, optional)* - The max length of the sequence. *(default: None)*
    The default value is used when it is None.

* **char_dict** *(dict, optional)* - The character-integer mapping dict of the sequence. *(default: None)*
    The value could be None or your own dict.
    The default dict is used when the value is None.
    The key of the dict is the character and the value is the corresponding integer.

* **right** *(bool, optional)* - The padding direction of the sequence. *(default: True)*
    If the value is True, padding to the right.
    If the value is False, padding to the left.

**OUTPUTS:**

* **drug_ft** *(torch.Tensor)* - The drug feature.
