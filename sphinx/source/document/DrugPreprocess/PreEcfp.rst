DeepDR.DataPreprocess.PreEcfp
===========================



.. code-block:: python

    def PreEcfp(smiles: str, radius: int = 2, nBits: int = 512)

**PARAMETERS:**

* **smiles** *(str)* - SMILES of the drug.
* **radius** *(int, optional)* - The radius of ECFP. *(default: 2)*
* **nBits** *(int, optional)* - The nBits of ECFP. *(default: 512)*

**OUTPUTS:**

* **drug_ft** *(torch.Tensor)* - The drug feature.
