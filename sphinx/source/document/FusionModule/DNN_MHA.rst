DeepDR.FusionModule.DNN_MHA
===========================



.. code-block:: python

    class DNN_MHA(cell_dim: int = 512, drug_dim: int = 768, dropout: float = 0.3, num_heads: int = 8,
    pool: bool = True, concat: bool = False, hid_dim_ls: list = None)

It can be used for fusion cell line characterization and drug characterization to predict drug response.

**PARAMETERS:**

* **cell_dim** *(int, optional)* - Dimension of cell line characterization. *(default: 512)*
* **drug_dim** *(int, optional)* - Dimension of drug characterization. *(default: 768)*
* **dropout** *(float, optional)* - The dropout rate of DNN. *(default: 0.3)*
* **num_heads** *(int, optional)* - The number of heads of the multi-head attention network. *(default: 8)*

* **pool** *(bool, optional)* - The pooling method of drug characterization. *(default: True)*
    If pool is True, the method is 'mix'. If pool is False, the method is  'attention'.

* **concat** *(bool, optional)* - Whether to concat for fusion. *(default: False)*
    Cell line characterization and drug characterization are added instead of concat when the value is True.

* **hid_dim_ls** *(int, optional)* - Dimension of hidden layers of DNN. *(default: None)*


.. code-block:: python

    forward(f: torch.Tensor, x)

**INPUTS:**

* **f** *(torch.Tensor)* - The cell line characterization. *[batch_size, cell_dim]*

* **x** *(torch.Tensor)* - The drug characterization.
    If the drug encoding mode is 'ECFP', 'SMILESVec' or 'MPGVec', its shape is *[batch_size, drug_dim]*.
    If the drug encoding mode is 'SMILES', its shape is *[batch_size, drug_dim, seq_len]*.
    If the drug encoding mode is 'Graph' or 'MPGGraph', its format is *(encoded_graph.x, graph)*.

**OUTPUTS:**

* **f** *(torch.Tensor)* - The predicted drug response. *[batch_size, 1]*
* **cell_ft** *(torch.Tensor)* - The cell line features. *[batch_size, cell_dim]*
* **drug_ft** *(torch.Tensor)* - The drug features. *[batch_size, drug_dim]*
