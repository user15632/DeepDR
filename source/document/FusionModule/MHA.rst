FusionModule.MHA
===========================

`Click here </en/latest/document/FusionModule/MHACode.html>`_ to view source code.


.. code-block:: python

    class MHA(cell_dim: int, drug_dim: int, hid_dim_ls: list = None, dropout: float = 0.3, num_heads: int = 8,
    mix_pool: bool = True, concat: bool = True, classify: bool = False)

It can be used for fusion cell and drug feature to predict drug response.

**PARAMETERS:**

* **cell_dim** *(int)* - Dimension of cell feature.
* **drug_dim** *(int)* - Dimension of drug feature.
* **hid_dim_ls** *(int, optional)* - Dimension of hidden layers of DNN. *(default: None)*
    When the value is None, [512, 256, 128] is used.

* **dropout** *(float, optional)* - The dropout rate of DNN. *(default: 0.3)*
* **num_heads** *(int, optional)* - The number of heads of the multi-head attention network. *(default: 8)*

* **mix_pool** *(bool, optional)* - The pooling method of drug feature. *(default: True)*
    If the value is True, the method is 'mix' (mean, max, and attention). If pool is False, the method is 'attention'.

* **concat** *(bool, optional)* - Whether to concat for fusion. *(default: False)*
    Cell and drug feature are added instead of concat when the value is True.

* **classify** *(bool, optional)* - Whether classification task. *(default: False)*

.. code-block:: python

    forward(f: torch.Tensor, x)

**INPUTS:**

* **f** *(torch.Tensor)* - The cell feature. *[batch_size, cell_dim]*

* **x** *(torch.Tensor)* - The drug feature.
    If the drug feature is 'ECFP' or 'Image', its shape is *[batch_size, drug_dim]*.
    If the drug feature is 'SMILES', its shape is *[batch_size, drug_dim, seq_len]*.
    If the drug feature is 'Graph', its format is *(encoded_graph.x, graph)*.

**OUTPUTS:**

* **f** *(torch.Tensor)* - The predicted drug response. *[batch_size, 1]*
* **cell_ft** *(torch.Tensor)* - The cell feature. *[batch_size, cell_dim]*
* **drug_ft** *(torch.Tensor)* - The drug feature. *[batch_size, drug_dim]*
