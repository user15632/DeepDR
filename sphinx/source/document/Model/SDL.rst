DeepDR.Model.SDL
===========================



.. code-block:: python

    class SDL(cell_encoder, cell_dim: int = 512, dropout: float = 0.3, hid_dim_ls: list = None)


**PARAMETERS:**

* **cell_encoder** - The cell encoder.
    The value could be 'DNN_EXP', 'CNN_EXP', 'DNN_GSVA', 'CNN_GSVA', 'DNN_CNV', 'CNN_CNV' or the cell encoder built through DeepDR.CellEncoder.

* **dropout** *(float, optional)* - The dropout rate of DNN. *(default: 0.3)*
* **hid_dim_ls** *(int, optional)* - Dimension of hidden layers of DNN. *(default: None)*


.. code-block:: python

    forward(cell_ft: torch.Tensor)

**OUTPUTS:**

* **res** *(torch.Tensor)* - The predicted drug response. *[batch_size, 1]*
* **cell_ft** *(torch.Tensor)* - The cell line features. *[batch_size, cell_dim]*
