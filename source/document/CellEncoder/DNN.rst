CellEncoder.DNN
===========================

`Click here </document/CellEncoder/DNNCode.html>`_ to view source code.


.. code-block:: python

    class DNN(in_dim: int, ft_dim: int, hid_dim: int = 100, num_layers: int = 2, dropout: float = 0.3)

DNN can be used to encode exp, gsva, mut or cnv of cells.

**PARAMETERS:**

* **in_dim** *(int)* - Dimension of original features.
* **ft_dim** *(int)* - Dimension of encoded features.
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 100)*
* **dropout** *(float, optional)* - Dropout rate of DNN. *(default: 0.3)*
* **num_layers** *(int, optional)* - Number of torch.nn.Linear layers. *(default: 2)*

**SHAPES:**

* **input:** Original features *[batch_size, in_dim]*
* **output:** Encoded features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of DNN.
