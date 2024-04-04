DeepDR.CellEncoder.DNN
===========================



.. code-block:: python

    class DNN(in_dim: int = 6163, ft_dim: int = 512, dropout: float = 0.3, hid_dim: int = 1024,
    num_layers: int = 2)

DNN can be used to encode exp, gsva or cnv of cell lines.

**PARAMETERS:**

* **in_dim** *(int, optional)* - Dimension of original features. *(default: 6163)*
* **ft_dim** *(int, optional)* - Dimension of encoded features. *(default: 512)*
* **dropout** *(float, optional)* - Dropout rate of DNN. *(default: 0.3)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 1024)*
* **num_layers** *(int, optional)* - Number of torch.nn.Linear layers. *(default: 2)*

**SHAPES:**

* **input:** Original features *[batch_size, in_dim]*
* **output:** Encoded features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of DNN.
