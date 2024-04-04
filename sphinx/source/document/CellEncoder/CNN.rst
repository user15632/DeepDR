DeepDR.CellEncoder.CNN
===========================



.. code-block:: python

    class CNN(in_dim: int = 6163, ft_dim: int = 512, kernel_size: int = 3, padding: int = 1,
    hid_dim: int = 64, num_layers: int = 3)

CNN can be used to encode exp, gsva or cnv of cell lines.

**PARAMETERS:**

* **in_dim** *(int, optional)* - Dimension of original features. *(default: 6163)*
* **ft_dim** *(int, optional)* - Dimension of encoded features. *(default: 512)*
* **kernel_size** *(int, optional)* - Kernel size of CNN. *(default: 3)*
* **padding** *(int, optional)* - Padding size of CNN. *(default: 1)*
* **hid_dim** *(int, optional)* - Number of hidden channels. *(default: 64)*
* **num_layers** *(int, optional)* - Number of torch.nn.Conv1d layers. *(default: 3)*

**SHAPES:**

* **input:** Original features *[batch_size, in_dim]*
* **output:** Encoded features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of CNN.
