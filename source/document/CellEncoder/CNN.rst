DeepDR.CellEncoder.CNN
===========================



.. code-block:: python

    class CNN(in_dim: int, ft_dim: int = 735, hid_channel_ls: list = None, kernel_size_conv: int = 7,
    stride_conv: int = 1, padding_conv: int = 0, kernel_size_pool: int = 3, stride_pool: int = 3,
    padding_pool: int = 0, batch_norm: bool = True, max_pool: bool = True, flatten: bool = True,
    debug: bool = False)

CNN can be used to encode exp, gsva, mut or cnv of cells.

**PARAMETERS:**

* **in_dim** *(int)* - Dimension of original feature.
* **ft_dim** *(int, optional)* - Dimension of encoded feature. *(default: 735)*

* **hid_channel_ls** *(list, optional)* - Hidden channels of CNN. *(default: None)*
    When the value is None, [40, 80, 60] is used.

* **kernel_size_conv** *(int, optional)* - Kernel size of torch.nn.Conv1d layer. *(default: 7)*
* **stride_conv** *(int, optional)* - Stride of torch.nn.Conv1d layer. *(default: 1)*
* **padding_conv** *(int, optional)* - Padding of torch.nn.Conv1d layer. *(default: 0)*

* **kernel_size_pool** *(int, optional)* - Kernel size of torch.nn.MaxPool1d layer. *(default: 3)*
* **stride_pool** *(int, optional)* - Stride of torch.nn.MaxPool1d layer. *(default: 3)*
* **padding_pool** *(int, optional)* - Padding of torch.nn.MaxPool1d layer. *(default: 0)*

* **batch_norm** *(bool, optional)* - Use torch.nn.BatchNorm1d layer or not. *(default: True)*
* **max_pool** *(bool, optional)* - Use torch.nn.MaxPool1d layer or not. *(default: True)*
* **flatten** *(bool, optional)* - Use torch.nn.Flatten() layer or not. *(default: True)*
* **debug** *(bool, optional)* - Print the shape of tensor f or not. *(default: False)*

**SHAPES:**

* **input:** Original features *[batch_size, in_dim]*
* **output:** Encoded features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of CNN.
