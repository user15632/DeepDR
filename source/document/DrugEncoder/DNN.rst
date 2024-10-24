DrugEncoder.DNN
===========================

`Click here </document/DrugEncoder/DNNCode.html>`_ to view source code.


.. code-block:: python

   class DNN(in_dim: int, ft_dim: int, hid_dim: int = 512, num_layers: int = 2, dropout: float = 0.3)

DNN can be used to encode ECFPs of drugs.

**PARAMETERS:**

* **in_dim** *(int)* - Dimension of preprocessed ECFPs.
* **ft_dim** *(int)* - Dimension of encoded ECFPs.
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 512)*
* **num_layers** *(int, optional)* - Number of torch.nn.Linear layers. *(default: 2)*
* **dropout** *(float, optional)* - Dropout rate of DNN. *(default: 0.3)*

**SHAPES:**

* **input:** Preprocessed ECFPs *[batch_size, in_dim]*
* **output:** Encoded ECFPs *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of DNN.
