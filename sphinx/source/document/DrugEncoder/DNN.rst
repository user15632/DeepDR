DeepDR.DrugEncoder.DNN
===========================



.. code-block:: python

   class DNN(in_dim: int = 512, ft_dim: int = 768, dropout: float = 0.3, hid_dim: int = 256,
   num_layers: int = 2)

DNN can be used to encode ECFPs of drugs.

**PARAMETERS:**

* **in_dim** *(int, optional)* - Dimension of preprocessed ECFPs. *(default: 512)*
* **ft_dim** *(int, optional)* - Dimension of encoded ECFPs. *(default: 768)*
* **dropout** *(float, optional)* - Dropout rate of DNN. *(default: 0.3)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 256)*
* **num_layers** *(int, optional)* - Number of torch.nn.Linear layers. *(default: 2)*

**SHAPES:**

* **input:** Preprocessed ECFPs *[batch_size, in_dim]*
* **output:** Encoded ECFPs *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of DNN.
