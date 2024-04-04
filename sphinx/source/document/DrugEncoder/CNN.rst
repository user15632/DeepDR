DeepDR.DrugEncoder.CNN
===========================



.. code-block:: python

   class CNN(num_embedding: int = 37, embedding_dim: int = 768, ft_dim: int = 768, kernel_size: int = 3,
   padding: int = 1, hid_dim: int = 256, num_layers: int = 2)

CNN can be used to encode SMILES of drugs.

**PARAMETERS:**

* **num_embedding** *(int, optional)* - Number of embeddings. *(default: 37)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded SMILES. *(default: 768)*
* **kernel_size** *(int, optional)* - Parameter kernel_size of torch.nn.Conv1d. *(default: 3)*
* **padding** *(int, optional)* - Parameter padding of torch.nn.Conv1d. *(default: 1)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 256)*
* **num_layers** *(int, optional)* - Number of torch.nn.Conv1d layers. *(default: 2)*

**SHAPES:**

* **input:** Preprocessed SMILES *[batch_size, seq_len]*
* **output:** Encoded SMILES *[batch_size, ft_dim, seq_len]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of CNN.
