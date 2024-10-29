DrugEncoder.GRU
===========================

`Click here </en/latest/document/DrugEncoder/GRUCode.html>`_ to view source code.


.. code-block:: python

   class GRU(num_embedding: int = 37, embedding_dim: int = 768, ft_dim: int = 768, dropout: float = 0.3,
   bidirectional: bool = True, num_layers: int = 2)

GRU can be used to encode SMILES of drugs.

**PARAMETERS:**

* **num_embedding** *(int, optional)* - Number of embeddings. *(default: 37)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded SMILES. *(default: 768)*
* **dropout** *(float, optional)* - Dropout rate of GRU. *(default: 0.3)*
* **bidirectional** *(bool, optional)* - Parameter bidirectional of torch.nn.GRU. *(default: True)*
* **num_layers** *(int, optional)* - Parameter num_layers of torch.nn.GRU. *(default: 2)*

**SHAPES:**

* **input:** Preprocessed SMILES *[batch_size, seq_len]*
* **output:** Encoded SMILES *[batch_size, ft_dim, seq_len]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of GRU.
