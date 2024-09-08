DeepDR.DrugEncoder.TrimNet
===========================



.. code-block:: python

   class TrimNet(x_num_embedding: int = 178, edge_num_embedding: int = 18, embedding_dim: int = 768,
   ft_dim: int = 2, num_heads: int = 4, dropout: float = 0.1, hid_dim: int = 32, depth: int = 3)

AttentiveFP can be used to encode graphs of drugs.

**PARAMETERS:**

* **x_num_embedding** *(int, optional)* - Number of node embeddings. *(default: 178)*
* **edge_num_embedding** *(int, optional)* - Number of edge embeddings. *(default: 18)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded drug features. *(default: 2)*
* **num_heads** *(int, optional)* - Number of TrimNet heads. *(default: 4)*
* **dropout** *(float, optional)* - Dropout rate of TrimNet. *(default: 0.1)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 32)*
* **depth** *(int, optional)* - Depth of TrimNet. *(default: 3)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** Encoded drug features *[batch_size, ft_dim]*

.. code-block:: python

	forward(g)

* **g** - The input of TrimNet.
