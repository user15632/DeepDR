DeepDR.DrugEncoder.GCN
===========================



.. code-block:: python

   class GCN(x_num_embedding: int = 178, embedding_dim: int = 768, ft_dim: int = 768, hid_dim: int = 256,
   num_layers: int = 2)

GCN can be used to encode graphs of drugs.

**PARAMETERS:**

* **x_num_embedding** *(int, optional)* - Number of node embeddings. *(default: 178)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded node features. *(default: 768)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 256)*
* **num_layers** *(int, optional)* - Number of torch_geometric.nn.conv.GCNConv layers. *(default: 2)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** (Encoded node features *[node_num, ft_dim]*, Preprocessed graphs)

.. code-block:: python

	forward(g)

* **g** - The input of GCN.
