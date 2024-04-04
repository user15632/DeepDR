DeepDR.DrugEncoder.GAT
===========================



.. code-block:: python

   class GAT(x_num_embedding: int = 178, edge_num_embedding: int = 18, embedding_dim: int = 768,
   ft_dim: int = 768, num_heads: int = 4, dropout: float = 0.3, hid_dim: int = 384, num_layers: int = 2)

GAT can be used to encode graphs of drugs.

**PARAMETERS:**

* **x_num_embedding** *(int, optional)* - Number of node embeddings. *(default: 178)*
* **edge_num_embedding** *(int, optional)* - Number of edge embeddings. *(default: 18)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded node features. *(default: 768)*
* **num_heads** *(int, optional)* - Parameter heads of torch_geometric.nn.conv.GATConv. *(default: 4)*
* **dropout** *(float, optional)* - Dropout rate of torch_geometric.nn.conv.GATConv. *(default: 0.3)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 384)*
* **num_layers** *(int, optional)* - Number of torch_geometric.nn.conv.GATConv layers. *(default: 3)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** (Encoded node features *[node_num, ft_dim]*, Preprocessed graphs)

.. code-block:: python

	forward(g)

* **g** - The input of GAT.
