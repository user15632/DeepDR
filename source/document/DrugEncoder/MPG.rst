DeepDR.DrugEncoder.MPG
===========================



.. code-block:: python

   class MPG(ft_dim: int = 768, MPG_dim: int = 768, freeze: bool = True, conv: bool = True,
   num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0, pt_path=None)

MPG can be used to encode graphs of drugs.

**PARAMETERS:**

* **ft_dim** *(int, optional)* - Dimension of encoded node features. *(default: 768)*
* **MPG_dim** *(int, optional)* - Dimension of MPG features. *(default: 768)*
* **freeze** *(bool, optional)* - Freeze MPG or not. *(default: True)*
* **conv** *(bool, optional)* - Use torch_geometric.nn.conv.GCNConv as output layer or not. *(default: True)*
* **num_layer** *(optional)* - Parameter num_layer of MPG. *(default: 5)*
* **emb_dim** *(optional)* - Parameter emb_dim of MPG. *(default: 768)*
* **heads** *(optional)* - Parameter heads of MPG. *(default: 12)*
* **num_message_passing** *(optional)* - Parameter num_message_passing of MPG. *(default: 3)*
* **drop_ratio** *(optional)* - Parameter drop_ratio of MPG. *(default: 0)*
* **pt_path** *(optional)* - Path of MolGNet.pt. *(default: None)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** (Encoded node features *[node_num, ft_dim]*, Preprocessed graphs)

.. code-block:: python

	forward(g)

* **g** - The input of MPG.
