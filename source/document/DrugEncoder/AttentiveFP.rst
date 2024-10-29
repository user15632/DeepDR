DrugEncoder.AttentiveFP
===========================

`Click here </en/latest/document/DrugEncoder/AttentiveFPCode.html>`_ to view source code.


.. code-block:: python

   class AttentiveFP(x_num_embedding: int = 178, edge_num_embedding: int = 18, embedding_dim: int = 768,
   ft_dim: int = 768, dropout: float = 0.3, hid_dim: int = 384, num_layers: int = 2, num_steps: int = 3)

AttentiveFP can be used to encode graphs of drugs.

**PARAMETERS:**

* **x_num_embedding** *(int, optional)* - Number of node embeddings. *(default: 178)*
* **edge_num_embedding** *(int, optional)* - Number of edge embeddings. *(default: 18)*
* **embedding_dim** *(int, optional)* - Dimension of embeddings. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded drug features. *(default: 768)*
* **dropout** *(float, optional)* - Dropout rate of AttentiveFP. *(default: 0.3)*
* **hid_dim** *(int, optional)* - Dimension of hidden layers. *(default: 384)*
* **num_layers** *(int, optional)* - Number of AttentiveFP layers. *(default: 2)*
* **num_steps** *(int, optional)* - Number of AttentiveFP steps. *(default: 3)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** Encoded drug features *[batch_size, ft_dim]*

.. code-block:: python

	forward(g)

* **g** - The input of AttentiveFP.
