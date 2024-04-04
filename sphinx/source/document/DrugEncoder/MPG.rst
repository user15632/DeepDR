DeepDR.DrugEncoder.MPG
===========================



.. code-block:: python

   class MPG(MPG_dim: int = 768, ft_dim: int = 768)

MPG can be used to encode graphs of drugs.

**PARAMETERS:**

* **MPG_dim** *(int, optional)* - Dimension of MPG features. *(default: 768)*
* **ft_dim** *(int, optional)* - Dimension of encoded node features. *(default: 768)*

**SHAPES:**

* **input:** Preprocessed graphs
* **output:** (Encoded node features *[node_num, ft_dim]*, Preprocessed graphs)

.. code-block:: python

	forward(g)

* **g** - The input of MPG.
