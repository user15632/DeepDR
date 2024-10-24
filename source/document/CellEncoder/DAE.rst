CellEncoder.DAE
===========================

`Click here </document/CellEncoder/DAECode.html>`_ to view source code.


.. code-block:: python

    class DAE(subset: bool = True, path: str = None)

DAE can be used to encode exp of cells.

**PARAMETERS:**

* **subset** *(bool, optional)* - Use screened gene expression profiles or not. *(default: True)*
* **path** *(bool, optional)* - Path of DAE_ALL.pt. *(default: None)*

**SHAPES:**

* **input:** Original features. *[batch_size, in_dim]*
* **output:** Encoded features. *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - The input of DAE.
