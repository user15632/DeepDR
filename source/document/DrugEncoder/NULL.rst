DrugEncoder.NULL
===========================

`Click here </en/latest/document/DrugEncoder/NULLCode.html>`_ to view source code.


.. code-block:: python

   class NULL()

Let the input and output the same.

**SHAPES:**

* **input:** Drug features *[batch_size, ft_dim]*
* **output:** Drug features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - Drug features.
