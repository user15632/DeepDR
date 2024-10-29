CellEncoder.NULL
===========================

`Click here </document/CellEncoder/NULLCode.html>`_ to view source code.


.. code-block:: python

   class NULL()

Let the input and output the same.

**SHAPES:**

* **input:** Cell feature. *[batch_size, ft_dim]*
* **output:** Cell feature. *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - Cell feature.
