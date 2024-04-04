DeepDR.CellEncoder.NULL
===========================



.. code-block:: python

   class NULL()

Let the input and output the same.

**SHAPES:**

* **input:** Cell features *[batch_size, ft_dim]*
* **output:** Cell features *[batch_size, ft_dim]*

.. code-block:: python

	forward(f: torch.Tensor)

* **f** *(torch.Tensor)* - Cell features.
