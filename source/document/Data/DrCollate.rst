Data.DrCollate
===========================

`Click here </document/Data/DrCollateCode.html>`_ to view source code.


.. code-block:: python

    class DrCollate(follow_batch=None,
                    exclude_keys=None):


It can be used to build the collate method in loading dataset.

**PARAMETERS:**

* **follow_batch** *(optional)* - Parameter of ``torch_geometric.data.Batch.from_data_list``. *(default: None)*
* **exclude_keys** *(optional)* - Parameter of ``torch_geometric.data.Batch.from_data_list``. *(default: None)*
