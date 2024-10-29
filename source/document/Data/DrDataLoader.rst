Data.DrDataLoader
===========================

`Click here </document/Data/DrDataLoaderCode.html>`_ to view source code.


.. code-block:: python

    def DrDataLoader(dataset: DrDataset,
                     batch_size: int,
                     shuffle: bool,
                     follow_batch=None,
                     exclude_keys=None):

It can be used to load dataset.

**PARAMETERS:**

* **dataset** - The dataset built by ``Data.DrDataset``.

* **batch_size** *(int)* - Parameter ``batch_size`` of ``torch.utils.data.DataLoader``.
* **shuffle** *(bool)* - Parameter ``shuffle`` of ``torch.utils.data.DataLoader``.

* **follow_batch** *(optional)* - Parameter of ``torch_geometric.data.Batch.from_data_list``. *(default: None)*
* **exclude_keys** *(optional)* - Parameter of ``torch_geometric.data.Batch.from_data_list``. *(default: None)*

**OUTPUTS:**

* **dataloader** - The loaded dataset.
