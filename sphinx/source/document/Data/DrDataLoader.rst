DeepDR.Data.DrDataLoader
===========================



.. code-block:: python

    def DrDataLoader(dataset, batch_size: int = 128, shuffle: bool = True, follow_batch=None, exclude_keys=None)


It can be used to get the loaded dataset.

**PARAMETERS:**

* **dataset** - The dataset built by DeepDR.Data.DrDataset.

* **batch_size** *(int, optional)* - Parameter batch_size of torch.utils.data.DataLoader. *(default: 128)*
* **shuffle** *(bool, optional)* - Parameter shuffle of torch.utils.data.DataLoader. *(default: True)*

* **follow_batch** *(optional)* - Parameter follow_batch of torch_geometric.data.Batch.from_data_list. *(default: None)*
* **exclude_keys** *(optional)* - Parameter exclude_keys of torch_geometric.data.Batch.from_data_list. *(default: None)*

**OUTPUTS:**

* **dataloader** - The loaded dataset.
