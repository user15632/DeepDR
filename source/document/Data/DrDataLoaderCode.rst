Data.DrDataLoader
===========================

`Click here </en/latest/document/Data/DrDataLoader.html>`_ to go back to the reference.


.. code-block:: python

    def DrDataLoader(dataset: DrDataset,
                     batch_size: int,
                     shuffle: bool,
                     follow_batch=None,
                     exclude_keys=None):
        """"""
        collate = DrCollate(follow_batch, exclude_keys)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
        return dataloader
