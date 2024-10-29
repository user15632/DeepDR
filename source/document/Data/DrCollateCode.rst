Data.DrCollate
===========================

`Click here </en/latest/document/Data/DrCollate.html>`_ to go back to the reference.


.. code-block:: python

    class DrCollate:
        """"""

        def __init__(self, follow_batch=None,
                     exclude_keys=None):
            self.follow_batch = follow_batch
            self.exclude_keys = exclude_keys

        def __call__(self, batch):
            cell_ft = torch.stack([g.cell_ft for g in batch])
            if type(batch[0].drug_ft) == torch.Tensor:
                drug_ft = torch.stack([g.drug_ft for g in batch])
            else:
                drug_ft = Batch.from_data_list([g.drug_ft for g in batch], self.follow_batch, self.exclude_keys)
            response = torch.stack([g.response for g in batch])
            return cell_ft, drug_ft, response, [g.cell_name for g in batch], [g.drug_name for g in batch]
