FusionModule.MHA
===========================

`Click here </document/FusionModule/MHA.html>`_ to go back to the reference.


.. code-block:: python

    class MHA(nn.Module):
        def __init__(self, cell_dim: int, drug_dim: int, hid_dim_ls: list = None, dropout: float = _dropout,
                     num_heads: int = _num_heads, mix_pool: bool = True, concat: bool = True, classify: bool = False):
            """hid_dim_ls"""
            super(MHA, self).__init__()
            self.classify = classify
            pool = 'mix' if mix_pool else 'attention'
            self.encode_dnn = DNN(cell_dim, drug_dim, hid_dim_ls, dropout, num_heads, pool, concat)

        def forward(self, f, x):
            f, cell_ft, drug_ft = self.encode_dnn(f, x)
            if self.classify:
                f = F.sigmoid(f)
            return f, cell_ft, drug_ft
