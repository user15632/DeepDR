CellEncoder.DNN
===========================

`Click here </en/latest/document/CellEncoder/DNN.html>`_ to go back to the reference.


.. code-block:: python

    class DNN(nn.Module):
        def __init__(self, in_dim: int, ft_dim: int, hid_dim: int = 100, num_layers: int = 2, dropout: float = 0.3):
            """hid_dim, num_layers"""
            super(DNN, self).__init__()
            assert num_layers >= 1
            dim_ls = [in_dim] + [hid_dim] * (num_layers - 1) + [ft_dim]
            self.encode_dnn = nn.ModuleList([nn.Linear(dim_ls[i], dim_ls[i + 1]) for i in range(num_layers - 1)])
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers - 1)])
            self.output = nn.Linear(dim_ls[-2], dim_ls[-1])

        def forward(self, f):
            for i in range(len(self.encode_dnn)):
                f = F.relu(self.encode_dnn[i](f))
                f = self.dropout[i](f)
            f = self.output(f)
            return f
