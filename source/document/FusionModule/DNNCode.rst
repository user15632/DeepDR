FusionModule.DNN
===========================

`Click here </en/latest/document/FusionModule/DNN.html>`_ to go back to the reference.


.. code-block:: python

    class DNN(nn.Module):
        def __init__(self, cell_dim: int, drug_dim: int, hid_dim_ls: list = None, dropout: float = _dropout,
                     num_heads: int = _num_heads, pool: str = 'mean', concat: bool = True, classify: bool = False):
            """"""
            super(DNN, self).__init__()
            if hid_dim_ls is None:
                hid_dim_ls = [512, 256, 128]
            self.pool = pool
            self.concat = concat
            self.classify = classify

            assert self.pool in ['attention', 'mean', 'max', 'mix']
            if self.pool in ['attention', 'mix']:
                self.attention = _MHA(cell_dim=cell_dim, drug_dim=drug_dim, dropout=dropout, num_heads=num_heads)
            if not self.concat:
                dim_ls = [drug_dim] + hid_dim_ls + [1]
                self.input_cell = nn.Linear(cell_dim, dim_ls[0])
                self.input_drug = nn.Linear(drug_dim, dim_ls[0])
            else:
                dim_ls = [cell_dim + drug_dim] + hid_dim_ls + [1]
                self.input_cell = nn.Linear(cell_dim, cell_dim)
                self.input_drug = nn.Linear(drug_dim, drug_dim)

            self.encode_dnn = nn.ModuleList([nn.Linear(dim_ls[i], dim_ls[i + 1]) for i in range(len(dim_ls) - 2)])
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(len(dim_ls) - 2)])
            self.output = nn.Linear(dim_ls[-2], dim_ls[-1])

        def forward(self, f, x):
            if type(x) == torch.Tensor:
                if len(x.shape) == 3:
                    if self.pool == 'mean':
                        x = torch.mean(x, dim=2)
                    elif self.pool == 'max':
                        x, _ = torch.max(x, dim=2)
                    elif self.pool == 'attention':
                        x = self.attention(f, x)
                    else:
                        x_m, _ = torch.max(x, dim=2)
                        x = torch.mean(x, dim=2) + x_m + self.attention(f, x)
            else:
                x, g = x
                if self.pool == 'mean':
                    x = global_mean_pool(x, g.batch)
                elif self.pool == 'max':
                    x = global_max_pool(x, g.batch)
                elif self.pool == 'attention':
                    x = self.attention(f, (x, g))
                else:
                    x = global_mean_pool(x, g.batch) + global_max_pool(x, g.batch) + self.attention(f, (x, g))

            cell_ft, drug_ft = f, x
            f = F.relu(self.input_cell(f))
            x = F.relu(self.input_drug(x))
            if not self.concat:
                f = f + x
            else:
                f = torch.cat((f, x), dim=1)
            for i in range(len(self.encode_dnn)):
                f = F.relu(self.encode_dnn[i](f))
                f = self.dropout[i](f)
            f = self.output(f)
            if self.classify:
                f = F.sigmoid(f)
            return f, cell_ft, drug_ft
