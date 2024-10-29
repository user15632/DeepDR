DrugEncoder.GCN
===========================

`Click here </en/latest/document/DrugEncoder/GCN.html>`_ to go back to the reference.


.. code-block:: python

    class GCN(nn.Module):
        def __init__(self, x_num_embedding: int = _x_num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                     hid_dim: int = 384, num_layers: int = 2):
            """hid_dim, num_layers"""
            super(GCN, self).__init__()
            assert num_layers >= 2
            self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
            self.reset_parameters()
            self.input = GCNConv(embedding_dim, hid_dim)
            self.encode_gcn = nn.ModuleList([GCNConv(hid_dim, hid_dim) for _ in range(num_layers - 2)])
            self.output = GCNConv(hid_dim, ft_dim)

        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        def forward(self, g):
            x, edge_index = g.x, g.edge_index
            x = self.x_embedding(x).sum(1)
            x = F.relu(self.input(x, edge_index))
            for layer in self.encode_gcn:
                x = x + F.relu(layer(x, edge_index))
            x = self.output(x, edge_index)
            return x, g
