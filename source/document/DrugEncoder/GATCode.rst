DrugEncoder.GAT
===========================

`Click here </document/DrugEncoder/GAT.html>`_ to go back to the reference.


.. code-block:: python

    class GAT(nn.Module):
        def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                     embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim, num_heads: int = _num_heads,
                     dropout: float = _dropout, hid_dim: int = 384, num_layers: int = 2):
            """hid_dim num_layers"""
            super(GAT, self).__init__()
            assert num_layers >= 2
            self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
            self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
            self.reset_parameters()
            self.input = GATConv(embedding_dim, hid_dim, heads=num_heads, concat=False, edge_dim=embedding_dim,
                                 dropout=dropout)
            self.encode_gat = nn.ModuleList(
                [GATConv(hid_dim, hid_dim, heads=num_heads, concat=False, edge_dim=embedding_dim, dropout=dropout)
                 for _ in range(num_layers - 2)])
            self.output = GATConv(hid_dim, ft_dim, heads=num_heads, concat=False, edge_dim=embedding_dim)

        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        def forward(self, g):
            x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
            x = self.x_embedding(x).sum(1)
            edge_attr = self.edge_embedding(edge_attr).sum(1)
            x = F.relu(self.input(x, edge_index, edge_attr))
            for layer in self.encode_gat:
                x = x + F.relu(layer(x, edge_index, edge_attr))
            x = self.output(x, edge_index, edge_attr)
            return x, g
