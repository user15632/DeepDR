DrugEncoder.TrimNet
===========================

`Click here </en/latest/document/DrugEncoder/TrimNet.html>`_ to go back to the reference.


.. code-block:: python

    class TrimNet(nn.Module):
        def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                     embedding_dim: int = _drug_dim, ft_dim: int = 2, num_heads: int = 4, dropout: float = 0.1,
                     hid_dim: int = 32, depth: int = 3):
            """"""
            super(TrimNet, self).__init__()
            self.dropout = dropout
            self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
            self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
            self.reset_parameters()
            self.lin0 = nn.Linear(embedding_dim, hid_dim)
            self.convs = nn.ModuleList([Block(hid_dim, embedding_dim, num_heads) for _ in range(depth)])
            self.set2set = Set2Set(hid_dim, processing_steps=3)
            self.out = nn.Sequential(nn.Linear(2 * hid_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
                                     nn.Dropout(p=self.dropout), nn.Linear(512, ft_dim))

        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        def forward(self, g):
            x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
            x = self.x_embedding(x).sum(1)
            edge_attr = self.edge_embedding(edge_attr).sum(1)
            x = F.celu(self.lin0(x))
            for conv in self.convs:
                x = x + F.dropout(conv(x, edge_index, edge_attr), p=self.dropout, training=self.training)
            x = self.set2set(x, batch)
            x = self.out(F.dropout(x, p=self.dropout, training=self.training))
            return x
