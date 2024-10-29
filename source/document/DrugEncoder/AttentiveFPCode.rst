DrugEncoder.AttentiveFP
===========================

`Click here </document/DrugEncoder/AttentiveFP.html>`_ to go back to the reference.


.. code-block:: python

    class AttentiveFP(nn.Module):
        def __init__(self, x_num_embedding: int = _x_num_embedding, edge_num_embedding: int = _edge_num_embedding,
                     embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim, dropout: float = _dropout,
                     hid_dim: int = 384, num_layers: int = 2, num_steps: int = 3):
            """hid_dim, num_layers, num_steps"""
            super(AttentiveFP, self).__init__()
            assert num_layers >= 1
            self.x_embedding = nn.Embedding(x_num_embedding, embedding_dim)
            self.edge_embedding = nn.Embedding(edge_num_embedding, embedding_dim)
            self.reset_parameters()
            self.encode_AttentiveFP = models.AttentiveFP(in_channels=embedding_dim, hidden_channels=hid_dim,
                                                         out_channels=ft_dim, edge_dim=embedding_dim, num_layers=num_layers,
                                                         num_timesteps=num_steps, dropout=dropout)

        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        def forward(self, g):
            x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
            x = self.x_embedding(x).sum(1)
            edge_attr = self.edge_embedding(edge_attr).sum(1)
            x = self.encode_AttentiveFP(x, edge_index, edge_attr, batch)
            return x
