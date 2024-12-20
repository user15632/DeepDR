DrugEncoder.GRU
===========================

`Click here </en/latest/document/DrugEncoder/GRU.html>`_ to go back to the reference.


.. code-block:: python

    class GRU(nn.Module):
        def __init__(self, num_embedding: int = _num_embedding, embedding_dim: int = _drug_dim, ft_dim: int = _drug_dim,
                     dropout: float = _dropout, bidirectional: bool = _bidirectional, num_layers: int = 2):
            """num_layers"""
            super(GRU, self).__init__()
            assert num_layers >= 1
            assert ft_dim % 2 == 0
            self.embedding = nn.Embedding(num_embedding, embedding_dim)
            self.encode_gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=ft_dim // (2 if bidirectional else 1),
                                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                                           batch_first=True)

        def forward(self, f):
            f = self.embedding(f)
            f, _ = self.encode_gru(f)
            f = f.permute(0, 2, 1).contiguous()
            return f
