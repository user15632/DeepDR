DrugEncoder.CNN
===========================

`Click here </en/latest/document/DrugEncoder/CNN.html>`_ to go back to the reference.


.. code-block:: python

    class CNN(nn.Module):
        def __init__(self, embedding: bool = True, num_embedding: int = _num_embedding, embedding_dim: int = 735,
                     hid_channel_ls: list = None, kernel_size_conv: int = 7, stride_conv: int = 1, padding_conv: int = 0,
                     kernel_size_pool: int = 3, stride_pool: int = 3, padding_pool: int = 0, batch_norm: bool = True,
                     max_pool: bool = True, flatten: bool = True, debug: bool = False):
            """tCNNS: let embedding=False batch_norm=False"""
            super(CNN, self).__init__()
            self.embedding = embedding
            self.num_embedding = num_embedding
            self.batch_norm = batch_norm
            self.max_pool = max_pool
            self.flatten = flatten
            self.debug = debug

            if hid_channel_ls is None:
                hid_channel_ls = [40, 80, 60]
            channel_ls = [embedding_dim if self.embedding else num_embedding] + hid_channel_ls

            if self.embedding:
                self.embed = nn.Embedding(num_embedding, embedding_dim)
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=channel_ls[i], out_channels=channel_ls[i + 1],
                                                 kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv)
                                       for i in range(len(channel_ls) - 1)])
            if self.batch_norm:
                self.norm = nn.ModuleList([nn.BatchNorm1d(channel_ls[i + 1]) for i in range(len(channel_ls) - 1)])
            if self.max_pool:
                self.pool = nn.ModuleList([nn.MaxPool1d(kernel_size=kernel_size_pool, stride=stride_pool,
                                                        padding=padding_pool) for _ in range(len(channel_ls) - 1)])
            if self.flatten:
                self.flat = nn.Flatten()

        def forward(self, f):
            f = self.embed(f) if self.embedding else F.one_hot(f.to(torch.int64), self.num_embedding).float()
            f = f.permute(0, 2, 1).contiguous()

            for i in range(len(self.conv)):
                f = self.conv[i](f)
                if self.batch_norm:
                    f = self.norm[i](f)
                f = F.relu(f)
                if self.max_pool:
                    f = self.pool[i](f)

            if self.flatten:
                f = self.flat(f)

            if self.debug:
                print(f.shape)
            return f
