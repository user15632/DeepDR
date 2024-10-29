DrugEncoder.MPG
===========================

`Click here </document/DrugEncoder/MPG.html>`_ to go back to the reference.


.. code-block:: python

    class MPG(nn.Module):
        def __init__(self, ft_dim: int = 768, MPG_dim: int = 768, freeze: bool = True, conv: bool = True,
                     num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0, pt_path=None):
            """"""
            super(MPG, self).__init__()
            if freeze is False:
                assert pt_path is not None
            self.freeze = freeze
            self.conv = conv
            if self.freeze is not True:
                self.net = MolGNet(num_layer=num_layer, emb_dim=emb_dim, heads=heads,
                                   num_message_passing=num_message_passing, drop_ratio=drop_ratio)
                self.net.load_state_dict(torch.load(pt_path))
            if self.conv:
                self.output = GCNConv(MPG_dim, ft_dim)

        def forward(self, g):
            if self.freeze:
                x = g.mpg_ft
            else:
                x = self.net(g)
            if self.conv:
                x = self.output(x, g.edge_index)
            return x, g
