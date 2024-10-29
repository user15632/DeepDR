CellEncoder.DAE
===========================

`Click here </en/latest/document/CellEncoder/DAE.html>`_ to go back to the reference.


.. code-block:: python

    class DAE(nn.Module):
        def __init__(self, subset: bool = True, path: str = None):
            """"""
            super(DAE, self).__init__()
            if subset:
                self.encoder = DNN(in_dim=6163, ft_dim=100)
                self.encoder.load_state_dict(torch.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/DAE.pt')))
                # self.decoder = DNN(in_dim=100, ft_dim=6163)
            else:
                assert path is not None
                self.encoder = DNN(in_dim=17420, hid_dim=512, num_layers=3, ft_dim=100)
                self.encoder.load_state_dict(torch.load(path))
                # self.decoder = DNN(in_dim=100, ft_dim=17420)

        def forward(self, f):
            return self.encoder(f)
