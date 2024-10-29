Model.SetSeed
===========================

`Click here </en/latest/document/Model/SetSeed.html>`_ to go back to the reference.


.. code-block:: python

    def SetSeed(seed: int):
        """"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
