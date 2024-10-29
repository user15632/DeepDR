Model.DrModel
===========================

`Click here </document/Model/DrModel.html>`_ to go back to the reference.


.. code-block:: python

    class _Model(nn.Module):
        def __init__(self, cell_encoder,
                     drug_encoder,
                     fusion_module,
                     cell_encoder_pt_path: str,
                     drug_encoder_pt_path: str):
            """"""

            super(_Model, self).__init__()
            self.CellEncoder = cell_encoder
            self.DrugEncoder = drug_encoder
            self.FusionModule = fusion_module
            if cell_encoder_pt_path is not None:
                self.CellEncoder.load_state_dict(torch.load(cell_encoder_pt_path))
            if drug_encoder_pt_path is not None:
                self.DrugEncoder.load_state_dict(torch.load(drug_encoder_pt_path))

        def forward(self, cell_ft, drug_ft):
            cell_ft = self.CellEncoder(cell_ft)
            drug_ft = self.DrugEncoder(drug_ft)
            res, cell_ft, drug_ft = self.FusionModule(cell_ft, drug_ft)
            return res, cell_ft, drug_ft


    def _ModelTP(cell_encoder,
                 drug_encoder,
                 fusion_module,
                 cell_encoder_pt_path: str,
                 drug_encoder_pt_path: str):
        """"""

        if cell_encoder_pt_path is not None:
            cell_encoder.load_state_dict(torch.load(cell_encoder_pt_path))
        if drug_encoder_pt_path is not None:
            drug_encoder.load_state_dict(torch.load(drug_encoder_pt_path))
        return cell_encoder, drug_encoder, fusion_module


    def DrModel(cell_encoder,
                drug_encoder,
                fusion_module,
                integrate: bool = True,
                cell_encoder_pt_path: str = None,
                drug_encoder_pt_path: str = None):

        if integrate:
            model = _Model(cell_encoder, drug_encoder, fusion_module, cell_encoder_pt_path, drug_encoder_pt_path)
        else:
            model = _ModelTP(cell_encoder, drug_encoder, fusion_module, cell_encoder_pt_path, drug_encoder_pt_path)
        return model
