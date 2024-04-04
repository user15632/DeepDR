DeepDR.Model.MDL
===========================



.. code-block:: python

    def MDL(cell_encoder, drug_encoder, fusion_module, integrate: bool = False, cell_encoder_path: str = None)


**PARAMETERS:**

* **cell_encoder** - The cell encoder.
    The value could be 'DNN_EXP', 'CNN_EXP', 'DAE_EXP', 'DNN_GSVA', 'CNN_GSVA', 'DNN_CNV', 'CNN_CNV'
    or the cell encoder built through DeepDR.CellEncoder.

* **drug_encoder** - The drug encoder.
    The value could be 'DNN_ECFP', 'DNN_SMILESVec', 'CNN_SMILES', 'LSTM_SMILES', 'GRU_SMILES',
    'GCN_Graph', 'GAT_Graph', 'MPG_Graph', 'AttentiveFP_Graph', 'TrimNet_Graph'
    or the drug encoder built through DeepDR.DrugEncoder.

* **fusion_module** - The fusion module.
    The value could be 'DNN', 'MHA_DNN' or the fusion module built through DeepDR.FusionModule.

* **integrate** *(bool, optional)* - Whether to integrate the model in a class.  *(default: False)*

* **cell_encoder_path** *(str, optional)* - The .pt file path of pre-trained DAE cell encoder. *(default: None)*


**OUTPUTS:**

* **model** - The MDL model.
