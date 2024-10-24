Model.DrModel
===========================

`Click here </document/Model/DrModelCode.html>`_ to view source code.


.. code-block:: python

    def DrModel(cell_encoder,
                drug_encoder,
                fusion_module,
                integrate: bool = True,
                cell_encoder_pt_path: str = None,
                drug_encoder_pt_path: str = None):

It can be used to build the drug response prediction model.

**PARAMETERS:**

* **cell_encoder** - The cell encoder built through ``DeepDR.CellEncoder``.

* **drug_encoder** - The drug encoder built through ``DeepDR.DrugEncoder``.

* **fusion_module** - The fusion module built through ``DeepDR.FusionModule``.

* **integrate** *(bool, optional)* - Whether to integrate the model in a class.  *(default: True)*

* **cell_encoder_pt_path** *(str, optional)* - The ``.pt`` path of pre-trained cell encoder. *(default: None)*

* **drug_encoder_pt_path** *(str, optional)* - The ``.pt`` path of pre-trained drug encoder. *(default: None)*

**OUTPUTS:**

* **model** - The drug response prediction model.
