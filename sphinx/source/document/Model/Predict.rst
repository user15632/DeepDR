DeepDR.Model.Predict
===========================



.. code-block:: python

    def Predict(model, data_loader, no_tag: bool = False, loss_func=None, classify: bool = False,
    save_path_prediction: str = None)


It can be used to predict drug responses using the trained model.

**PARAMETERS:**

* **model** - The trained model.
    The value could be the model built by DeepDR.Model.MDL or DeepDR.Model.SDL and trained by DeepDR.Model.Train.

* **data_loader** - The loaded data got by DeepDR.Data.DrDataLoader.

* **no_tag** *(bool, optional)* - Whether the dataset contains no tags. *(default: False)*

* **loss_func** *(optional)* - The loss function. *(default: None)*
    The torch.nn.MSELoss is used when the value is None and parameter classify is None.

* **classify** *(bool, optional)* - Whether classification task. *(default: False)*

* **save_path_prediction** *(str, optional)* - The path to save the prediction results. *(default: None)*
    The value is expected to end in '.csv'.

**OUTPUTS:**

    When the value of no_tag is False

* **epoch_loss** *(float)* - The value of loss.
* **real** *(list)* - The real value of drug responses.
* **pre** *(list)* - The predicted value of drug responses.
* **cell_ls** *(list)* - The cell lines' name corresponding to drug responses.
* **drug_ls** *(list)* - The drugs' name corresponding to drug responses.

    When the value of no_tag is True

* **pre** *(list)* - The predicted value of drug responses.
* **cell_ls** *(list)* - The cell lines' name corresponding to drug responses.
* **drug_ls** *(list)* - The drugs' name corresponding to drug responses.
