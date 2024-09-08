DeepDR.Model.Predict
===========================



.. code-block:: python

    def Predict(model, data, save_path_prediction: str = None)


It can be used to predict drug responses using the trained model.

**PARAMETERS:**

* **model** - The model built by DeepDR.Model.DrModel and trained by DeepDR.Model.Train.

* **data** - The data built by DeepDR.Model.DrData.

* **save_path_prediction** *(str, optional)* - The path to save the prediction results. *(default: None)*
    The value is expected to end in '.csv'.

**OUTPUTS:**

* **pre** *(list)* - The predicted value of drug responses.
* **cell_ls** *(list)* - The cell lines' name corresponding to drug responses.
* **drug_ls** *(list)* - The drugs' name corresponding to drug responses.
