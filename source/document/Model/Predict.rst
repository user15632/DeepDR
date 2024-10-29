Model.Predict
===========================

`Click here </en/latest/document/Model/PredictCode.html>`_ to view source code.


.. code-block:: python

    def Predict(model,
                data,
                save_path_prediction: str = None):

It can be used to predict drug responses using the trained model.

**PARAMETERS:**

* **model** - The model built by ``Model.DrModel`` and trained by ``Model.Train``.

* **data** - The data built by ``Model.DrData``.

* **save_path_prediction** *(str, optional)* - Save path of predictions. It is expected to end in ``".csv"``.

**OUTPUTS:**

* **pre** *(list)* - The predicted value of drug responses.
* **cell_ls** *(list)* - Cell names corresponding to drug responses.
* **drug_ls** *(list)* - Drug names corresponding to drug responses.
