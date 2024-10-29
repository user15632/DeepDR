DR prediction via DeepDR
==================================

The following are detailed tutorials on how to implement DR prediction.

Import as follows before starting DR Prediction:

.. code-block:: python

    from DeepDR import Data, Model, CellEncoder, DrugEncoder, FusionModule


Step 1: build and clean data
--------

.. code-block:: python

    data = Data.DrData(Data.DrRead.PairDef('CCLE', 'ActArea'), 'EXP', 'Graph').clean()



Build data with ``Data.DrData``, and then clean data using ``.clean`` on the instantiated ``Data.DrData``.
`Click here </document/Tutorials1.html>`_ for detailed tutorials.


Step 2: split response data
--------
.. code-block:: python

    train_data, val_data, _ = data.split('cell_out', fold=1, ratio=[0.8, 0.2, 0.0], seed=1)


Split response data using ``.split`` on the instantiated ``Data.DrData``.
`Click here </document/Data/DrData.html#self-split>`_ for details.

The ``train_data`` and ``val_data`` are lists, and each element in the list is the instantiated ``Data.DrData``.
The training data has the same index as the corresponding validation data.
The ``test_data`` is the instantiated ``Data.DrData`` (not used in this example, represented as ``_``).



Step 3: build and load dataset
--------
.. code-block:: python

    train_loader = Data.DrDataLoader(Data.DrDataset(train_data[0]), batch_size=64, shuffle=True)
    val_loader = Data.DrDataLoader(Data.DrDataset(val_data[0]), batch_size=64, shuffle=False)


Based on the instantiated ``Data.DrData``, build dataset with ``Data.DrDataset``.
`Click here </document/Data/DrDataset.html>`_ for details.

Load dataset with ``Data.DrDataLoader``.
`Click here </document/Data/DrDataLoader.html>`_ for details.


Step 4: build prediction model
--------
.. code-block:: python

    model = Model.DrModel(CellEncoder.DNN(6163, 100), DrugEncoder.MPG(), FusionModule.DNN(100, 768))


Build prediction model with ``Model.DrModel``.
`Click here </document/Model/DrModel.html>`_ for details.


Step 5: train and validate model
--------

.. code-block:: python

    result = Model.Train(model, epochs=100, lr=1e-4, train_loader=train_loader, val_loader=val_loader)


Train and validate model with ``Model.Train``.
`Click here </document/Model/Train.html>`_ for details.

The ``result`` is a tuple where the first element is the trained model.


Step 6: make prediction
--------
.. code-block:: python

    data.pair_ls = [['CAL120', '5-Fluorouracil'], ['CAL51', 'Afuresertib']]
    result = Model.Predict(model=result[0], data=data)


For simplicity, replace ``.pair_ls`` in the instantiated ``Data.DrData`` above with the pairs you want to predict.
The ``.pair_ls`` needs to be set to a list, each element in the list is a sub-list,
each element in the sub-list in turn is the cell name, drug name, and drug response (optional).

Then, make prediction with ``Model.Predict``.
`Click here </document/Model/Predict.html>`_ for details.
