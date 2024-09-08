An example of using DeepDR
==================================



Use DeepDR to train the model and then make predictions.

.. code-block:: python

    from DeepDR import Data, Model, CellEncoder, DrugEncoder, FusionModule
    data = Data.Read(dataset='CCLE', response='ActArea', cell_ft='EXP', drug_ft='Graph')
    train_data, val_data = Data.Split(data, mode='cell_out', ratio=[0.8, 0.2])
    train_loader = Data.DrDataLoader(Data.DrDataset(train_data), batch_size=64, shuffle=True)
    val_loader = Data.DrDataLoader(Data.DrDataset(val_data), batch_size=64, shuffle=False)
    model = Model.DrModel(CellEncoder.DNN(6163, 100), DrugEncoder.MPG(), FusionModule.DNN(100, 768))
    result = Model.Train(model, epochs=100, lr=1e-4, train_loader=train_loader, val_loader=val_loader)
    data.pair_ls = [['CAL120', '5-Fluorouracil'], ['CAL51', 'Afuresertib']]
    result = Model.Predict(model=result[0], data=data)

