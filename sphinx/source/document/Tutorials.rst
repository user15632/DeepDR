An example of using DeepDR
==================================



Use DeepDR to train the model.

.. code-block:: python

    from DeepDR import Data
    from DeepDR.Model import MDL, Train

    pair_list = Data.Read(pair_list='CCLE_ActArea', cell_dict='CCLE_EXP')
    train_pair, _, test_pair = Data.Split(pair_list, ratio=[0.8, 0, 0.2])
    train_load = Data.DrDataLoader(Data.DrDataset(train_pair, drug_encoding='MPGGraph', cell_dict='CCLE_EXP'))
    model, loss_func, _, _ = Train(MDL('DNN_EXP', 'MPG_Graph', 'MHA_DNN', integrate=True),
                                   train_loader=train_load, epochs=1, lr=1e-4)


Use DeepDR to make predictions based on the trained model above.

.. code-block:: python

    from DeepDR.Model import Predict

    test_load = Data.DrDataLoader(Data.DrDataset(test_pair, drug_encoding='MPGGraph', cell_dict='CCLE_EXP'))
    loss, real, pre, cell_ls, drug_ls = Predict(model, test_load, loss_func=loss_func)

