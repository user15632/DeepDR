import joblib
import random
import torch
import os
from DeepDR import Data, Model, CellEncoder, DrugEncoder, FusionModule

seeds = [1, 10, 100]

for seed in seeds:
    batch_size = 64
    Model.SetSeed(seed)

    """"""
    lab = 'GDSC_drug_out'
    txt_path = 'tuning_GDSC_drug_out_1_1.txt'
    data_path = 'GDSC2_IC50_drug_out_seed={}.pkl'.format(seed)

    model_ls = ['DAE(EXP_ALL)_DNN(ECFP)_DNN']

    cell_ft_ls = [joblib.load('GDSC_EXP_ALL.pkl')]

    drug_ft_ls = ['ECFP' for _ in range(len(model_ls))]

    cell_encoder_ls = [CellEncoder.DNN(in_dim=17420, hid_dim=512, num_layers=3, ft_dim=100)]
    cell_encoder_ls[0].load_state_dict(torch.load('DAE_ALL.pt'))

    drug_encoder_ls = [DrugEncoder.DNN(in_dim=512, ft_dim=512) for _ in range(len(model_ls))]

    fusion_module_ls = [FusionModule.DNN(cell_dim=100, drug_dim=512)]
    """"""

    with open(txt_path, 'a') as file0:
        print('', file=file0)
        print(seed, file=file0)

    lr_ls = [1e-3, 1e-4, 1e-5]
    for i in range(len(model_ls)):
        val_loss_ls = []
        with open(txt_path, 'a') as file0:
            print(model_ls[i], file=file0)

        Model.SetSeed(seed)
        train_data, val_data, test_data = joblib.load(data_path)
        train_data.cell_ft, train_data.drug_ft = cell_ft_ls[i], drug_ft_ls[i]
        val_data.cell_ft, val_data.drug_ft = cell_ft_ls[i], drug_ft_ls[i]
        test_data.cell_ft, test_data.drug_ft = cell_ft_ls[i], drug_ft_ls[i]
        train_loader = Data.DrDataLoader(Data.DrDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = Data.DrDataLoader(Data.DrDataset(val_data), batch_size=batch_size, shuffle=False)
        test_loader = Data.DrDataLoader(Data.DrDataset(test_data), batch_size=batch_size, shuffle=False)

        for j in range(len(lr_ls)):
            Model.SetSeed(seed)
            model = Model.DrModel(cell_encoder_ls[i], drug_encoder_ls[i], fusion_module_ls[i])
            if not os.path.exists('./models_tuning_{}'.format(lab)):
                os.makedirs('./models_tuning_{}'.format(lab))
            model_path = './models_tuning_{}/model={}_lr={}_seed={}.pkl'.format(lab, model_ls[i], str(lr_ls[j]), seed)
            ce, de, fm, _, _, val_loss, _ = Model.Train(model, train_loader=train_loader, val_loader=val_loader,
                                                        epochs=100, lr=lr_ls[j], ratio=[0.4, 1, 1],
                                                        save_path_model=model_path, early_stop='val')
            val_loss_ls.append(min(val_loss[50:]))
            with open(txt_path, 'a') as file0:
                print(lr_ls[j], val_loss.index(min(val_loss[50:])), min(val_loss[50:]), file=file0)

        lr = lr_ls[val_loss_ls.index(min(val_loss_ls))]
        with open(txt_path, 'a') as file0:
            print('best lr: ', lr, file=file0)

        Model.SetSeed(seed)
        model = Model.DrModel(cell_encoder_ls[i], drug_encoder_ls[i], fusion_module_ls[i])
        if not os.path.exists('./models_{}'.format(lab)):
            os.makedirs('./models_{}'.format(lab))
        pre_path = './models_{}/model={}_lr={}_seed={}.csv'.format(lab, model_ls[i], str(lr), seed)
        model_path = './models_{}/model={}_lr={}_seed={}.pkl'.format(lab, model_ls[i], str(lr), seed)
        ce, de, fm, _, _, _, test_loss = Model.Train(model, train_loader=train_loader, test_loader=test_loader,
                                                     epochs=100, lr=lr, ratio=[0.4, 1, 1], early_stop='test',
                                                     save_path_prediction=pre_path, save_path_model=model_path)
        with open(txt_path, 'a') as file0:
            print(test_loss.index(min(test_loss[50:])), min(test_loss[50:]), file=file0)
