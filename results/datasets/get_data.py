from DeepDR import Data
import joblib

seeds = [1, 10, 100]

for seed in seeds:
    data = Data.Read('CCLE', response='ActArea', cell_ft=None, drug_ft=None, clean=True,
                     cell_dict_for_clean=['GDSC_EXP.pkl', 'GDSC_PES.pkl', 'GDSC_MUT.pkl', 'GDSC_CNV.pkl',
                                          'VAE_dict.pkl'])
    train_data, val_data, test_data = Data.Split(data, mode='cell_out', ratio=[0.8, 0.1, 0.1], seed=seed, save=False)
    joblib.dump((train_data, val_data, test_data), 'CCLE_ActArea_cell_out_seed={}.pkl'.format(seed))

for seed in seeds:
    data = Data.Read('GDSC2', response='IC50', cell_ft=None, drug_ft=None, clean=True,
                     cell_dict_for_clean=['GDSC_EXP.pkl', 'GDSC_PES.pkl', 'GDSC_MUT.pkl', 'GDSC_CNV.pkl',
                                          'VAE_dict.pkl'])
    train_data, val_data, test_data = Data.Split(data, mode='cell_out', ratio=[0.8, 0.1, 0.1], seed=seed, save=False)
    joblib.dump((train_data, val_data, test_data), 'GDSC2_IC50_cell_out_seed={}.pkl'.format(seed))

for seed in seeds:
    data = Data.Read('CCLE', response='ActArea', cell_ft=None, drug_ft=None, clean=True,
                     cell_dict_for_clean=['GDSC_EXP.pkl', 'GDSC_PES.pkl', 'GDSC_MUT.pkl', 'GDSC_CNV.pkl',
                                          'VAE_dict.pkl'])
    train_data, val_data, test_data = Data.Split(data, mode='drug_out', ratio=[0.8, 0.1, 0.1], seed=seed, save=False)
    joblib.dump((train_data, val_data, test_data), 'CCLE_ActArea_drug_out_seed={}.pkl'.format(seed))

for seed in seeds:
    data = Data.Read('GDSC2', response='IC50', cell_ft=None, drug_ft=None, clean=True,
                     cell_dict_for_clean=['GDSC_EXP.pkl', 'GDSC_PES.pkl', 'GDSC_MUT.pkl', 'GDSC_CNV.pkl',
                                          'VAE_dict.pkl'])
    train_data, val_data, test_data = Data.Split(data, mode='drug_out', ratio=[0.8, 0.1, 0.1], seed=seed, save=False)
    joblib.dump((train_data, val_data, test_data), 'GDSC2_IC50_drug_out_seed={}.pkl'.format(seed))
