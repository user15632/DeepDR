import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

path = './models_CCLE_cell_out/'
file_ls = os.listdir(path)

model_ls = ['DNN(EXP)_DNN(ECFP)_DNN',
            'DNN(PES)_DNN(ECFP)_DNN',
            'DNN(MUT)_DNN(ECFP)_DNN',
            'DNN(CNV)_DNN(ECFP)_DNN',
            'DNN(EXP_ALL)_DNN(ECFP)_DNN',
            'DNN(MUT_ALL)_DNN(ECFP)_DNN',
            'DNN(CNV_ALL)_DNN(ECFP)_DNN',
            'CNN(EXP)_DNN(ECFP)_DNN',
            'DAE(EXP)_DNN(ECFP)_DNN',
            'DAE(EXP)_CNN(SMILES)_DNN',
            'DAE(EXP)_AFP(Graph)_DNN',
            'DAE(EXP)_MPG(Graph)_DNN',
            'DAE(EXP)_MPG(Graph)_MHA',
            'tCNNS',
            'Precily',
            'DeepDSC']

print(len(model_ls))

for each in model_ls:
    ffls = []
    for f in file_ls:
        if f[-3:] == 'csv' and each in f:
            ffls.append(f)
    mses = []
    pearsons = []
    r2s = []
    for f in ffls:
        csv = pd.read_csv(path + f, header=0, sep=',', dtype=str)
        real = np.array([float(each) for each in list(csv.iloc[:, 2])])
        p = np.array(csv.iloc[:, 3:], dtype=float)
        mse_ls = np.mean((np.expand_dims(real, 1) - p) ** 2, axis=0).tolist()
        idx = mse_ls.index(min(mse_ls[50:]))

        p = np.array([float(each) for each in list(csv.iloc[:, idx + 3])])
        mses.append(np.mean((real - p) ** 2))
        pearsons.append(np.corrcoef(real, p)[0][1])
        r2s.append(r2_score(real, p))
    mses = np.array(mses)
    pearsons = np.array(pearsons)
    r2s = np.array(r2s)
    mse = np.mean(mses)
    mse_std = np.std(mses)
    pearson = np.mean(pearsons)
    pearson_std = np.std(pearsons)
    r2 = np.mean(r2s)
    r2_std = np.std(r2s)
    print(each, ' ', len(ffls))
    print('{:.4f} ({:.4f}) & {:.4f} ({:.4f}) & {:.4f} ({:.4f})'.format(float(mse), float(mse_std),
                                                                       float(pearson), float(pearson_std),
                                                                       float(r2), float(r2_std)))
