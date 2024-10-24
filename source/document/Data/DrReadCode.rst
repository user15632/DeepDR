Data.DrRead
===========================

`Click here </document/Data/DrRead.html>`_ to go back to the reference.


.. code-block:: python

    class DrRead:
    """"""

        @staticmethod
        def PairCSV(csv_path: str):
            assert csv_path[-4:] == '.csv'
            index = [0, 1, 2]
            print('Start reading!')
            csv = pd.read_csv(csv_path, header=0, sep=',', dtype=str)
            Cell = [NormalizeName(each) for each in list(csv.iloc[:, index[0]])]
            Drug = list(csv.iloc[:, index[1]])
            Tag = [float(_) for _ in list(csv.iloc[:, index[2]])]
            pair_ls = [[Cell[i], Drug[i], Tag[i]] for i in range(len(Cell))]
            print('Reading completed!')
            return pair_ls

        @staticmethod
        def PairDef(dataset: str,
                    response: str):
            assert dataset in ['CCLE', 'GDSC1', 'GDSC2']
            assert response in ['ActArea', 'AUC', 'IC50']
            pair_ls = dataset + '_' + response
            assert pair_ls in ['CCLE_ActArea', 'CCLE_IC50', 'GDSC1_AUC', 'GDSC1_IC50', 'GDSC2_AUC', 'GDSC2_IC50']
            csv_path = os.path.join(os.path.split(__file__)[0], 'DefaultData/' + pair_ls + '.csv')
            return DrRead.PairCSV(csv_path)

        @staticmethod
        def FeatCell(csv_path: str,
                     subset: bool,
                     subset_path: str = None,
                     save_feat_path: str = None,
                     save_gene_path: str = None):

            return GetCellFeat(csv_path, subset, subset_path, save_feat_path, save_gene_path)

        @staticmethod
        def FeatDrug(csv_path: str,
                     MPG_path: str,
                     save_SMILES_path: str = None,
                     save_MPG_path: str = None):

            return GetDrugFeat(csv_path, MPG_path, save_SMILES_path, save_MPG_path)
