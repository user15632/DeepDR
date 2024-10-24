Data.DrDataset
===========================

`Click here </document/Data/DrDataset.html>`_ to go back to the reference.


.. code-block:: python

    class DrDataset(Dataset, ABC):
        """"""

        def __init__(self, dr_data: DrData,
                     radius: int = 2,
                     nBits: int = 512,
                     max_len: int = 230,
                     char_dict: dict = None,
                     right: bool = True,
                     mpg: bool = True):

            super().__init__()
            self._pair_ls = dr_data.pair_ls
            self._cell_ft = dr_data.cell_ft
            self._drug_ft = dr_data.drug_ft
            self._radius = radius
            self._nBits = nBits
            self._max_len = max_len
            self._char_dict = char_dict
            self._right = right
            self._SMILES_dict = dr_data.smiles_dict
            self._MPG_dict = dr_data.mpg_dict
            self._MPG = mpg
            self._data = DrDataset.preprocess(self)

        def __getitem__(self, idx):
            data = self._data[idx]
            return data

        def __len__(self):
            return len(self._data)

        def preprocess(self):

            if type(self._cell_ft) == str:
                assert self._cell_ft in ['EXP', 'PES', 'MUT', 'CNV']
                cell_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/GDSC_{}.pkl'.format(self._cell_ft)))
            else:
                cell_dict = self._cell_ft

            if type(self._drug_ft) == str:
                assert self._drug_ft in ['ECFP', 'SMILES', 'Graph', 'Image']
                if self._SMILES_dict is None:
                    drug_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/SMILES_dict.pkl'))
                else:
                    drug_dict = self._SMILES_dict
            else:
                drug_dict = self._drug_ft

            if self._MPG_dict is None:
                self._MPG_dict = joblib.load(os.path.join(os.path.split(__file__)[0], 'DefaultData/MPG_dict.pkl'))

            data = []
            for i in tqdm(range(len(self._pair_ls))):
                each_pair = self._pair_ls[i]
                if type(cell_dict[each_pair[0]]) == torch.Tensor:
                    cell_ft = cell_dict[each_pair[0]]
                else:
                    cell_ft = torch.tensor(cell_dict[each_pair[0]], dtype=torch.float32)
                cell_ft = (cell_ft - cell_ft.mean()) / cell_ft.std(dim=0)

                if self._drug_ft == 'ECFP':
                    drug_ft = PreEcfp(drug_dict[each_pair[1]], self._radius, self._nBits)
                elif self._drug_ft == 'SMILES':
                    drug_ft = PreSmiles(drug_dict[each_pair[1]], self._max_len, self._char_dict, self._right)
                elif self._drug_ft == 'Graph':
                    drug_ft = PreGraph(drug_dict[each_pair[1]])
                    if self._MPG:
                        try:
                            drug_ft = _Add_seg_id(_Self_loop(Data(x=drug_ft.x, edge_index=drug_ft.edge_index,
                                                                  edge_attr=drug_ft.edge_attr,
                                                                  mpg_ft=self._MPG_dict[each_pair[1]])))
                        except KeyError:
                            print('MPG feature missing! Set MPG=False or run Data.DrRead.FeatDrug')
                elif self._drug_ft == 'Image':
                    drug_ft = ImageDataset([drug_dict[each_pair[1]]])[0]
                else:
                    if type(drug_dict[each_pair[1]]) == torch.Tensor:
                        drug_ft = drug_dict[each_pair[1]]
                    else:
                        drug_ft = torch.tensor(drug_dict[each_pair[1]], dtype=torch.float32)

                data.append(Data(cell_ft=cell_ft, drug_ft=drug_ft,
                                 response=torch.tensor([each_pair[2]], dtype=torch.float32),
                                 cell_name=each_pair[0], drug_name=each_pair[1]))
            return data
