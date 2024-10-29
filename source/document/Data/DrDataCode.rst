Data.DrData
===========================

`Click here </en/latest/document/Data/DrData.html>`_ to go back to the reference.


.. code-block:: python

    class DrData:
    """"""

        def __init__(self, pair_ls: list,
                     cell_ft: str or dict,
                     drug_ft: str or dict,
                     smiles_dict: dict = None,
                     mpg_dict: dict = None):

            if type(cell_ft) == str:
                assert cell_ft in ['EXP', 'PES', 'MUT', 'CNV']
            if type(drug_ft) == str:
                assert drug_ft in ['ECFP', 'SMILES', 'Graph', 'Image']

            self.pair_ls = pair_ls
            self.cell_ft = cell_ft
            self.drug_ft = drug_ft
            self.smiles_dict = smiles_dict
            self.mpg_dict = mpg_dict

        def __len__(self):
            return len(self.pair_ls)

        def clean(self, cell_ft_ls: list = None):
            if cell_ft_ls is None:
                cell_ft_ls = [self.cell_ft] if type(self.cell_ft) == dict else ['GDSC_{}.pkl'.format(self.cell_ft)]
            else:
                cell_ft_ls += [self.cell_ft] if type(self.cell_ft) == dict else ['GDSC_{}.pkl'.format(self.cell_ft)]
            smiles_dict = 'SMILES_dict.pkl' if self.smiles_dict is None else self.smiles_dict
            drug_dict_for_clean = self.drug_ft if type(self.drug_ft) == dict else smiles_dict
            pair_ls = self.pair_ls
            for each in cell_ft_ls:
                pair_ls = _Clean(pair_ls, each, drug_dict_for_clean)
            return DrData(pair_ls, self.cell_ft, self.drug_ft, self.smiles_dict, self.mpg_dict)

        def split(self, mode: str,
                  fold: int,
                  ratio: list,
                  seed: int,
                  save: bool = True,
                  save_path: str = None):

            return _Split(self, mode, fold, ratio, seed, save, save_path)
