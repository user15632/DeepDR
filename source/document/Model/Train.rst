Model.Train
===========================

`Click here </en/latest/document/Model/TrainCode.html>`_ to view source code.


.. code-block:: python

    def Train(model, epochs: int, lr: float, train_loader, val_loader=None, test_loader=None,
              loss_func=None, optimizer=None, ratio: list = None, classify: bool = False,
              save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
              no_wandb: bool = True, project=None, name=None, config=None, early_stop=None):


It can be used to train the drug response prediction model.

**PARAMETERS:**

* **model** - The model built by ``Model.DrModel``.
* **epochs** *(int)* - The number of epochs.
* **lr** *(float)* - The learning rate.

* **train_loader** - The train loader got by ``Data.DrDataLoader``.
* **val_loader** *(optional)* - The val loader got by ``Data.DrDataLoader``. *(default: None)*
* **test_loader** *(optional)* - The test loader got by ``Data.DrDataLoader``. *(default: None)*

* **loss_func** *(optional)* - The loss function. When the value is ``None``, ``torch.nn.MSELoss`` is used if ``classify=False``, and ``torch.nn.BCEWithLogitsLoss()`` is used if ``classify=True``. *(default: None)*

* **optimizer** *(optional)* - The optimizer. The ``torch.optim.Adam`` is used when the value is ``None``.

* **ratio** *(list, optional)* - The learning rate ratio of cell encoder, drug encoder and fusion module. The ``[1, 1, 1]`` is used when the value is ``None``. *(default: None)*

* **classify** *(bool, optional)* - Whether classification task. *(default: False)*

* **save_path_prediction** *(str, optional)* - Save path of predictions. It is expected to end in ``".csv"``.

* **save_path_model** *(str, optional)* - Save path of trained model. It is expected to end in ``".pkl"``.

* **save_path_log** *(str, optional)* - Save path of training log. It is expected to end in ``".txt"``.

* **no_wandb** *(bool, optional)* - Whether not to use wandb. *(default: True)*

* **project** *(optional)* - Parameter of wandb. *(default: None)*
* **name** *(optional)* - Parameter of wandb. *(default: None)*
* **config** *(optional)* - Parameter of wandb. *(default: None)*

* **early_stop** *(optional)* - Early stop detection dataset. ``"val"``, ``"test"``, or ``None`` is available.

**OUTPUTS:**

* **model** - The trained model.
* **loss_func** - The loss function.
* **optimizer** - The optimizer.
* **val_epoch_loss_ls** *(list)* - The value of loss on validation set.
* **test_epoch_loss_ls** *(list)* - The value of loss on test set.
