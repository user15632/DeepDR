DeepDR.Model.Train
===========================



.. code-block:: python

    def Train(model, train_loader, val_loader=None, test_loader=None, loss_func=None, optimizer=None,
    epochs: int = 100, lr: float = 1e-4, ratio: list = None, classify: bool = False,
    save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
    no_wandb: bool = True, project=None, name=None, config=None)


**PARAMETERS:**

* **model** - The model built by DeepDR.Model.MDL or DeepDR.Model.SDL.

* **train_loader** - The train loader got by DeepDR.Data.DrDataLoader.
* **val_loader** *(optional)* - The val loader got by DeepDR.Data.DrDataLoader. *(default: None)*
* **test_loader** *(optional)* - The test loader got by DeepDR.Data.DrDataLoader. *(default: None)*

* **loss_func** *(optional)* - The loss function. *(default: None)*
    The torch.nn.MSELoss is used when the value is None and parameter classify is None.

* **optimizer** *(optional)* - The optimizer. *(default: None)*
    The torch.optim.Adam is used when the value is None.

* **epochs** *(int, optional)* - The number of epochs. *(default: 150)*
* **lr** *(float, optional)* - The learning rate. *(default: 1e-4)*

* **ratio** *(list, optional)* - The learning rate ratio of cell encoder, drug encoder and fusion module. *(default: None)*
    The [1, 1, 1] is used when the value is None.

* **classify** *(bool, optional)* - Whether classification task. *(default: False)*

* **save_path_prediction** *(str, optional)* - The path to save the prediction results. *(default: None)*
    The value is expected to end in '.csv'.

* **save_path_model** *(str, optional)* - The path to save the trained model. *(default: None)*
    The value is expected to end in '.pkl'.
    The trained model is saved in format (model, loss_func, optimizer).

* **save_path_log** *(str, optional)* - The path to save the training log. *(default: None)*
    The value is expected to end in '.txt'.

* **no_wandb** *(bool, optional)* - Whether not to use wandb. *(default: True)*

* **project** *(optional)* - Parameter of wandb. *(default: None)*
* **name** *(optional)* - Parameter of wandb. *(default: None)*
* **config** *(optional)* - Parameter of wandb. *(default: None)*

**OUTPUTS:**

* **cell_encoder** - The trained cell encoder.
* **drug_encoder** - The trained drug encoder.
* **fusion_module** - The trained fusion module.
* **loss_func** - The loss function.
* **optimizer** - The optimizer.
* **val_epoch_loss** *(float)* - The value of loss on validation set.

* **model** - The trained model.
* **loss_func** - The loss function.
* **optimizer** - The optimizer.
* **val_epoch_loss** *(float)* - The value of loss on validation set.
