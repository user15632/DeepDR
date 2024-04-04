import time
import wandb
import torch
import joblib
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def SetSeed(seed: int):
    """"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _TrainingTP(cell_encoder, drug_encoder, fusion_module, data_loader, loss_func, optimizer):
    """"""
    cell_encoder.train()
    drug_encoder.train()
    fusion_module.train()
    real = []
    pre = []
    cell_ls = []
    drug_ls = []
    epoch_loss = 0
    length = 0
    for it, (cell_ft, drug_ft, response, cell_name, drug_name) in enumerate(data_loader):
        cell_ls += cell_name
        drug_ls += drug_name
        cell_ft, drug_ft, response = cell_ft.to(device), drug_ft.to(device), response.to(device)
        prediction, cell_ft, drug_ft = fusion_module(cell_encoder(cell_ft), drug_encoder(drug_ft))
        loss = loss_func(prediction, response)
        real += torch.squeeze(response).cpu().tolist()
        pre += torch.squeeze(prediction).cpu().tolist()
        response = torch.squeeze(response).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        length += len(response)
        epoch_loss += loss.detach().item() * len(response)
    epoch_loss /= length
    return cell_encoder, drug_encoder, fusion_module, optimizer, epoch_loss, real, pre, cell_ls, drug_ls


def _InferenceTP(cell_encoder, drug_encoder, fusion_module, data_loader, loss_func):
    """"""
    cell_encoder.eval()
    drug_encoder.eval()
    fusion_module.eval()
    with torch.no_grad():
        real = []
        pre = []
        cell_ls = []
        drug_ls = []
        epoch_loss = 0
        length = 0
        for it, (cell_ft, drug_ft, response, cell_name, drug_name) in enumerate(data_loader):
            cell_ls += cell_name
            drug_ls += drug_name
            cell_ft, drug_ft, response = cell_ft.to(device), drug_ft.to(device), response.to(device)
            prediction, cell_ft, drug_ft = fusion_module(cell_encoder(cell_ft), drug_encoder(drug_ft))
            loss = loss_func(prediction, response)
            real += torch.squeeze(response).cpu().tolist()
            pre += torch.squeeze(prediction).cpu().tolist()
            response = torch.squeeze(response).cpu().tolist()
            length += len(response)
            epoch_loss += loss.detach().item() * len(response)
        epoch_loss /= length
    return epoch_loss, real, pre, cell_ls, drug_ls


def _TrainTP(model_tp, train_loader, val_loader=None, test_loader=None, loss_func=None, optimizer=None,
             epochs: int = 100, lr: float = 1e-4, ratio: list = None, classify: bool = False,
             save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
             no_wandb: bool = True, project=None, name=None, config=None):
    """"""
    if save_path_prediction is not None:
        assert save_path_prediction[-4:] == '.csv'
    if save_path_model is not None:
        assert save_path_model[-4:] == '.pkl'
    if save_path_log is not None:
        assert save_path_log[-4:] == '.txt'
    if ratio is None:
        ratio = [1, 1, 1]
    assert len(ratio) == 3
    cell_encoder, drug_encoder, fusion_module = model_tp
    cell_encoder = cell_encoder.to(device)
    drug_encoder = drug_encoder.to(device)
    fusion_module = fusion_module.to(device)
    if loss_func is None:
        if not classify:
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
    if optimizer is None:
        params = [
            {'params': cell_encoder.parameters(), 'lr': ratio[0] * lr},
            {'params': drug_encoder.parameters(), 'lr': ratio[1] * lr},
            {'params': fusion_module.parameters(), 'lr': ratio[2] * lr}
        ]
        optimizer = optim.Adam(params, lr=lr)
    real_pre_test_dict = dict()
    real_pre_val_dict = dict()
    val_epoch_loss = None
    if not no_wandb:
        wandb.init(
            project=project,
            name=name,
            config=config,
            mode="disabled" if no_wandb else "online"
        )
    print('Start training!')
    for epoch in range(epochs):
        start = time.time()
        loss_dict = dict()

        cell_encoder, drug_encoder, fusion_module, optimizer, epoch_loss, real, pre, cell_ls, drug_ls = _TrainingTP(cell_encoder, drug_encoder, fusion_module, train_loader, loss_func, optimizer)
        print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), end='  ')
        if save_path_log is not None:
            with open(save_path_log, 'a') as file:
                print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), end='  ', file=file)
        loss_dict['train_loss'] = epoch_loss

        if val_loader is not None:
            epoch_loss, real, pre, cell_ls, drug_ls = _InferenceTP(cell_encoder, drug_encoder, fusion_module, val_loader, loss_func)
            print('val loss {:.6f}'.format(epoch_loss), end='  ')
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('val loss {:.6f}'.format(epoch_loss), end='  ', file=file)
            val_epoch_loss = epoch_loss
            if save_path_prediction is not None:
                if epoch == 0:
                    real_pre_val_dict['cell'] = cell_ls
                    real_pre_val_dict['drug'] = drug_ls
                    real_pre_val_dict['real'] = real
                real_pre_val_dict['epoch_{}'.format(epoch)] = pre
                dataframe = pd.DataFrame(real_pre_val_dict)
                dataframe.to_csv(save_path_prediction[:-4] + '_val' + save_path_prediction[-4:], index=False, sep=',')
            loss_dict['val_loss'] = epoch_loss

        if test_loader is not None:
            epoch_loss, real, pre, cell_ls, drug_ls = _InferenceTP(cell_encoder, drug_encoder, fusion_module, test_loader, loss_func)
            print('test loss {:.6f}'.format(epoch_loss), end='  ')
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('test loss {:.6f}'.format(epoch_loss), end='  ', file=file)
            if save_path_prediction is not None:
                if epoch == 0:
                    real_pre_test_dict['cell'] = cell_ls
                    real_pre_test_dict['drug'] = drug_ls
                    real_pre_test_dict['real'] = real
                real_pre_test_dict['epoch_{}'.format(epoch)] = pre
                dataframe = pd.DataFrame(real_pre_test_dict)
                dataframe.to_csv(save_path_prediction[:-4] + '_test' + save_path_prediction[-4:], index=False, sep=',')
            loss_dict['test_loss'] = epoch_loss

        if not no_wandb:
            wandb.log(loss_dict)

        end = time.time()
        print('time consumed {:.6f}'.format(end - start))
        if save_path_log is not None:
            with open(save_path_log, 'a') as file:
                print('time consumed {:.6f}'.format(end - start), file=file)

    if not no_wandb:
        wandb.finish()

    if save_path_model is not None:
        joblib.dump((cell_encoder, drug_encoder, fusion_module, loss_func, optimizer), save_path_model)
    print('Training completed!')
    return cell_encoder, drug_encoder, fusion_module, loss_func, optimizer, val_epoch_loss


def _Training(model, data_loader, loss_func, optimizer):
    """"""
    model.train()
    real = []
    pre = []
    cell_ls = []
    drug_ls = []
    epoch_loss = 0
    length = 0
    for it, (cell_ft, drug_ft, response, cell_name, drug_name) in enumerate(data_loader):
        cell_ls += cell_name
        drug_ls += drug_name
        cell_ft, drug_ft, response = cell_ft.to(device), drug_ft.to(device), response.to(device)
        prediction, cell_ft, drug_ft = model(cell_ft, drug_ft)
        loss = loss_func(prediction, response)
        real += torch.squeeze(response).cpu().tolist()
        pre += torch.squeeze(prediction).cpu().tolist()
        response = torch.squeeze(response).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        length += len(response)
        epoch_loss += loss.detach().item() * len(response)
    epoch_loss /= length
    return model, optimizer, epoch_loss, real, pre, cell_ls, drug_ls


def _Inference(model, data_loader, loss_func):
    """"""
    model.eval()
    with torch.no_grad():
        real = []
        pre = []
        cell_ls = []
        drug_ls = []
        epoch_loss = 0
        length = 0
        for it, (cell_ft, drug_ft, response, cell_name, drug_name) in enumerate(data_loader):
            cell_ls += cell_name
            drug_ls += drug_name
            cell_ft, drug_ft, response = cell_ft.to(device), drug_ft.to(device), response.to(device)
            prediction, cell_ft, drug_ft = model(cell_ft, drug_ft)
            loss = loss_func(prediction, response)
            real += torch.squeeze(response).cpu().tolist()
            pre += torch.squeeze(prediction).cpu().tolist()
            response = torch.squeeze(response).cpu().tolist()
            length += len(response)
            epoch_loss += loss.detach().item() * len(response)
        epoch_loss /= length
    return epoch_loss, real, pre, cell_ls, drug_ls


def _Train(model, train_loader, val_loader=None, test_loader=None, loss_func=None, optimizer=None, epochs: int = 100,
           lr: float = 1e-4, classify: bool = False, save_path_prediction: str = None, save_path_model: str = None,
           save_path_log: str = None, no_wandb: bool = True, project=None, name=None, config=None):
    """"""
    if save_path_prediction is not None:
        assert save_path_prediction[-4:] == '.csv'
    if save_path_model is not None:
        assert save_path_model[-4:] == '.pkl'
    if save_path_log is not None:
        assert save_path_log[-4:] == '.txt'
    model = model.to(device)
    if loss_func is None:
        if not classify:
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
    if optimizer is None:
        params = [
            {'params': model.parameters(), 'lr': lr}
        ]
        optimizer = optim.Adam(params, lr=lr)
    real_pre_test_dict = dict()
    real_pre_val_dict = dict()
    val_epoch_loss = None
    if not no_wandb:
        wandb.init(
            project=project,
            name=name,
            config=config,
            mode="disabled" if no_wandb else "online"
        )
    print('Start training!')
    for epoch in range(epochs):
        start = time.time()
        loss_dict = dict()

        model, optimizer, epoch_loss, real, pre, cell_ls, drug_ls = _Training(model, train_loader, loss_func, optimizer)
        print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), end='  ')
        if save_path_log is not None:
            with open(save_path_log, 'a') as file:
                print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), end='  ', file=file)
        loss_dict['train_loss'] = epoch_loss

        if val_loader is not None:
            epoch_loss, real, pre, cell_ls, drug_ls = _Inference(model, val_loader, loss_func)
            print('val loss {:.6f}'.format(epoch_loss), end='  ')
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('val loss {:.6f}'.format(epoch_loss), end='  ', file=file)
            val_epoch_loss = epoch_loss
            if save_path_prediction is not None:
                if epoch == 0:
                    real_pre_val_dict['cell'] = cell_ls
                    real_pre_val_dict['drug'] = drug_ls
                    real_pre_val_dict['real'] = real
                real_pre_val_dict['epoch_{}'.format(epoch)] = pre
                dataframe = pd.DataFrame(real_pre_val_dict)
                dataframe.to_csv(save_path_prediction[:-4] + '_val' + save_path_prediction[-4:], index=False, sep=',')
            loss_dict['val_loss'] = epoch_loss

        if test_loader is not None:
            epoch_loss, real, pre, cell_ls, drug_ls = _Inference(model, test_loader, loss_func)
            print('test loss {:.6f}'.format(epoch_loss), end='  ')
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('test loss {:.6f}'.format(epoch_loss), end='  ', file=file)
            if save_path_prediction is not None:
                if epoch == 0:
                    real_pre_test_dict['cell'] = cell_ls
                    real_pre_test_dict['drug'] = drug_ls
                    real_pre_test_dict['real'] = real
                real_pre_test_dict['epoch_{}'.format(epoch)] = pre
                dataframe = pd.DataFrame(real_pre_test_dict)
                dataframe.to_csv(save_path_prediction[:-4] + '_test' + save_path_prediction[-4:], index=False, sep=',')
            loss_dict['test_loss'] = epoch_loss

        if not no_wandb:
            wandb.log(loss_dict)

        end = time.time()
        print('time consumed {:.6f}'.format(end - start))
        if save_path_log is not None:
            with open(save_path_log, 'a') as file:
                print('time consumed {:.6f}'.format(end - start), file=file)

    if not no_wandb:
        wandb.finish()

    if save_path_model is not None:
        joblib.dump((model, loss_func, optimizer), save_path_model)
    print('Training completed!')
    return model, loss_func, optimizer, val_epoch_loss


def Train(model, train_loader, val_loader=None, test_loader=None, loss_func=None, optimizer=None,
          epochs: int = 100, lr: float = 1e-4, ratio: list = None, classify: bool = False,
          save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
          no_wandb: bool = True, project=None, name=None, config=None):
    if type(model) == tuple:
        cell_encoder, drug_encoder, fusion_module, loss_func, optimizer, val_epoch_loss = _TrainTP(model, train_loader, val_loader, test_loader, loss_func, optimizer, epochs, lr, ratio, classify, save_path_prediction, save_path_model, save_path_log, no_wandb, project, name, config)
        return cell_encoder, drug_encoder, fusion_module, loss_func, optimizer, val_epoch_loss
    else:
        model, loss_func, optimizer, val_epoch_loss = _Train(model, train_loader, val_loader, test_loader, loss_func, optimizer, epochs, lr, classify, save_path_prediction, save_path_model, save_path_log, no_wandb, project, name, config)
        return model, loss_func, optimizer, val_epoch_loss
