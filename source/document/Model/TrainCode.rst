Model.Train
===========================

`Click here </document/Model/Train.html>`_ to go back to the reference.


.. code-block:: python

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


    def _TrainTP(model_tp, epochs: int, lr: float, train_loader, val_loader=None, test_loader=None,
                 loss_func=None, optimizer=None, ratio: list = None, classify: bool = False,
                 save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
                 no_wandb: bool = True, project=None, name=None, config=None, early_stop=None):
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
        assert early_stop in ['val', 'test', None]
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
        real_pre_val_dict = dict()
        real_pre_test_dict = dict()
        val_epoch_loss_ls = []
        test_epoch_loss_ls = []
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
            if not classify:
                print('Epoch {}, train loss {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch, epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
            else:
                print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss))
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    if not classify:
                        print('Epoch {}, train loss {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch, epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                    else:
                        print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), file=file)
            loss_dict['train_loss'] = epoch_loss

            if val_loader is not None:
                epoch_loss, real, pre, cell_ls, drug_ls = _InferenceTP(cell_encoder, drug_encoder, fusion_module, val_loader, loss_func)
                if not classify:
                    print('         val loss   {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
                else:
                    print('         val loss   {:.6f}'.format(epoch_loss))
                if save_path_log is not None:
                    with open(save_path_log, 'a') as file:
                        if not classify:
                            print('         val loss   {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                        else:
                            print('         val loss   {:.6f}'.format(epoch_loss), file=file)
                val_epoch_loss_ls.append(epoch_loss)
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
                if not classify:
                    print('         test loss  {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
                else:
                    print('         test loss  {:.6f}'.format(epoch_loss))
                if save_path_log is not None:
                    with open(save_path_log, 'a') as file:
                        if not classify:
                            print('         test loss  {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                        else:
                            print('         test loss  {:.6f}'.format(epoch_loss), file=file)
                test_epoch_loss_ls.append(epoch_loss)
                if save_path_prediction is not None:
                    if epoch == 0:
                        real_pre_test_dict['cell'] = cell_ls
                        real_pre_test_dict['drug'] = drug_ls
                        real_pre_test_dict['real'] = real
                    real_pre_test_dict['epoch_{}'.format(epoch)] = pre
                    dataframe = pd.DataFrame(real_pre_test_dict)
                    dataframe.to_csv(save_path_prediction[:-4] + '_test' + save_path_prediction[-4:], index=False, sep=',')
                loss_dict['test_loss'] = epoch_loss

            if epoch >= (epochs // 2):
                if early_stop == 'val' and val_epoch_loss_ls[-1] == min(val_epoch_loss_ls[epochs // 2:]):
                    trained_model = (cell_encoder.cpu(), drug_encoder.cpu(), fusion_module.cpu())
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    cell_encoder = cell_encoder.to(device)
                    drug_encoder = drug_encoder.to(device)
                    fusion_module = fusion_module.to(device)
                elif early_stop == 'test' and test_epoch_loss_ls[-1] == min(test_epoch_loss_ls[epochs // 2:]):
                    trained_model = (cell_encoder.cpu(), drug_encoder.cpu(), fusion_module.cpu())
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    cell_encoder = cell_encoder.to(device)
                    drug_encoder = drug_encoder.to(device)
                    fusion_module = fusion_module.to(device)
                else:
                    trained_model = (cell_encoder.cpu(), drug_encoder.cpu(), fusion_module.cpu())
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    cell_encoder = cell_encoder.to(device)
                    drug_encoder = drug_encoder.to(device)
                    fusion_module = fusion_module.to(device)

            if not no_wandb:
                wandb.log(loss_dict)

            end = time.time()
            print('         time consumed {:.6f}'.format(end - start))
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('         time consumed {:.6f}'.format(end - start), file=file)

        if not no_wandb:
            wandb.finish()
        print('Training completed!')
        return trained_model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls


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


    def _Train(model, epochs: int, lr: float, train_loader, val_loader=None, test_loader=None,
               loss_func=None, optimizer=None, classify: bool = False,
               save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
               no_wandb: bool = True, project=None, name=None, config=None, early_stop=None):
        """"""
        if save_path_prediction is not None:
            assert save_path_prediction[-4:] == '.csv'
        if save_path_model is not None:
            assert save_path_model[-4:] == '.pkl'
        if save_path_log is not None:
            assert save_path_log[-4:] == '.txt'
        assert early_stop in ['val', 'test', None]
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
        real_pre_val_dict = dict()
        real_pre_test_dict = dict()
        val_epoch_loss_ls = []
        test_epoch_loss_ls = []
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
            if not classify:
                print('Epoch {}, train loss {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch, epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
            else:
                print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss))
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    if not classify:
                        print('Epoch {}, train loss {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch, epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                    else:
                        print('Epoch {}, train loss {:.6f}'.format(epoch, epoch_loss), file=file)
            loss_dict['train_loss'] = epoch_loss

            if val_loader is not None:
                epoch_loss, real, pre, cell_ls, drug_ls = _Inference(model, val_loader, loss_func)
                if not classify:
                    print('         val loss   {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
                else:
                    print('         val loss   {:.6f}'.format(epoch_loss))
                if save_path_log is not None:
                    with open(save_path_log, 'a') as file:
                        if not classify:
                            print('         val loss   {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                        else:
                            print('         val loss   {:.6f}'.format(epoch_loss), file=file)
                val_epoch_loss_ls.append(epoch_loss)
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
                if not classify:
                    print('         test loss  {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)))
                else:
                    print('         test loss  {:.6f}'.format(epoch_loss))
                if save_path_log is not None:
                    with open(save_path_log, 'a') as file:
                        if not classify:
                            print('         test loss  {:.6f}  r2 {:.4f}  pcc {:.4f}'.format(epoch_loss, Metric.R2(real, pre), Metric.PCC(real, pre)), file=file)
                        else:
                            print('         test loss  {:.6f}'.format(epoch_loss), file=file)
                test_epoch_loss_ls.append(epoch_loss)
                if save_path_prediction is not None:
                    if epoch == 0:
                        real_pre_test_dict['cell'] = cell_ls
                        real_pre_test_dict['drug'] = drug_ls
                        real_pre_test_dict['real'] = real
                    real_pre_test_dict['epoch_{}'.format(epoch)] = pre
                    dataframe = pd.DataFrame(real_pre_test_dict)
                    dataframe.to_csv(save_path_prediction[:-4] + '_test' + save_path_prediction[-4:], index=False, sep=',')
                loss_dict['test_loss'] = epoch_loss

            if epoch >= (epochs // 2):
                if early_stop == 'val' and val_epoch_loss_ls[-1] == min(val_epoch_loss_ls[epochs // 2:]):
                    trained_model = model.cpu()
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    model = model.to(device)
                elif early_stop == 'test' and test_epoch_loss_ls[-1] == min(test_epoch_loss_ls[epochs // 2:]):
                    trained_model = model.cpu()
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    model = model.to(device)
                else:
                    trained_model = model.cpu()
                    if save_path_model is not None:
                        joblib.dump(trained_model, save_path_model)
                    model = model.to(device)

            if not no_wandb:
                wandb.log(loss_dict)

            end = time.time()
            print('         time consumed {:.6f}'.format(end - start))
            if save_path_log is not None:
                with open(save_path_log, 'a') as file:
                    print('         time consumed {:.6f}'.format(end - start), file=file)

        if not no_wandb:
            wandb.finish()
        print('Training completed!')
        return trained_model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls


    def Train(model, epochs: int, lr: float, train_loader, val_loader=None, test_loader=None,
              loss_func=None, optimizer=None, ratio: list = None, classify: bool = False,
              save_path_prediction: str = None, save_path_model: str = None, save_path_log: str = None,
              no_wandb: bool = True, project=None, name=None, config=None, early_stop=None):
        if type(model) == tuple:
            model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls = _TrainTP(model, epochs, lr, train_loader, val_loader, test_loader, loss_func, optimizer, ratio, classify, save_path_prediction, save_path_model, save_path_log, no_wandb, project, name, config, early_stop)
            return model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls
        else:
            model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls = _Train(model, epochs, lr, train_loader, val_loader, test_loader, loss_func, optimizer, classify, save_path_prediction, save_path_model, save_path_log, no_wandb, project, name, config, early_stop)
            return model, loss_func, optimizer, val_epoch_loss_ls, test_epoch_loss_ls
