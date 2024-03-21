import time
import torch
import numpy as np
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from Codes.FeatureEngineeringUtils.btc_feature_engineering_utils import DefTrainTestValidationLoader, \
    LogTrainTestValidationLoader, ExTrainTestValidationLoader
from Codes.ModelingUtils.models import ConvFC, FullyConnectedNetwork, FullyLSTMNetwork, FullyGRUNetwork, \
    FullyRNNNetwork, ConvLSTMNet
from Codes.ModelingUtils.methods import QualifiedConvCandlesDataset, QualifiedWeightedConvCandlesDataset, train_epoch, \
    test_epoch, save_model_on_validation_improvement, load_model, eval_predict, eval_search, eval_search0, \
    fast_def_evaluate, defevaluate2, plot_stats_box2

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


class Pipeline:
    def __init__(self, model_name, model_type, batch_size, update_scaler=False, weighted=False, silent=False, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.raw_targets = ['target3', 'target', 'stop', 'open', 'high', 'low', 'close']
        self.batch_size = batch_size
        self.update_scaler = update_scaler
        self.weighted = weighted
        self.silent = silent

        self.log_sta = kwargs.get('log_stationary', False)
        self.l2_loss = kwargs.get('l2_loss', False)
        self.excluded = kwargs.get('excluded', False)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.cropped_dataset = None

        self.train_loader = None
        self.test_loader = None

        self.cwdn = None

        self.train_preds = None
        self.test_preds = None
        self.val_preds = None
        self.cropped_preds = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        if self.log_sta:
            self.data_loader = LogTrainTestValidationLoader(self.model_name, 'btc_15m', target='target3',
                                                            raw_targets=self.raw_targets, n_input_steps=21,
                                                            training_portion=0.75, test_portion=0.4,
                                                            update_scaler=self.update_scaler, silent=self.silent)
        elif self.excluded:
            self.data_loader = ExTrainTestValidationLoader(self.model_name, 'btc_15m', target='target3',
                                                           raw_targets=self.raw_targets, n_input_steps=21,
                                                           training_portion=0.75, test_portion=0.4,
                                                           update_scaler=self.update_scaler, silent=self.silent)
        else:
            self.data_loader = DefTrainTestValidationLoader(self.model_name, 'btc_15m', target='target3',
                                                            raw_targets=self.raw_targets, n_input_steps=21,
                                                            training_portion=0.75, test_portion=0.4,
                                                            update_scaler=self.update_scaler, silent=self.silent)

        train_features, train_labels = self.data_loader.get_reframed_train_data()
        val_features, val_labels = self.data_loader.get_reframed_val_data()
        test_features, test_labels = self.data_loader.get_reframed_test_data()

        self.input_dim_15_1 = np.shape(train_features)[1]
        self.input_dim_15_2 = np.shape(train_features)[2]

        if self.weighted:
            a = np.count_nonzero(train_labels.reshape(-1) < 0) / len(train_labels)
            b = np.count_nonzero(train_labels.reshape(-1) > 0) / len(train_labels)
            self.train_dataset = QualifiedWeightedConvCandlesDataset(train_features, train_labels.reshape(-1, 1), w0=b,
                                                                     w1=a)
            self.val_dataset = QualifiedWeightedConvCandlesDataset(val_features, val_labels.reshape(-1, 1), w0=b, w1=a)
            self.test_dataset = QualifiedWeightedConvCandlesDataset(test_features, test_labels.reshape(-1, 1))
        else:
            self.train_dataset = QualifiedConvCandlesDataset(train_features, train_labels.reshape(-1, 1))
            self.val_dataset = QualifiedConvCandlesDataset(val_features, val_labels.reshape(-1, 1))
            self.test_dataset = QualifiedConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def generate_model(self, seed=16734, silent=False):
        if seed is not None:
            torch.manual_seed(seed)

        if self.model_type == 'LSTM':
            self.cwdn = FullyLSTMNetwork(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        elif self.model_type == 'GRU':
            self.cwdn = FullyGRUNetwork(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        elif self.model_type == 'RNN':
            self.cwdn = FullyRNNNetwork(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        elif self.model_type == 'CNN':
            self.cwdn = ConvFC(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        elif self.model_type == 'MLP':
            self.cwdn = FullyConnectedNetwork(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        elif self.model_type == 'CNNLSTM':
            self.cwdn = ConvLSTMNet(self.input_dim_15_1, self.input_dim_15_2, device=self.device)
        else:
            raise Exception('This model_type is not implemented!')

        model_data = self.cwdn.to(self.device, dtype=torch.double)
        if silent is True:
            pass
        else:
            print(model_data)

    def train_model(self, lr=5e-4, weight_decay=2e-3, patience=3, just_load=False):
        self.train_preds = None
        self.test_preds = None
        self.val_preds = None
        if just_load is False:
            adam_optimizer2 = torch.optim.AdamW(self.cwdn.parameters(), lr=lr, weight_decay=weight_decay)

            if self.weighted:
                loss_fn1 = nn.MSELoss(reduction='none')
                loss_fn2 = nn.L1Loss(reduction='none')
                # loss_fn2 = self.custom_loss
            else:
                loss_fn1 = nn.MSELoss(reduction='mean')
                loss_fn2 = nn.L1Loss(reduction='mean')
                # loss_fn2 = self.custom_loss

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            if self.l2_loss:
                w1 = 1
                w2 = 0
            else:
                w1 = 0.7
                w2 = 0.3
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.cwdn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               w1=w1, w2=w2, optimizer=adam_optimizer2, weighted=self.weighted)
                total_loss_test = test_epoch(self.cwdn, self.device, self.val_loader, loss_fn1, loss_fn2, w1=w1,
                                             w2=w2, weighted=self.weighted)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.cwdn, adam_optimizer2, epoch,
                                                         f'{self.model_name}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.cwdn = load_model(f'Files//torch_models//{self.model_name}.pth', self.cwdn)

    def search_results(self):
        model = self.cwdn
        if self.train_preds is None:
            self.train_preds = eval_predict(model, self.train_dataset, self.device, self.silent)
        if self.val_preds is None:
            self.val_preds = eval_predict(model, self.val_dataset, self.device, self.silent)
        if self.test_preds is None:
            self.test_preds = eval_predict(model, self.test_dataset, self.device, self.silent)

        if not self.silent:
            print('best of the train_val')
            sep = eval_search(self.train_dataset.targets, self.data_loader.y2_train.copy(),
                              self.train_preds.reshape(-1, 1),
                              self.val_dataset.targets, self.data_loader.y2_val.copy(),
                              self.val_preds.reshape(-1, 1))

            print('best of the val_test')
            _ = eval_search(self.val_dataset.targets, self.data_loader.y2_val.copy(),
                            self.val_preds.reshape(-1, 1),
                            self.test_dataset.targets, self.data_loader.y2_test.copy(),
                            self.test_preds.reshape(-1, 1))
            return sep
        else:
            sep, _ = eval_search0(self.train_dataset.targets, self.data_loader.y2_train.copy(),
                                  self.train_preds.reshape(-1, 1))
            return sep

    def fast_evaluation(self, separator=0, stage='test', break_results=False):
        model_name = self.model_name + '_' + stage if break_results else 'Null'

        if stage == 'train':
            y = self.train_dataset.targets
            y2 = self.data_loader.y2_train.copy()
            preds = self.train_preds.reshape(-1, 1)
            results = fast_def_evaluate(y, y2, preds, separator=separator, n=96, silent=self.silent,
                                        model_name=model_name)
        elif stage == 'validation':
            y = self.val_dataset.targets
            y2 = self.data_loader.y2_val.copy()
            preds = self.val_preds.reshape(-1, 1)
            results = fast_def_evaluate(y, y2, preds, separator=separator, n=96, silent=self.silent,
                                        model_name=model_name)
        elif stage == 'test':
            y = self.test_dataset.targets
            y2 = self.data_loader.y2_test.copy()
            preds = self.test_preds.reshape(-1, 1)
            results = fast_def_evaluate(y, y2, preds, separator=separator, n=96, silent=self.silent,
                                        model_name=model_name)
        else:
            results = None

        return results

    def generate_results(self, multiclass=False, separator=0, n=96):
        cms2 = []
        cm0, _ = defevaluate2(self.train_dataset.targets, self.data_loader.y2_train.copy(),
                              self.train_preds.reshape(-1, 1), separator=separator, multiclass=multiclass, n=n)
        cms2.append(cm0)
        cm1, _ = defevaluate2(self.val_dataset.targets, self.data_loader.y2_val.copy(),
                              self.val_preds.reshape(-1, 1), separator=separator, multiclass=multiclass, n=n)
        cms2.append(cm1)
        cm2, _ = defevaluate2(self.test_dataset.targets, self.data_loader.y2_test.copy(),
                              self.test_preds.reshape(-1, 1), separator=separator, multiclass=multiclass, n=n)
        cms2.append(cm2)

        fig, ax = plt.subplots(1, 3, figsize=(19.2, 4.8))
        # fig.suptitle('Confusion matrices')
        disp0 = ConfusionMatrixDisplay(cms2[0])
        disp0.plot(ax=ax[0])
        ax[0].set_title('train set')
        disp1 = ConfusionMatrixDisplay(cms2[1])
        disp1.plot(ax=ax[1])
        ax[1].set_title('validation set')
        disp2 = ConfusionMatrixDisplay(cms2[2])
        disp2.plot(ax=ax[2])
        ax[2].set_title('test set')
        plt.show()
        fig.savefig(f'Results/{self.model_name}_confusion2.png')

    def generate_test_stats(self, th=0, groups=14, precision=False, just_load=False):
        if just_load:
            with open(f'Results/{self.model_name}_samples_preds.txt', 'r') as file:
                content = file.read()[1:-1]
                content = content.split('[')[1:]
                b_preds = [int(item[0:1]) for item in content]
            with open(f'Results/{self.model_name}_samples_y.txt', 'r') as file:
                content = file.read()[1:-1]
                content = content.split('[')[1:]
                b_y_test = [int(item[0:1]) for item in content]
        else:
            b_preds = self.test_preds.reshape(-1, 1) > th
            b_preds = b_preds.astype('int32')
            b_y_test = self.test_dataset.targets > np.average(self.test_dataset.targets)
            b_y_test = b_y_test.astype('int32')
            with open(f'Results/{self.model_name}_samples_preds.txt', 'w') as file:
                file.write(str(list(b_preds)))
            with open(f'Results/{self.model_name}_samples_y.txt', 'w') as file:
                file.write(str(list(b_y_test)))

        scores = plot_stats_box2(b_y_test, b_preds, groups=groups, precision=precision)
        apx = 'precision' if precision else 'accuracy'
        with open(f'Results/{self.model_name}_tpr_{apx}.txt', 'w') as file:
            print(scores, file=file)
        return scores
