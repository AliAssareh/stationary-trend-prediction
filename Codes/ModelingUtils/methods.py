from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch


class QualifiedConvCandlesDataset(Dataset):
    def __init__(self, features_df, labels_df):
        try:
            assert len(features_df) == len(labels_df)
        except Exception as ve:
            raise Exception(f'length of features_df {len(features_df)} dose not math length '
                            f'of labels_df {len(labels_df)}')
        self.features = features_df
        self.targets = labels_df

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx, :].reshape(1, *np.shape(self.features[idx, :]))
        y = self.targets[idx, :]

        return (x, y)


class QualifiedWeightedConvCandlesDataset(Dataset):
    def __init__(self, features_df, labels_df, w0=1, w1=1):
        try:
            assert len(features_df) == len(labels_df)
        except Exception as ve:
            raise Exception(f'length of features_df {len(features_df[3])} dose not math length '
                            f'of labels_df {len(labels_df)}')
        self.w0 = w0
        self.w1 = w1
        self.features = features_df
        self.targets = labels_df

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx, :].reshape(1, *np.shape(self.features[idx, :]))
        y = self.targets[idx, :]
        w = np.asarray([self.w0 if el <= 0 else self.w1 for el in y]).astype('float64').reshape(1, 1)
        return (x, w, y)


def train_epoch(model, device, dataloader, loss_fn1, loss_fn2, w1=0.75, w2=0.25, optimizer=None, weighted=False):
    model.train()
    total_loss = 0.0
    if weighted:
        for x, w, y in dataloader:
            x = x.to(device, dtype=torch.double)
            w = w.to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            w = torch.reshape(w, (1, w.size(0)))
            l1 = loss_fn1(model(x), y)
            l2 = loss_fn2(model(x), y)
            ll1 = torch.matmul(w, l1)
            ll2 = torch.matmul(w, l2)
            loss = w1 * ll1 + w2 * ll2

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
    else:
        for x, y in dataloader:
            x = x.to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            loss = w1 * loss_fn1(model(x), y) + w2 * loss_fn2(model(x), y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
    return total_loss / len(dataloader.dataset)


def test_epoch(model, device, dataloader, loss_fn1, loss_fn2, w1=0.75, w2=0.25, weighted=False):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        if weighted:
            for x, w, y in dataloader:
                x = x.to(device, dtype=torch.double)
                w = w.to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                w = torch.reshape(w, (1, w.size(0)))
                l1 = loss_fn1(model(x), y)
                l2 = loss_fn2(model(x), y)
                ll1 = torch.matmul(w, l1)
                ll2 = torch.matmul(w, l2)
                loss = w1 * ll1 + w2 * ll2

                total_loss = total_loss + loss.item()
        else:
            for x, y in dataloader:
                x = x.to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                loss = w1 * loss_fn1(model(x), y) + w2 * loss_fn2(model(x), y)

                total_loss = total_loss + loss.item()
    return total_loss / len(dataloader.dataset)


def save_model_on_validation_improvement(model, optimizer, epoch, name):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, f'Files//torch_models//{name}.pth')


def load_model(path, model, optimizer=None):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    if optimizer == None:
        return model
    else:
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer


def eval_predict(model, dataset, device, silent=False):
    predictions = []
    model.eval()
    with torch.no_grad():
        if silent:
            for sample in dataset:
                x = torch.tensor(sample[0].reshape(1, *np.shape(sample[0]))).to(device)
                y_hat = model(x).cpu().numpy()
                predictions.append(y_hat)
        else:
            for sample in tqdm(dataset):
                x = torch.tensor(sample[0].reshape(1, *np.shape(sample[0]))).to(device)
                y_hat = model(x).cpu().numpy()
                predictions.append(y_hat)
    return np.array(predictions)


def simple_evaluation(b_y_test, y2_test, b_preds):
    profit_series = y2_test.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds == b_y_test)), 'target']
    loss_series = y2_test.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds != b_y_test)), 'stop']
    pure_profit = 100 * (profit_series.sum() + loss_series.sum())
    return [pure_profit]


def eval(y, y2, preds, separator):
    results = []
    b_preds = preds > separator
    b_preds = b_preds.astype('int32')
    b_y_test = y > np.average(y)
    b_y_test = b_y_test.astype('int32')
    res = simple_evaluation(b_y_test, y2.copy(), b_preds)
    results.extend(res)
    return results


def eval_search(y_val, y2_val, preds_val, y_test, y2_test, preds_test):
    data = []
    for separator in tqdm(np.arange(-0.21, 0.10, 0.01)):
        r1 = eval(y_val, y2_val.copy(), preds_val, separator)
        r2 = eval(y_test, y2_test.copy(), preds_test, separator)
        data.append([separator, *r1, *r2])

    df = pd.DataFrame(data, columns=['separator', 'val', 'test'])
    pos_df = df[(df.val > 0) & (df.test > 0)].copy()

    if len(pos_df) > 0:
        temp_df = pos_df[(pos_df.val > 1) & (pos_df.test > 1)]
        sep = 0
        if len(temp_df) > 0:
            pos_df.loc[:, 'avg'] = 2 * (temp_df.val * temp_df.test) / (temp_df.val + temp_df.test)
            print('max_a:')
            mtrain = pos_df.loc[pos_df.val.idxmax(), ['separator', 'val']]
            print(f'separator: {mtrain.separator}, a return: {mtrain.val}', end='\n')
            sep = mtrain.separator
            print('max_b:')
            mtest = pos_df.loc[pos_df.test.idxmax(), ['separator', 'test']]
            print(f'separator: {mtest.separator}, b return: {mtest.test}', end='\n')
            print('max_avg:')
            print(pos_df.loc[pos_df.avg.idxmax(), :], end='\n\n')
        else:
            print('no separator had positive val and test!')
        return sep
    else:
        print('There where no profitable separator')
        return 0


def eval_search0(y, y2, preds):
    data = []
    rng = np.arange(-0.20, 0.10, 0.01)
    for separator in tqdm(rng):
        r1 = eval(y, y2.copy(), preds, separator)
        data.append([separator, *r1])
    df = pd.DataFrame(data, columns=['separator', 'val'])

    pos_df = df[df.val > 0]

    if len(pos_df) > 0:
        temp_df = pos_df.copy()
        if len(temp_df) > 0:
            mtrain = pos_df.loc[pos_df.val.idxmax(), ['separator', 'val']]
            return mtrain.separator, mtrain.val
        else:
            return None, None
    else:
        return None, None


def fast_def_evaluate(y, y2, preds, separator, n=96, silent=True, model_name='Null'):
    if separator is None:
        b_preds = preds > np.average(y)
        b_preds = b_preds.astype('int32')
    else:
        b_preds = preds > separator
        b_preds = b_preds.astype('int32')
    b_y = y > np.average(y)
    b_y = b_y.astype('int32')
    cm0 = confusion_matrix(b_y, b_preds, normalize=None)
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    acc = 100 * (cm0[0, 0] + cm0[1, 1]) / (cm0[0, 0] + cm0[0, 1] + cm0[1, 0] + cm0[1, 1])
    tpr = 100 * (cm0[1, 1]) / (cm0[0, 1] + cm0[1, 1])

    y2['pnl'] = np.array(0).astype(float)
    y2.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds == b_y)), 'pnl'] = y2['target']
    y2.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds != b_y)), 'pnl'] = y2['stop']
    pp0 = 100 * y2.pnl.sum()

    capita0 = y2.pnl.cumsum() * 100
    capita0 = capita0 + 100
    peak0 = 0
    valley0 = 0
    mdd0 = 0
    for i in range(len(capita0)):
        if capita0.iloc[i] > peak0:
            peak0 = capita0.iloc[i]
            valley0 = capita0.iloc[i]

        valley0 = min(valley0, capita0.iloc[i])
        new_mdd0 = 100 * (peak0 - valley0) / peak0 if peak0 != 0 else 0
        mdd0 = max(mdd0, new_mdd0)

    high = y2.high
    low = y2.low
    peak = 0
    valley = 0
    mdd = 0
    for i in range(len(high)):
        if high.iloc[i] > peak:
            peak = high.iloc[i]
            valley = high.iloc[i]

        valley = min(valley, low.iloc[i])
        new_mdd = -100 * (valley - peak) / peak if peak != 0 else 0
        mdd = max(mdd, new_mdd)

    if model_name != 'Null':
        bins = int(len(y2) / (n * 14))
        size = int(len(y2) / bins)
        gross_profit_list = []
        n_gross_profit_list = []
        gross_loss_list = []
        n_loss_list = []
        profit_list = []
        n_trade_list = []
        bnh_list = []
        mdds = []
        for i in range(bins):
            if i != (bins - 1):
                y2_part = y2.iloc[i * size:(i + 1) * size, :]
            else:
                y2_part = y2.iloc[i * size:, :]

            profit_series = y2_part.loc[y2_part.pnl > 0, 'pnl']
            loss_series = y2_part.loc[y2.pnl < 0, 'pnl']
            open_price = y2_part.iloc[0, y2_part.columns.get_loc('open')]
            close_price = y2_part.iloc[-1, y2_part.columns.get_loc('close')]
            gross_profit_list.append(100 * profit_series.sum())
            n_gross_profit_list.append(len(profit_series))
            gross_loss_list.append(-100 * loss_series.sum())
            n_loss_list.append(len(loss_series))
            profit_list.append(100 * (profit_series.sum() + loss_series.sum()))
            n_trades = len(profit_series) + len(loss_series)
            n_trade_list.append(100 * len(profit_series) / n_trades if n_trades != 0 else 0)
            bnh_list.append(100 * (close_price - open_price) / open_price)
            mdds.append(mdd)

        df = pd.DataFrame(data={'gross_profit': gross_profit_list, 'n_gross': n_gross_profit_list,
                                'loss': gross_loss_list, 'n_loss': n_loss_list, 'profit': profit_list,
                                'n_profit': n_trade_list, 'bnh': bnh_list, 'mdd': mdds})

        df.to_csv(f'Results/{model_name}_break0.csv')
        with open(f'./Confusion matrices/cm_{model_name}0.npy', 'wb') as f:
            np.save(f, cm0)

    if not silent:
        print(f'accuracy: {acc}')
        print(f'TPR: {tpr}')
        print(f'AUC: {roc_auc}')
        print(f'pure_profit={pp0}%  during {len(y2)} candles({len(y2) / n} days)')
        print(f'MDD of model: {mdd0}')

    return (acc, tpr, roc_auc, pp0, mdd0)


def defevaluate2(y_test, y2_test, preds, separator=None, multiclass=False, n=6):
    plot_samples(y_test, preds, multiclass, separator)
    plt.figure(figsize=(6.4, 4.8))

    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    if separator is None:
        b_preds = preds > np.average(y_test)
        b_preds = b_preds.astype('int32')
    else:
        b_preds = preds > separator
        b_preds = b_preds.astype('int32')
    b_y_test = y_test > np.average(y_test)
    b_y_test = b_y_test.astype('int32')

    cm0 = confusion_matrix(b_y_test, b_preds, normalize=None)
    disp0 = ConfusionMatrixDisplay(cm0)
    fig, ax = plt.subplots(1, 3, figsize=(19.2, 4.8))
    disp0.plot(ax=ax[0])
    ax[0].set_title('confusion matrix')
    for text_arr in disp0.text_:
        for text in text_arr:
            text.set_fontsize(16)

    cm1 = confusion_matrix(b_y_test, b_preds, normalize='true')
    disp1 = ConfusionMatrixDisplay(cm1)
    disp1.plot(ax=ax[1])
    ax[1].set_title('normalized confusion matrix 1')
    for text_arr in disp1.text_:
        for text in text_arr:
            text.set_fontsize(16)

    cm2 = confusion_matrix(b_y_test, b_preds, normalize='pred')
    disp2 = ConfusionMatrixDisplay(cm2)
    disp2.plot(ax=ax[2])
    ax[2].set_title('normalized confusion matrix 2')
    for text_arr in disp2.text_:
        for text in text_arr:
            text.set_fontsize(16)
    plt.show()

    profit_series = y2_test.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds == b_y_test)), 'target']
    loss_series = y2_test.loc[((b_preds != np.zeros_like(b_preds)) & (b_preds != b_y_test)), 'stop']
    print(f'accuracy: {(cm0[0, 0] + cm0[1, 1]) / (cm0[0, 0] + cm0[0, 1] + cm0[1, 0] + cm0[1, 1])}')
    print(f'pure_profit={100 * (profit_series.sum() + loss_series.sum())}%  during {len(y2_test)}'
          f' candles({len(y2_test) / n} days)\n')

    open_price = y2_test.iloc[0, y2_test.columns.get_loc('open')]
    close_price = y2_test.iloc[-1, y2_test.columns.get_loc('close')]
    print(f'B&H profit: {100 * (close_price - open_price) / open_price}')

    print('details: \n\n')

    size = int(len(y2_test) / 10)
    gross_profit_list = []
    n_gross_profit_list = []
    gross_loss_list = []
    n_gross_loss_list = []
    profit_list = []
    n_trade_list = []
    bnh_list = []
    for i in range(10):
        if i != 9:
            y2_test_part = y2_test.iloc[i * size:(i + 1) * size, :]
            b_preds_part = b_preds[i * size:(i + 1) * size, 0]
            b_y_test_part = b_y_test[i * size:(i + 1) * size, 0]
        else:
            y2_test_part = y2_test.iloc[i * size:, :]
            b_preds_part = b_preds[i * size:, 0].reshape(-1, 1)
            b_y_test_part = b_y_test[i * size:, 0].reshape(-1, 1)

        profit_series = y2_test_part.loc[
            ((b_preds_part == np.ones_like(b_preds_part)) & (b_preds_part == b_y_test_part)), 'target']
        loss_series = y2_test_part.loc[
            ((b_preds_part == np.ones_like(b_preds_part)) & (b_preds_part != b_y_test_part)), 'stop']
        open_price = y2_test_part.iloc[0, y2_test_part.columns.get_loc('open')]
        close_price = y2_test_part.iloc[-1, y2_test_part.columns.get_loc('close')]
        gross_profit_list.append(100 * profit_series.sum())
        n_gross_profit_list.append(len(profit_series))
        gross_loss_list.append(100 * loss_series.sum())
        n_gross_loss_list.append(len(loss_series))
        profit_list.append(100 * (profit_series.sum() + loss_series.sum()))
        n_trades = len(profit_series) + len(loss_series)
        n_trade_list.append(100 * len(profit_series) / n_trades if n_trades != 0 else 0)
        bnh_list.append(100 * (close_price - open_price) / open_price)

    print(f'profits: avg: {np.average(gross_profit_list)}, std: {np.std(gross_profit_list)}')
    print([f'{item:6.0f}' for item in n_gross_profit_list])
    print([f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in gross_profit_list], end='\n\n')
    print(f'loss: avg: {np.average(gross_loss_list)}, std: {np.std(gross_loss_list)}')
    print([f'{item:6.0f}' for item in n_gross_loss_list])
    print([f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in gross_loss_list], end='\n\n')
    print(f'pure profit: avg: {np.average(profit_list)}, std: {np.std(profit_list)}')
    tpr = [f'{item:6.1f}%' for item in n_trade_list]
    print(tpr)
    pp = [f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in profit_list]
    print(pp, end='\n\n')
    print(f'B&H profit: avg: {np.average(bnh_list)}, std: {np.std(bnh_list)}')
    bnh = [f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in bnh_list]
    print(bnh, end='\n\n')
    print(f'n_candles: {size}, \n\n')

    """ from eval3"""
    print('form evaluate3 \n\n')

    cm = np.array([[0, 0],
                   [0, 0]])
    y2_test['pnl'] = np.array(0).astype(float)
    pnl_idx = y2_test.columns.get_loc('pnl')
    y2_test['buy'] = 0
    buy_idx = y2_test.columns.get_loc('buy')
    y2_test['sell'] = 0
    sell_idx = y2_test.columns.get_loc('sell')
    y2_test['r_profit'] = 0
    r_profit_idx = y2_test.columns.get_loc('r_profit')
    target_idx = y2_test.columns.get_loc('target')
    stop_idx = y2_test.columns.get_loc('stop')
    i = 1
    win_js = []
    loss_js = []
    while i < len(b_preds):
        if b_preds[i] == 0 and b_y_test[i] == 0:
            cm[0, 0] = cm[0, 0] + 1
            i = i + 1
        elif b_preds[i] == 0 and b_y_test[i] == 1:
            cm[1, 0] = cm[1, 0] + 1
            i = i + 1
        elif b_preds[i] == 1 and b_y_test[i] == 0:
            y2_test.iloc[i, buy_idx] = 1
            cm[0, 1] = cm[0, 1] + 1
            j = 1
            loss = y2_test.iloc[i, stop_idx]
            loss_js.append(j)
            nn = i + j if i + j < len(y2_test) else len(y2_test) - 1
            y2_test.iloc[nn, sell_idx] = 1
            y2_test.iloc[nn, pnl_idx] = loss
            y2_test.iloc[nn, r_profit_idx] = -1
            i = i + j
        else:
            y2_test.iloc[i, buy_idx] = 1
            cm[1, 1] = cm[1, 1] + 1
            j = 1
            profit = y2_test.iloc[i, target_idx]
            win_js.append(j)
            nn = i + j if i + j < len(y2_test) else len(y2_test) - 1
            y2_test.iloc[nn, sell_idx] = 1
            y2_test.iloc[nn, pnl_idx] = profit
            y2_test.iloc[nn, r_profit_idx] = 1
            i = i + j

    bins = int(len(y2_test) / (n * 14))
    size = int(len(y2_test) / bins)
    gross_profit_list = []
    n_gross_profit_list = []
    gross_loss_list = []
    n_loss_list = []
    profit_list = []
    n_trade_list = []
    bnh_list = []
    for i in range(bins):
        if i != (bins - 1):
            y2_test_part = y2_test.iloc[i * size:(i + 1) * size, :]
        else:
            y2_test_part = y2_test.iloc[i * size:, :]

        profit_series = y2_test_part.loc[y2_test_part.pnl > 0, 'pnl']
        loss_series = y2_test_part.loc[y2_test.pnl < 0, 'pnl']
        open_price = y2_test_part.iloc[0, y2_test_part.columns.get_loc('open')]
        close_price = y2_test_part.iloc[-1, y2_test_part.columns.get_loc('close')]
        gross_profit_list.append(100 * profit_series.sum())
        n_gross_profit_list.append(len(profit_series))
        gross_loss_list.append(-100 * loss_series.sum())
        n_loss_list.append(len(loss_series))
        profit_list.append(100 * (profit_series.sum() + loss_series.sum()))
        n_trades = len(profit_series) + len(loss_series)
        n_trade_list.append(100 * len(profit_series) / n_trades if n_trades != 0 else 0)
        bnh_list.append(100 * (close_price - open_price) / open_price)

    ticks = np.arange(bins)
    fig = plt.figure(figsize=(25, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.hlines(0, xmin=ticks[0] - 0.25, xmax=ticks[-1] + 0.75, color='black')
    ax.bar(ticks + 0.00, gross_profit_list, color='b', width=0.25, edgecolor='black')
    ax.bar(ticks + 0.25, profit_list, color='g', width=0.25, edgecolor='black')
    ax.bar(ticks + 0.50, bnh_list, color='r', width=0.25, edgecolor='black')
    ax.legend(labels=['zero_line', 'gross', 'profit', 'B&H'])
    ax.set_xticks([tick + 0.25 for tick in ticks], labels=ticks)
    plt.show()

    capita = y2_test.pnl.cumsum() * 100
    capita = capita + 100
    peak = 0
    valley = 0
    mdd = 0
    for i in range(len(capita)):
        if capita.iloc[i] > peak:
            peak = capita.iloc[i]
            valley = capita.iloc[i]

        valley = min(valley, capita.iloc[i])
        new_mdd = -100 * (valley - peak) / peak if peak != 0 else 0
        mdd = max(mdd, new_mdd)

    fig = plt.figure(figsize=(25, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.hlines(100, xmin=0, xmax=len(capita))
    ax.plot([i for i in range(len(capita))], capita)
    plt.show()

    print(f'MDD of model: {mdd}')

    high = y2_test.high
    low = y2_test.low
    peak = 0
    valley = 0
    mdd = 0
    for i in range(len(high)):
        if high.iloc[i] > peak:
            peak = high.iloc[i]
            valley = high.iloc[i]

        valley = min(valley, low.iloc[i])
        new_mdd = -100 * (valley - peak) / peak if peak != 0 else 0
        mdd = max(mdd, new_mdd)
    print(f'MDD of B&H: {mdd}')

    print(f'profits: avg: {np.average(gross_profit_list)}, std: {np.std(gross_profit_list)}')
    print([f'{item:6.0f}' for item in n_gross_profit_list])
    print([f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in gross_profit_list], end='\n\n')
    print(f'loss: avg: {np.average(gross_loss_list)}, std: {np.std(gross_loss_list)}')
    print([f'{item:6.0f}' for item in n_loss_list])
    print([f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in gross_loss_list], end='\n\n')
    print(f'pure profit: avg: {np.average(profit_list)}, std: {np.std(profit_list)}')
    tpr = [f'{item:6.1f}%' for item in n_trade_list]
    print(tpr)
    pp = [f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in profit_list]
    print(pp, end='\n\n')
    print(f'B&H profit: avg: {np.average(bnh_list)}, std: {np.std(bnh_list)}')
    bnh = [f'{item:6.3f}' if item >= 0 else f'{item:6.2f}' for item in bnh_list]
    print(bnh, end='\n\n')
    print(f'winning signals on average took {np.average(win_js)} candles with std: {np.std(win_js)}')
    print(f'losing signals on average took {np.average(loss_js)} candles with std: {np.std(loss_js)}')
    print(f'n_candles in each interval: {size} ({size / n} days)')
    print(f'number of bins: {bins}')

    return cm2, (tpr, pp, bnh)


def plot_samples(labels, preds, multiclass, separator):
    y_min = np.amin(labels)
    y_max = np.amax(labels)
    plt.figure(figsize=(25, 4.8))
    plt.plot(preds, labels, 'b|')
    plt.plot(preds[0:12], labels[0:12], 'r|')
    plt.ylabel('label')
    plt.xlabel('prediction')
    plt.title('predictions vs labels')
    plt.yticks([-1, 1])
    if multiclass is True:
        sep = 0.33 if separator is None else separator
        plt.vlines(-1 * sep, ymin=y_min, ymax=y_max)
        plt.vlines(sep, ymin=y_min, ymax=y_max)
    else:
        if separator is None:
            plt.vlines(np.average(labels), ymin=y_min, ymax=y_max)
        else:
            plt.vlines(separator, ymin=y_min, ymax=y_max)
    plt.show()


def plot_stats_box2(b_y_test, b_preds, groups=30, precision=False):
    batch_size = int(len(b_preds) / groups)
    scores = []
    for i in range(groups):
        if i != groups - 1:
            predictions = b_preds[i * batch_size: (i + 1) * batch_size]
            labels = b_y_test[i * batch_size: (i + 1) * batch_size]
        else:
            predictions = b_preds[i * batch_size:]
            labels = b_y_test[i * batch_size:]
        cm0 = confusion_matrix(labels, predictions, normalize=None)
        if precision is False:
            accuracy = (cm0[0, 0] + cm0[1, 1]) / (cm0[0, 0] + cm0[0, 1] + cm0[1, 0] + cm0[1, 1])
            scores.append(100 * accuracy)
        else:
            p = cm0[1, 1] / (cm0[0, 1] + cm0[1, 1])
            scores.append(100 * p)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8))
    name = 'Precisions' if precision else 'Accuracies'
    ax.set_title(f'Boxplot of {name} on 15 days basis')
    ax.boxplot(scores, notch=True)

    return scores