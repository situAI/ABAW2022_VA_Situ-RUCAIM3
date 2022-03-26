import time
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, precision_score

def remove_padding(batch_data, lengths):
    ans = []
    for i in range(batch_data.shape[0]):
        ans.append(batch_data[i, :lengths[i]])
    return ans

def scratch_data(data_lst):
    data = np.concatenate(data_lst)
    return data

def smooth_predictions(preds, window=40, mean_or_binomial=True):
    """
    Args:
      preds: list of (subj_timesteps, dim), len(list)=num_subjs
      window: int (one side timesteps to average)
    Return:
      smoothed_preds: shape=preds
    """
    if window == 0:
        return preds
    if not mean_or_binomial:
        from scipy.stats import binom
        binomial_weights = np.zeros((window * 2 + 1,))
        for i in range(window + 1):
            binomial_weights[i] = binomial_weights[-i - 1] = binom.pmf(i, window * 2, 0.5)
    smoothed_preds = []
    for pred in preds:
        smoothed_pred = []
        for t in range(pred.shape[0]):
            left = np.max([0, t - window])
            right = np.min([pred.shape[0], t + window])
            if mean_or_binomial:
                smoothed_pred.append(np.mean(pred[left: right + 1]))
            else:
                if left <= 0:
                    weights = binomial_weights[window - t:]
                elif right >= pred.shape[0]:
                    weights = binomial_weights[:pred.shape[0] - t - window - 1]
                else:
                    weights = binomial_weights
                smoothed_pred.append(np.sum(pred[left: right + 1] * weights))
        smoothed_preds.append(np.array(smoothed_pred))
    smoothed_preds = np.array(smoothed_preds, dtype=object)
    return smoothed_preds

def smooth_func(pred, label=None, best_window=None, logger=None):
    start = time.time()
    if best_window is None:
        best_ccc, best_window = 0, 0
        # for window in range(0, 30, 5):
        for window in range(0, 100, 5):
            smoothed_preds = smooth_predictions(pred, window=window)
            if label is not None:
                mse, rmse, pcc, ccc = evaluate_regression(y_true=scratch_data(label),
                                                        y_pred=scratch_data(smoothed_preds))
                if logger:
                    logger.info('In smooth Eval \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}'.format(
                        window, mse, rmse, pcc, ccc))
                else:
                    print('In smoothing \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}'.format(
                        window, mse, rmse, pcc, ccc))
                
                if ccc > best_ccc:
                    best_ccc, best_window = ccc, window
        end = time.time()
        time_usage = end - start
        if logger:
            logger.info('Smooth: best window {:d} best_ccc {:.4f} \t Time Taken {:.4f}'.format(best_window, best_ccc, time_usage))
        else:
            print('Smooth: best window {:d} best_ccc {:.4f} \t Time Taken {:.4f}'.format(best_window, best_ccc, time_usage))
        smoothed_preds = smooth_predictions(pred, window=best_window)
    elif best_window is not None:
        smoothed_preds = smooth_predictions(pred, window=best_window)
    
    return smoothed_preds, best_window
    
def evaluate_regression(y_true, y_pred):
    """ Evaluate the regression performance
        Params:
        y_true, y_pred: np.array()
        Returns:
        mse, rmse, pcc, ccc
    """
    assert y_true.ndim==1 and y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_true, y_pred)[0][1]
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    ccc = 2 * np.cov(y_true, y_pred, ddof=0)[0][1] / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return mse, rmse, pcc, ccc

# def evaluate_classification(y_true, y_pred):
#     acc = accuracy_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
#     cm = confusion_matrix(y_true, y_pred)
#     return acc, recall, f1, cm, .66 * f1 + .34 * recall

def evaluate_classification(y_true, y_pred):
    '''
    Evaluate the classification performance
        Params:
        y_true, y_pred: np.array()
        Returns:
        acc, recall, precision, f1, cm
    '''
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    # recall_for_each_class = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average='macro') 
    # preision_for_each_class = precision_score(y_true, y_pred, average=None) 
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, recall, precision, f1, cm