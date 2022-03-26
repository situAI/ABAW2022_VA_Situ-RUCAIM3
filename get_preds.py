import os
import numpy as np
import json
import pandas as pd
from opts.test_opts import TestOptions
from utils.metrics import evaluate_regression, scratch_data

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def ensemble_all_preds(all_preds):
    pred_keys = sorted(all_preds[0].keys())
    results = {}
    for video_id in pred_keys:
        all_pred_vid = [all_preds[i][video_id]['pred'] for i in range(len(all_preds))]
        all_pred_vid = np.asarray(all_pred_vid)
        all_pred_vid = np.mean(all_pred_vid, axis=0)
        results[video_id] = {'video_id': video_id,
                                'pred': all_pred_vid.tolist()
        }
    return results

def make_csv(pred_data, save_dir, target='arousal'):
    pred_keys = sorted(pred_data.keys())
    for video_id in pred_keys:
        save_path = os.path.join(save_dir, video_id+'.txt')
        pred = pred_data[video_id]['pred']
        df = pd.DataFrame(pred,columns=[target])
        df.to_csv(save_path, index=None)

if __name__ == '__main__':
    opt = TestOptions().parse()
    name = opt.name
    smoothed = True if opt.smoothed == 'y' else False
    test_log_dir = opt.test_log_dir
    test_results = opt.test_results.strip().split(';')
    print(test_results)
    print('---------------------------------')

    val_all_preds = []
    tst_all_preds = []
    
    weights = []
    smooth_weights = []
    
    for test_result in test_results:
        if len(test_result) == 0:
            continue
        test_result = test_result.replace(' ', '')
        print('Get {} pred from {}: '.format(opt.test_target, test_result))
        if smoothed:
            val_pred_path = os.path.join(test_log_dir, test_result, 'val_pred_{}_smooth.json'.format(opt.test_target))
            tst_pred_path = os.path.join(test_log_dir, test_result, 'tst_pred_{}_smooth.json'.format(opt.test_target))
        else:
            val_pred_path = os.path.join(test_log_dir, test_result, 'val_pred_{}_nosmooth.json'.format(opt.test_target))
            tst_pred_path = os.path.join(test_log_dir, test_result, 'tst_pred_{}_nosmooth.json'.format(opt.test_target))
        with open(val_pred_path, 'r') as f:
            val_pred = json.load(f)
        with open(tst_pred_path, 'r') as f:
            tst_pred = json.load(f)
        val_all_preds.append(val_pred)
        tst_all_preds.append(tst_pred)

    # eval val ensemble prediction
    video_ids = sorted(list(val_all_preds[0].keys()))
    val_ensemble_preds = ensemble_all_preds(val_all_preds)
    val_targets = []
    val_preds = []
    for video in video_ids:
        val_preds.append(val_ensemble_preds[video]['pred'])
        val_targets.append(val_all_preds[0][video]['label'])
    total_pred = scratch_data(val_preds)
    total_label = scratch_data(val_targets)
    mse, rmse, pcc, ccc = evaluate_regression(total_label, total_pred)
    if smoothed:
        print('Ensemble %s Val result mse %.6f rmse %.6f pcc %.6f ccc %.6f with smooth' % (opt.test_target, mse, rmse, pcc, ccc))
    else:
        print('Ensemble %s Val result mse %.6f rmse %.6f pcc %.6f ccc %.6f without smooth' % (opt.test_target, mse, rmse, pcc, ccc))
    
    # save test ensemble result
    tst_ensemble_pred = ensemble_all_preds(tst_all_preds)
    ensemble_dir = os.path.join(opt.submit_dir, opt.name, opt.test_target)
    mkdir(ensemble_dir)
    make_csv(tst_ensemble_pred, ensemble_dir, opt.test_target)