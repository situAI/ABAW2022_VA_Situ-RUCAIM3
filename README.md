# Multi-modal Emotion Estimation for in-the-wild Videos

This is a repository of our solution for the Valence-Arousal Estimation Challenge of the [ABAW 2022 Challenge](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/).

Paper link: [Multi-modal Emotion Estimation for in-the-wild Videos](https://arxiv.org/pdf/2203.13032.pdf)



## Train

run the "transformer" model:

```bash
bash scripts/train_transformer.sh both [features] [normed_features] [data_root] [run_id] [gpu_id]
```

- `features`: All features used to train the model. Split by commas.
- `normed_features`: Features need to normalize before feed into the model. Split by commas. If there are no features need to be normed, then input `None`.
- `data_root`: the root path of your data and label files
- `run_id`: The same run_id takes the same seed.
- `gpu_id`: Which gpu to use when train the model. 

for example:

```bash
bash scripts/train_transformer.sh both affectnet,vggish,wav2vec None /data/ABAW_VA_2022/processed_data/ 1 0
```



run the "transformer-lstm" model:

```bash
bash scripts/train_transformer_lstm.sh both [features] [normed_features] [data_root] [residual] [normed_features] [run_id] [gpu_id]
```

- residual: Whether to use the residual connection to skip the transformer encoder



run the "transformer-attention" model:

```bash
bash scripts/train_transformer_attention.sh both [features] [normed_features] [data_root] [run_id] [gpu_id]
```



run the "fclstm-xl" model:

```bash
bash scripts/train_lstm.sh both [features] [normed_features] [data_root] [run_id] [gpu_id]
```



## Inference

### 1. Get json file of predictions on val and test set

```bash
bash scripts/test.sh
```

You need to modify some parameters in this file before run:

- `gpu_id`
- `test_target`：'arousal' or 'valence'
- `test_checkpoints`: Checkpoints of models you want to do inference on the val and test set. Support for multiple inputs, split by semicolon.
  - format: checkpoint_name/prefix
    - prefix: part of the pth file name before `_net_xxx.pth` (correspoding to the best eval epoch on your test_target, you can find the number in your log files)
  - for example, if you train a model, the best epoch of arousal is 2 and the best epoch of valence is 3. There will exists `2_net_xxx.pth` and `3_net_xxx.pth` in the dir: `./checkpoints/[checkpoint_name]/`. If you want to do inference on the "arousal" target, you need to input "[checkpoint_name]/2" here.

There will be 7 files in the dir: `./test_results/[checkpoint_name]/` after running the above shell:

- `val_pred_[valence/arousal]_nosmooth_ori.json`: original predictions on the val set of the model. There might be some values out of the range of [-1, 1]

- `val_pred_[valence/arousal]_nosmooth.json`: restrict the values in the range of [-1, 1]

- `val_pred_[valence/arousal]_smooth.json`: Smoothed sequence of the predictions. We set the smooth window size of 50 for valence and 20 for arousal. All values are in the range of [-1, 1]

- `val_pred_[valence/arousal]_result.txt`: record the evaluation results on the val set

- `tst_pred_[valence/arousal]_nosmooth.json`

- `tst_pred_[valence/arousal]_smooth.json`



### 2. Do ensemble and generate the txt file of predictions on the test set

```bash
bash scripts/get_preds.sh
```

You need to modify some parameters in this file before run:

- `test_target`: 'valence' or 'arousal'
- `name`: the result files will be saved at `./submit/[name]/[test_target]/`
- `smoothed`: Whether to smooth the prediction sequences. 'y' or 'n'.
- `test_results`: The [checkpoint_name]s of all models you want to do ensemble. Split by semicolon.



## Requirements

1. python 3.7
2. CUDA 10.0
3. Install other requirements using

```bash
pip install requirements.txt
```
