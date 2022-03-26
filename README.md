#  Multi-modal Emotion Estimation for in-the-wild Videos

This is a repository for our solution for the [ABAW 2022 Valence-Arousal Challenge](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/), and you can also view the paper submission [here](https://arxiv.org/abs/2203.13032).

## Requirements

`pip install -r requirements.txt`

## Data preparation

In our proposed method, extracting all kinds of features is the first thing to do.

+ Visual Features:
    + `IResNet100` pretrained on `Glint360K` for face recognition, which you can find in [InsightFace](https://github.com/deepinsight/insightface), and then training it on `AffectNet`, `RAF-DB` and `FER+` for expression classification.
    + `IResNet100` pretrained on `Glint360K` for face recognition, and then training it on our own `Facial Action Units` dataset that we made by ourselves.
+ Audio Features:
    + `ComParE2016` using [openSMILE](https://github.com/audeering/opensmile).
    + `eGeMAPS` using [openSMILE](https://github.com/audeering/opensmile)
    + `VGGish` using [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
    + `wav2vec` using [huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)

After that, you may edit `config/te.yml` as follows:

`TYPE`: `IA` means that using `Image` and `Audio` features

`Data.img_dir`: the root path of `aligned images` which is officially available

`Data.train_seq_file`: the generated csv file which is the sequence-style data

`Data.val_seq_file`: the generated csv file which is the sequence-style data

`Data.train_label_file`: the train data and annotations

`Data.val_label_file`: the validation data and annotations

`Data.train_img_feat_list`: the visual feature list of training phase

`Data.val_img_feat_list`: the visual feature list of validation phase

`Data.train_audio_feat_list`: the audio feature list of training phase

`Data.val_audio_feat_list`: the audio feature list of validation phase

`Data.seq_len`: the sequence length of which batch

`Data.loader`: train or validation or test loader settings

`Model.*`: the model detail, which is corresponding to `model/model_zoo`

`Log.*` the configuration of log file and checkpoint path

`Solver.*` the configuration of `solver.py`



## Cite

If this helps you, please cite:

```
```





