import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
from PIL import Image
import torch
from torchvision import transforms
import torchaudio
import torchaudio.transforms as audio_transforms
import random

def get_transforms():
    train_data_aug = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    val_data_aug = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return train_data_aug, val_data_aug


def load_feat(feat_dir, flag='train'):
    npy_fns = sorted([i for i in os.listdir(feat_dir) if i.endswith('npy') and i.startswith(flag)])
    feat_list = []
    for fn in npy_fns:
        tmp_feat = np.load(os.path.join(feat_dir, fn))
        feat_list.append(tmp_feat)
    feats = np.concatenate(feat_list, axis=0)
    # z-score
    feats = (feats - np.mean(feats, axis=1)[:, np.newaxis])/(np.std(feats, axis=1)[:, np.newaxis])
    idx_list = []
    idx_list = pd.read_csv(os.path.join(feat_dir, flag+'.csv'))
    idx_dict = pd.Series(range(len(idx_list)), index=list(idx_list.embedd_idx))
    return idx_dict, feats        


class IA_AlignAffWild2(Dataset):
    
    def __init__(self, img_dir, seq_csv_file, label_csv_file, img_feat_list, audio_feat_list, transform, flag='val', seq_len=32):
        self.seq_df = pd.read_csv(seq_csv_file)
        self.label_df = pd.read_csv(label_csv_file, index_col='main_idx')
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.main_idx_dict = pd.Series(range(len(self.label_df)), index=list(self.label_df.index))
        self.img_idx_dicts, self.img_feat_nps = [], []
        for img_feat_path in img_feat_list:
            tmp_dict, tmp_feat = load_feat(img_feat_path, flag)
            self.img_idx_dicts.append(tmp_dict)
            self.img_feat_nps.append(tmp_feat)
        self.audio_idx_dicts, self.audio_feat_nps = [], []
        for audio_feat_path in audio_feat_list:
            tmp_dict, tmp_feat = load_feat(audio_feat_path, flag)
            self.audio_idx_dicts.append(tmp_dict)
            self.audio_feat_nps.append(tmp_feat)
        self.transform = transform
        self.lens = len(self.seq_df)

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        main_idx = self.seq_df.iloc[idx].main_idx
        start_idx = int(self.label_df.loc[main_idx].frame_idx)
        vn = self.label_df.loc[main_idx].vn
        vlabels = []
        alabels = []
        img_feats = []
        wav_feats = []
        main_idxs = []
        for i in range(self.seq_len):
            main_idx = '%s/%05d'%(vn, start_idx+i)
            if self.main_idx_dict.get(main_idx) is None:
                vlabels.append(-5)
                alabels.append(-5)
                img_feats.append(img_feats[-1])
                wav_feats.append(wav_feats[-1])
                main_idxs.append(main_idxs[-1])
                continue
            
            tmp_img_feats = []
            for feat_i in range(len(self.img_idx_dicts)):
                if self.img_idx_dicts[feat_i].get(main_idx) is None:
                    if self.img_idx_dicts[feat_i].get(self.label_df.loc[main_idx].img_path) is not None:
                        np_idx = self.img_idx_dicts[feat_i][self.label_df.loc[main_idx].img_path]
                        tmp_img_feats.append(self.img_feat_nps[feat_i][np_idx])
                    elif i != 0:
                        tmp_img_feats=img_feats[-1]
                        break
                    else:
                        print('wrong img:', main_idx, 'feat_i', feat_i)
                        return self[(idx+random.randint(1, 100))%self.lens]
                else:
                    np_idx = self.img_idx_dicts[feat_i][main_idx]
                    tmp_img_feats.append(self.img_feat_nps[feat_i][np_idx])
            if isinstance(tmp_img_feats, list):
                tmp_img_feats = torch.unsqueeze(torch.from_numpy(np.concatenate(tmp_img_feats, axis=0)), 0)
            img_feats.append(tmp_img_feats)
            
            tmp_feats = []
            for feat_i in range(len(self.audio_idx_dicts)):
                if self.audio_idx_dicts[feat_i].get(main_idx) is None:
                    if i == 0:
                        print('wrong audio:', main_idx, 'feat_i', feat_i)
                        return self[(idx+random.randint(1, 100))%self.lens]
                    tmp_feats = wav_feats[-1]
                    break
                else:
                    np_idx = self.audio_idx_dicts[feat_i][main_idx]
                    tmp_feats.append(self.audio_feat_nps[feat_i][np_idx])
            if isinstance(tmp_feats, list):
                tmp_feats = torch.unsqueeze(torch.from_numpy(np.concatenate(tmp_feats, axis=0)), 0)
            wav_feats.append(tmp_feats)

            v = self.label_df.loc[main_idx].valence
            a = self.label_df.loc[main_idx].arousal
            vlabels.append(v)
            alabels.append(a)
            main_idxs.append(main_idx)
        
        img_feats = torch.unsqueeze(torch.cat(img_feats, dim=0), 0)
        wav_feats = torch.unsqueeze(torch.cat(wav_feats, dim=0), 0)
        return img_feats, wav_feats, vlabels, alabels, main_idxs


def IA_collate_fn(batch):
    new_audio_feats = []
    new_vlabels = []
    new_alabels = []
    new_img_feats = []
    new_main_idxs = []
    for img_feats, audio_feats, vlabels, alabels, main_idxs in batch:
        new_img_feats.append(img_feats)
        new_audio_feats.append(audio_feats)
        new_vlabels.extend(vlabels)
        new_alabels.extend(alabels)
        new_main_idxs.extend(main_idxs)
    new_img_feats = torch.cat(new_img_feats, dim=0)
    new_audio_feats = torch.cat(new_audio_feats, dim=0)
    new_vlabels = torch.Tensor(new_vlabels)
    new_alabels = torch.Tensor(new_alabels)
    return new_img_feats, new_audio_feats, new_vlabels, new_alabels, new_main_idxs

def get_loader(cfg):
    if cfg.Data.loader.transforms:
        train_data_aug, val_data_aug = eval(cfg.Data.loader.transforms)()
    else:
        train_data_aug, val_data_aug = get_transforms()

    IA_aff_wild_train_dataset = IA_AlignAffWild2(img_dir = cfg.Data.img_dir, 
                                                 seq_csv_file = cfg.Data.train_seq_file,
                                                 label_csv_file = cfg.Data.train_label_file,
                                                 img_feat_list = cfg.Data.train_img_feat_list,
                                                 audio_feat_list = cfg.Data.train_audio_feat_list,
                                                 transform=train_data_aug,
                                                 flag='train',
                                                 seq_len=cfg.Data.seq_len)
    IA_aff_wild_valid_dataset = IA_AlignAffWild2(img_dir = cfg.Data.img_dir,
                                                 seq_csv_file = cfg.Data.val_seq_file,
                                                 label_csv_file = cfg.Data.val_label_file,
                                                 img_feat_list = cfg.Data.val_img_feat_list,
                                                 audio_feat_list = cfg.Data.val_audio_feat_list,
                                                 transform=val_data_aug,
                                                 flag='val',
                                                 seq_len=cfg.Data.seq_len)
    
    IA_aff_wild_train_loader = DataLoader(IA_aff_wild_train_dataset, batch_size=cfg.Data.loader.batch_size,
                                             num_workers=cfg.Data.loader.num_workers, shuffle=True, pin_memory=True, collate_fn=IA_collate_fn, drop_last=True)
    if cfg.Data.loader.test_batch_size:
        test_batch_size = cfg.Data.loader.test_batch_size
    else:
        test_batch_size = cfg.Data.loader.batch_size
    IA_aff_wild_valid_loader = DataLoader(IA_aff_wild_valid_dataset, batch_size=test_batch_size,
                                             num_workers=cfg.Data.loader.num_workers, shuffle=False, pin_memory=True, collate_fn=IA_collate_fn, drop_last=True) # fixme [True|False]
    return IA_aff_wild_train_loader, IA_aff_wild_valid_loader

