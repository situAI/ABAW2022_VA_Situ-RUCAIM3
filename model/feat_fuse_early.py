import torch
import torch.nn as nn


class FeatFuseEarly(nn.Module):
    def __init__(self, cfg):
        super(FeatFuseEarly, self).__init__()
        print('cfg.Model.pretrain_path', cfg.Model.pretrain_path)
        self.feat_fc = nn.Conv1d(cfg.Model.img_dim+cfg.Model.audio_dim, 1024, 1, padding=0)
        self.decoder = TransEncoder(inc=1024, outc=512, dropout=cfg.Solver.dropout, nheads=4, nlayer=4)
        self.vhead = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, cfg.Model.bin_num),
                )
        self.ahead = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, cfg.Model.bin_num),
                )
        self.dropout = nn.Dropout(p=cfg.Solver.dropout)

    def forward(self, x):
        
        img_feat, audio_feat = x
        bs, seq_len, _ = img_feat.shape

        img_feat = torch.transpose(img_feat, 1, 2)        
        audio_feat = torch.transpose(audio_feat, 1, 2)        
        feat = torch.cat([img_feat, audio_feat], dim=1)
        feat = self.feat_fc(feat)
        out = self.decoder(feat)
        
        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        vout = self.vhead(out)
        aout = self.ahead(out)
        return img_feat, vout, aout

