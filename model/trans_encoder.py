import torch
import torch.nn as nn


class TransEncoder(nn.Module):
    def __init__(self, inc=512, outc=512, dropout=0.6, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.d_model = outc
        self.dim_feedforward = outc
        self.dropout = dropout
        self.conv1 = nn.Conv1d(inc, self.d_model, kernel_size=1, stride=1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        return out

