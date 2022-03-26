import torch
import numpy as np
import os
import math
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.regressor import FcRegressor
from .networks.transformer import TransformerEncoder
from utils.loss import CCCLoss, MSELoss, MultipleLoss
from utils.bins import get_center_and_bounds


class TransformerAttentionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of transformer')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='transformer encoder hidden states')
        parser.add_argument('--num_layers', default=4, type=int, help='number of transformer encoder layers')
        parser.add_argument('--ffn_dim', default=1024, type=int, help='dimension of FFN layer of transformer encoder')
        parser.add_argument('--nhead', default=4, type=int, help='number of heads of transformer encoder')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, choices=['valence', 'arousal', 'both'], help='one of [arousal, valence]')
        parser.add_argument('--use_pe', action='store_true', help='whether to use position encoding')
        parser.add_argument('--loss_type', type=str, default='mse', nargs='+',
                            choices=['mse', 'ccc', 'batch_ccc', 'amse', 'vmse', 'accc', 'vccc', 'batch_accc',
                                     'batch_vccc', 'ce'])
        parser.add_argument('--loss_weights', type=float, default=1, nargs='+')
        parser.add_argument('--cls_loss', default=False, action='store_true', help='whether to cls and average as loss')
        parser.add_argument('--cls_weighted', default=False, action='store_true', help='whether to use weighted cls')
        parser.add_argument('--former_attention', default='None', type=str, choices=['single', 'multihead', 'None'], help='whether to use the former seg feature')
        parser.add_argument('--mem_avg_len', type=int, default=50, help='the len of average mem')
        parser.add_argument('--max_mem_len', type=int, default=-1, help='the max len of mem')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['reg']
        self.model_names = ['_seq', '_reg', '_attn']
        self.pretrained_model = []
        self.max_seq_len = opt.max_seq_len
        self.use_pe = opt.use_pe
        self.loss_type = opt.loss_type
        self.net_attn = torch.nn.MultiheadAttention(embed_dim=opt.hidden_size, num_heads=4).to(self.device)
        self.former_attention = opt.former_attention
        self.mem_avg_len = opt.mem_avg_len
        self.max_mem_len = opt.max_mem_len

        # net seq (already include a linear projection before the transformer encoder)
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        if 'ce' in opt.loss_type:
            opt.cls_loss = True
        
        self.net_seq = TransformerEncoder(opt.input_dim, opt.num_layers, opt.nhead, \
                                        dim_feedforward=opt.ffn_dim, affine=True, \
                                        affine_dim=opt.hidden_size, use_pe=self.use_pe)

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.target_name = opt.target
        self.loss_type = opt.loss_type
        self.cls_loss = opt.cls_loss

        bin_centers, bin_bounds = get_center_and_bounds(opt.cls_weighted)
        self.bin_centers = dict([(key, np.array(value)) for key, value in bin_centers.items()])
        self.bin_bounds = dict([(key, np.array(value)) for key, value in bin_bounds.items()])

        self.hidden_size = opt.hidden_size
        output_dim = 22 if self.cls_loss else 1
        if self.target_name == 'both':
            output_dim *= 2

        self.net_reg = FcRegressor(opt.hidden_size, layers, output_dim, dropout=opt.dropout_rate)
        # settings
        if self.isTrain:
            if self.isTrain:
                self.criterion_reg = MultipleLoss(reduction='mean', loss_names=opt.loss_type, weights=opt.loss_weights)
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    
    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.feature = input['feature'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if load_label:
            if self.target_name == 'both':
                self.target = torch.stack([input['valence'], input['arousal']], dim=2).to(self.device)
            else:
                self.target = input[self.target_name].to(self.device)

            if self.cls_loss:
                if self.target_name == 'both':
                    #[B, L, 2]
                    self.cls_target = torch.stack([input['valence_cls'], input['arousal_cls']], dim=2).to(self.device)
                else:
                    self.cls_target = input[self.target_name].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        mems = None
        last_hidden_mems = None
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction, logits, last_hidden_mems = self.forward_step(feature_step, mask, last_hidden_mems)
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                cls_target = None if not self.cls_loss else self.cls_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask, logits, cls_target)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = torch.nn.Dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward_step(self, input, mask, last_hidden_mems):
        out, hidden_states = self.net_seq(input) # hidden_states: layers * (seq_len, bs, ft_dim)
        last_hidden = hidden_states[-1].transpose(0, 1) # (bs, seq_len, ft_dim)
        
        if last_hidden_mems is not None and self.former_attention != 'None':
            if self.former_attention == 'single':
                attention_last_hidden, _ = self.attention(query=last_hidden, value=last_hidden_mems, key=last_hidden_mems)
            elif self.former_attention == 'multihead':
                attention_last_hidden, _ = self.net_attn(query=last_hidden.permute(1, 0, 2), value=last_hidden_mems.permute(1, 0, 2), key=last_hidden_mems.permute(1, 0, 2))
                attention_last_hidden = attention_last_hidden.permute(1, 0, 2)
            attention_last_hidden = attention_last_hidden + last_hidden
        else:
            attention_last_hidden = last_hidden
        prediction, _ = self.net_reg(attention_last_hidden)
        
        logits = None
        if self.cls_loss:
            logits = prediction.reshape(prediction.shape[:-1] + (22, 2)).transpose(1, 2) #[B, L, 44] -> [B, L, 22, 2] -> [B, 22, L, 2]
            if self.target_name == 'both':
                prediction = prediction.reshape(prediction.shape[:-1] + (22, 2))
                weights = torch.cat([torch.FloatTensor(self.bin_centers['valence']),
                                     torch.FloatTensor(self.bin_centers['arousal'])]).reshape(1, 1, -1, 2).cuda()
            else:
                weights = torch.FloatTensor(self.bin_centers[self.target_name]).reshape(1, 1, -1, 1).cuda()
                prediction = prediction.unsqueeze(-1)
            prediction = F.softmax(prediction, dim=-2)
            prediction = torch.sum(prediction * weights, dim=-2)

        mems = torch.split(last_hidden.detach(), self.mem_avg_len, dim=1)
        meaned_mems = []
        for i in range(len(mems)):
            meaned_mems.append(torch.mean(mems[i], dim=1, keepdim=True))
        meaned_mems = torch.cat(meaned_mems, dim=1)
        if last_hidden_mems is not None:
            meaned_mems = torch.cat([last_hidden_mems, meaned_mems], dim=1)
            if self.max_mem_len != -1 and meaned_mems.size(1) > self.max_mem_len:
                meaned_mems = meaned_mems[:, -self.max_mem_len:, :]
        
        return prediction, logits, meaned_mems

    def _forward_step(self, input, mems):
        hidden_states, new_mems = self.net_seq(input, mems) # hidden_states: (seq_len, bs, ft_dim)
        last_hidden = hidden_states.transpose(0, 1) # (bs, seq_len, ft_dim)
        prediction, _ = self.net_reg(last_hidden)
        logits = None
        if self.cls_loss:
            logits = prediction.reshape(prediction.shape[:-1] + (22, 2)).transpose(1, 2) #[B, L, 44] -> [B, L, 22, 2] -> [B, 22, L, 2]
            if self.target_name == 'both':
                prediction = prediction.reshape(prediction.shape[:-1] + (22, 2))
                weights = torch.cat([torch.FloatTensor(self.bin_centers['valence']),
                                     torch.FloatTensor(self.bin_centers['arousal'])]).reshape(1, 1, -1, 2).cuda()
            else:
                weights = torch.FloatTensor(self.bin_centers[self.target_name]).reshape(1, 1, -1, 1).cuda()
                prediction = prediction.unsqueeze(-1)
            prediction = F.softmax(prediction, dim=-2)
            #weights = torch.FloatTensor([(-1.0 + i/10.0) for i in range(21)]).reshape(1, 1, -1, 1).cuda()
            prediction = torch.sum(prediction * weights, dim=-2)
        return prediction, logits, new_mems
   
    def backward_step(self, pred, target, mask, logits=None, cls_target=None):
        """Calculate the loss for back propagation"""
        mask = mask.unsqueeze(-1)  # -> [B, L, 1]
        if self.target_name != 'both':
            target = target.unsqueeze(-1)  # -> [B, L, 1]
        else:
            mask = mask.expand(mask.shape[0], mask.shape[1], 2)  # -> [B, L, 2]

        self.loss_reg = self.criterion_reg(pred, target, mask, logits, cls_target)
        self.loss_reg.backward(retain_graph=False)
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)