import torch
import numpy as np
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.regressor import FcRegressor
from .networks.lstm_encoder import LSTMEncoder, BiLSTMEncoder
from .networks.fc_encoder import FcEncoder
from utils.loss import CCCLoss, MSELoss, MultipleLoss

class FcLstmXLModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, choices=['arousal', 'valence', 'both'])
        parser.add_argument('--loss_type', type=str, default='mse', nargs='+',
                            choices=['mse', 'ccc', 'batch_ccc', 'amse', 'vmse', 'accc', 'vccc', 'batch_accc',
                                     'batch_vccc', 'ce'])
        parser.add_argument('--loss_weights', type=float, default=1, nargs='+')
        parser.add_argument('--cls_loss', default=False, action='store_true', help='whether to cls and average as loss')
        parser.add_argument('--cls_weighted', default=False, action='store_true', help='whether to use weighted cls')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['_fc', '_seq', '_reg']
        self.pretrained_model = []
        self.max_seq_len = opt.max_seq_len
        # net seq
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        # net fc fusion
        self.net_fc = FcEncoder(opt.input_dim, [opt.hidden_size], dropout=0.1, dropout_input=False)
        self.net_seq = LSTMEncoder(opt.hidden_size, opt.hidden_size)
        self.hidden_mul = 1

        self.target_name = opt.target
        self.loss_type = opt.loss_type
        self.cls_loss = opt.cls_loss

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size

        output_dim = 20 if self.cls_loss else 1
        if self.target_name == 'both':
            output_dim *= 2

        self.net_reg = FcRegressor(opt.hidden_size * self.hidden_mul, layers, output_dim, dropout=opt.dropout_rate)

        # settings
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

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        previous_h = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        previous_c = torch.zeros(self.hidden_mul, batch_size, self.hidden_size).float().to(self.device) 
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction, (previous_h, previous_c) = self.forward_step(feature_step, (previous_h, previous_c))
            previous_h = previous_h.detach()
            previous_c = previous_c.detach()
            if self.hidden_mul == 2:
                previous_h[1].fill_(0.0)
                previous_c[1].fill_(0.0)
            
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    
    def forward_step(self, input, states):
        fusion = self.net_fc(input)
        hidden, (h, c) = self.net_seq(fusion, states)
        prediction, _ = self.net_reg(hidden)
        if self.cls_loss:
            if self.target_name == 'both':
                prediction = prediction.reshape(prediction.shape[:-1] + (20, 2))
            else:
                prediction = prediction.unsqueeze(-1)
            prediction = F.softmax(prediction, dim=-2)
            weights = torch.FloatTensor([(-0.95 + i/10.0) for i in range(20)]).reshape(1, 1, -1, 1).cuda()
            prediction = torch.sum(prediction * weights, dim=-2)
        return prediction, (h, c)
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        #pred: [B, L, 1] or [B, L, 2]
        #target: [B, L] or [B, L, 2]
        #mask: [B, L]

        mask = mask.unsqueeze(-1) # -> [B, L, 1]
        if self.target_name != 'both':
            target = target.unsqueeze(-1)  # -> [B, L, 1]
        else:
            mask = mask.expand(mask.shape[0], mask.shape[1], 2) # -> [B, L, 2]

        batch_size = target.size(0)
        self.loss_MSE = self.criterion_reg(pred, target, mask)
        #self.loss_MSE = self.loss_MSE / batch_size
        self.loss_MSE.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)