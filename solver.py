import numpy as np
from model.model_zoo import *
import torch
import torch.nn as nn
import logging
from torch.optim import AdamW, Adam
import os
from data.FS_data import get_loader
from utils.loss import *
from utils.metric import concordance_correlation_coefficient
import torch.nn.functional as F
import json
from munch import DefaultMunch
import yaml

device_ids = [0,1,2,3]


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Solver:
    def __init__(self):
        config_path = 'config/te.yml'
        yaml_dict = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        cfg = DefaultMunch.fromDict(yaml_dict)
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(cfg.Log.log_file_path, cfg.Log.log_file_name),
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filemode='a')
        self.model = eval(cfg.Model.model_name)(cfg)
        if cfg.Model.pretrained_path:
            pretrain_dict = torch.load(cfg.Model.pretrained_path)
            model_dict = self.model.state_dict()
            pretrain_dict = {k.replace('module.', ''):v for k,v in pretrain_dict.items()}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
        if len(device_ids)>1:
            self.model = nn.DataParallel(self.model).cuda(device=device_ids[0])
        else:
            self.model = self.model.cuda()

        print("Model Loaded.........................")
        self.epoch = cfg.Solver.epoch
        self.lr = cfg.Solver.lr
        self.weight_decay = cfg.Solver.weight_decay
        
        self.optimizer = eval(cfg.Solver.optimizer)(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train_loader, self.valid_loader = get_loader(cfg)
        print('Data Loaded..........................')
        self.len_train_loader = len(self.train_loader)
        self.len_valid_loader = len(self.valid_loader)
        self.eval_every = len(self.train_loader)//4+1

        if cfg.Solver.loss == 'ccc':
            self.criterion = CCCLoss(cfg.Model.bin_num).cuda(device=device_ids[0])
        elif cfg.Solver.loss == 'ccc_mse_ce':    
            self.criterion = CCC_MSE_CE_Loss(cfg.Model.bin_num).cuda(device=device_ids[0])
        else:    
            self.criterion = CCC_CE_Loss(cfg.Model.bin_num).cuda(device=device_ids[0])
        self.bins = torch.Tensor(np.linspace(-1, 1, num=cfg.Model.bin_num)).cuda(device=device_ids[0])
        self.edges = np.linspace(-1, 1, num= cfg.Model.bin_num+1)

        self.mse = nn.MSELoss().cuda(device=device_ids[0])
        self.save_dir = cfg.Log.checkpoint_path

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logging.info("cfg:"+str(cfg))        
        

    def train(self):
        for t in range(self.epoch):
            logging.info("==============EPOCH {} START================".format(t + 1))
            self.model.train()
            label_a = list()
            label_v = list()
            pred_a = list()
            pred_v = list()
            v_rmse = 0.0
            a_rmse = 0.0

            for i, (img_feat, audio, valence, arousal, _) in enumerate(self.train_loader):
                img_feat = img_feat.cuda(device=device_ids[0])
                audio = audio.cuda(device=device_ids[0])
                valence = valence.cuda(device=device_ids[0])
                arousal = arousal.cuda(device=device_ids[0])
                self.optimizer.zero_grad()

                img_feat, v, a = self.model((img_feat, audio))
    
                if 'ce' in cfg.Solver.loss or 'ccc' in cfg.Solver.loss: 
                    v_loss = self.criterion(v, valence)
                    a_loss = self.criterion(a, arousal)
                
                if cfg.Model.bin_num!=1:
                    a = F.softmax(a, dim=-1)
                    a = (self.bins*a).sum(-1)
                    v = F.softmax(v, dim=-1)
                    v = (self.bins*v).sum(-1)
                else:
                    v = v.squeeze()
                    a = a.squeeze()

                final_loss = v_loss + a_loss

                vtmp = self.mse(valence, v)
                atmp = self.mse(arousal, a)
                v_rmse += torch.sqrt(vtmp)
                a_rmse += torch.sqrt(atmp)
                pred_a = pred_a + a.detach().cpu().tolist()
                label_a = label_a + arousal.detach().cpu().tolist()
                pred_v = pred_v + v.detach().cpu().tolist()
                label_v = label_v + valence.detach().cpu().tolist()

                iteration = i + 1

                if iteration % 20 == 0 or iteration == 1:
                    logging.info(
                        "epoch: {}/{}, iteration: {}/{}, final loss: {:.4f}, v_loss: {:.4f}, a_loss: {:.4f}".format(
                            t + 1, self.epoch, iteration, self.len_train_loader, final_loss, v_loss, a_loss))

                if iteration % self.eval_every == 0:
                    with torch.no_grad():
                        self.model.eval()
                        self.valid(self.cfg.checkpoint_path, t+1, iteration)
                        torch.save(self.model.state_dict(), '%s/ckpt_epoch_%s_iter_%d.pt' % (self.save_dir, str(t + 1), iteration))
                        self.model.train()

                final_loss.backward()
                self.optimizer.step()

            v_rmse = v_rmse / self.len_train_loader
            a_rmse = a_rmse / self.len_train_loader
            a_ccc = concordance_correlation_coefficient(pred_a, label_a)
            v_ccc = concordance_correlation_coefficient(pred_v, label_v)

            logging.info(
                "EPOCH: {}/{}, TRAIN VALENCE RMSE: {:.4f}, TRAIN VALENCE CCC: {:.4f}, TRAIN AROUSAL RMSE: {:.4f}, TRAIN AROUSAL CCC: {:.4f}".format(
                    t + 1, self.epoch, v_rmse, v_ccc, a_rmse, a_ccc))

            self.model.eval()
            torch.save(self.model.state_dict(), '%s/ckpt_epoch_%s.pt' % (self.save_dir, str(t + 1)))

            with torch.no_grad():
                self.model.eval()
                self.valid(self.cfg.checkpoint_path, t+1, iteration)
    
    def convert_target(self, y):
        y_numpy = y.data.cpu().numpy()
        y_dig = np.digitize(y_numpy, self.edges) - 1
        y_dig[y_dig==cfg.Model.bin_num] = cfg.Model.bin_num -1
        y = torch.cuda.LongTensor(y_dig)
        return y
    

    @torch.no_grad()
    def valid(self, basename, epoch, iteration, viz=False, npy_path='./'):
        self.model.eval()
        pred_a = dict()
        pred_v = dict()
        label_a = dict()
        label_v = dict()
        v_rmse = 0.0
        a_rmse = 0.0

        for i, (img_feat, audio, valence, arousal, main_idx) in enumerate(self.valid_loader):
            img_feat = img_feat.cuda(device=device_ids[0])
            audio = audio.cuda(device=device_ids[0])
            valence = valence.cuda(device=device_ids[0])
            arousal = arousal.cuda(device=device_ids[0])
            _, v, a = self.model((img_feat, audio))
            
            self.edges = np.linspace(-1, 1, num= cfg.Model.bin_num+1)
            if cfg.Model.bin_num != 1:
                v = F.softmax(v, dim=-1)
                v = self.bins*v
                v = v.sum(-1)
                a = F.softmax(a, dim=-1)
                a = self.bins*a
                a = a.sum(-1)
            else:
                v = v.squeeze()
                a = a.squeeze()
            vtmp = self.mse(valence, v)
            atmp = self.mse(arousal, a)
            v_rmse += torch.sqrt(vtmp)
            a_rmse += torch.sqrt(atmp)
            
            v = v.cpu().numpy()
            a = a.cpu().numpy()
            valence = valence.cpu().numpy()
            arousal = arousal.cpu().numpy()
            flags = valence==-5
            v = np.delete(v, flags)
            a = np.delete(a, flags)
            main_idx = np.delete(main_idx, flags)
            valence = np.delete(valence, flags)
            arousal = np.delete(arousal, flags)
            for key_idx, key in enumerate(main_idx):
                if not pred_a.get(key):
                    pred_v[key] = [float(v[key_idx])]
                    pred_a[key] = [float(a[key_idx])]
                    label_v[key] = [float(valence[key_idx])]
                    label_a[key] = [float(arousal[key_idx])]
                else:
                    pred_v[key].append(float(v[key_idx]))
                    pred_a[key].append(float(a[key_idx]))

        k_list = np.array(sorted(pred_v.keys()))
        pred_v = [float(np.mean(value)) for k, value in sorted(pred_v.items(), key=lambda x:x[0])]
        pred_a = [float(np.mean(value)) for k, value in sorted(pred_a.items(), key=lambda x:x[0])]
        label_v = [value[0] for k, value in sorted(label_v.items(), key=lambda x:x[0])]
        label_a = [value[0] for k, value in sorted(label_a.items(), key=lambda x:x[0])]
        v_rmse = v_rmse / self.len_valid_loader
        a_rmse = a_rmse / self.len_valid_loader

        a_ccc = concordance_correlation_coefficient(pred_a, label_a)
        v_ccc = concordance_correlation_coefficient(pred_v, label_v)
        
        logging.info(
                "basename: {}, epoch: {}, iteration: {}, TEST VALENCE RMSE: {:.4f}, TEST_VALENCE CCC: {:.4f}, TEST AROUSAL RMSE: {:.4f}, TEST AROUSAL CCC: {:.4f}, ".format(
                basename, epoch, iteration, v_rmse, v_ccc, a_rmse, a_ccc
            ))


if __name__=='__main__':
    solver = Solver()
    solver.train()
