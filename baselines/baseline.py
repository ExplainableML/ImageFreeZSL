import torch
import utility.util as util
from regressor import REGRESSOR
import torchvision
import torch.nn as nn
import os

class Baseline(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
    
    def evaluate_weights(self, pred_weights):   
        self.unseen_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim]
        self.unseen_model.fc.bias.data[:] = pred_weights[:, self.input_dim]

        self.ext_model.fc.weight.data[len(self.seenclasses):, :] = pred_weights[:, :self.input_dim] 
        self.ext_model.fc.bias.data[len(self.seenclasses):] = pred_weights[:, self.input_dim]

        if self.opt.zst:
            self.acc_target, self.acc_zst_unseen = self.val_zst()
        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = self.val_gzsl()