
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import label
import numpy as np
import pdb

class FCCELoss(nn.Module):
    
    def __init__(self, cfg):
        super(FCCELoss, self).__init__()
        
        
        class_weight = None
        if 'CLASS_WEIGHT' in cfg['LOSS'].keys() and len(cfg['LOSS']['CLASS_WEIGHT']) > 0:
            class_weight = cfg['LOSS']['CLASS_WEIGHT']
            class_weight = torch.FloatTensor(class_weight).cuda()
        
        ignore_index = -100
        if 'IGNORE_INDEX' in cfg['LOSS'].keys():
            ignore_index = cfg['LOSS']['IGNORE_INDEX']
        
        reduction = 'elementwise_mean'
        if 'REDUCTION' in cfg['LOSS'].keys():
            reduction = cfg['LOSS']['REDUCTION']
            
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)#, reduction=reduction)
   
    def forward(self, input, target):
        
        
        return self.ce_loss(input, target)

class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            alpha = cfg.LOSS.WEIGHT
        
        if isinstance(alpha,(float,int,long)): self.alpha = torch.FloatTensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.FloatTensor(alpha)
        self.size_average = size_average
        
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())
        pt = logpt.exp()
        
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)
        
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def compute_dice(input, target):
    dice = torch.zeros([input.size(0), input.size(1)]).to(torch.device("cuda"))
    pred = F.softmax(input, dim=1)
    for i in range(1, input.size(1)):
        input_i = pred[:,i,...].contiguous().view(input.size(0), -1)
        target_i = (target == i).float().view(input.size(0), -1)

        num = (input_i * target_i)
        num = torch.sum(num, dim=1)

        den1 = input_i * input_i
        den1 = torch.sum(den1, dim=1)

        den2 = target_i * target_i
        den2 = torch.sum(den2, dim=1)

        epsilon = 1e-6
        dice[:, i] = (2 * num + epsilon) / (den1 + den2 + epsilon + 1e-10)

    return dice
    
class DiceLoss(nn.Module):
   
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    
    def forward(self, input, target):
        
        dice = compute_dice(input, target)
        dice = dice[:, 1:] #we ignore bg dice val, and take the fg
        dice = torch.sum(dice, dim=1)
        dice = dice / (input.size(1) - 1)
        dice_total = -1.0 * torch.sum(dice) / dice.size(0) #divide by batch_sz
        return 1.0 + dice_total

class ELDiceLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(ELDiceLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        smooth = 1.0
        
        pred = F.softmax(input, dim=1)
        loss = 0 #torch.Tensor([0]).float().to(torch.device("cuda"))
        for i in range(1, pred.size(1)):
            pred_i = pred[:,i,:,:,:]
            target_i = (target == i).float()
            
            intersect = (pred_i * target_i).sum()
            union = torch.sum(pred_i) + torch.sum(target_i)
            dice = (2*intersect + smooth) / (union + smooth)
            
            if target_i.sum().item() != 0:
                loss += (-torch.log(dice))**self.gamma
            else:
                loss += 1 - dice
        loss = loss / (input.size(1) - 1)
        return loss
        
    
class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, input, target):
        dice = compute_dice(input, target)
        dice = dice[:, 1:]
        
        pt = dice.contiguous().view(target.size(0), -1)
        #assert (pt < 1).any() and (pt > 1e-12).any(), pt
        logpt = torch.log(pt)
        
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction = 'mean')
   
    def forward(self, input, target):
        
        
        return self.mse_loss(input, target)
        
        
        
class SurfaceLoss(nn.Module):
    
    def __init__(self):
        super(SurfaceLoss, self).__init__()
    
    def forward(self, input, dist_map):
        
        pc = F.softmax(input, dim=1)[:,1,:,:,:]
        loss = pc * dist_map
        
        return loss.mean()
    
    
class MixLoss(nn.Module):
    
    def __init__(self, loss, cfg):
        
        super(MixLoss, self).__init__()
        self.loss = loss
        self.cfg = cfg
        
    def forward(self, input, target, heatmap):
        
        if 'WEIGHT' in self.cfg['LOSS'].keys() and len(self.cfg['LOSS']['WEIGHT']) > 0:
            weight = self.cfg['LOSS']['WEIGHT']
        else:
            weight = [1.0] * len(self.loss)
        
        loss = 0
        loss_class = 0
        loss_regress = 0
        for i in range(len(self.loss)):
            if self.loss[i].__class__.__name__ != 'MSELoss':
                input_class = input[:,0:2,:,:,:]
                loss_class += weight[i] * self.loss[i](input_class, target)
            else:
                input_class = input[:,2,:,:,:]
                loss_regress += weight[i] * self.loss[i](input_class, heatmap)
        loss = loss_class + loss_regress
        
        return loss
            
        
        