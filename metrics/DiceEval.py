from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import time

class diceEval:
    
    def __init__(self, nClasses):
        
        self.nClasses = nClasses
        self.reset()
    
    def reset(self):
        
        self.total_dice = 0
        self.total_num = 0

    def addBatch(self, predict, gt):
        batch_size = predict.size(0)
        dice = torch.zeros([predict.size(0), self.nClasses])
        
        dice = dice.to(torch.device('cuda'))
        
        for i in range(1, self.nClasses):
            
            input_i = (predict == i).float().view(batch_size, -1)
            target_i = (gt == i).float().view(batch_size, -1)
            
            tt = time.time()
            num = (input_i * target_i)
            num = torch.sum(num, dim=1)
            
            den1 = torch.sum(input_i, dim=1)
            den2 = torch.sum(target_i, dim=1)
            
            epsilon = 1e-6
            
            dice[:, i] = (2 * num + epsilon) / (den1 + den2 + epsilon)
        
        dice = dice[:, 1:]
        dice = torch.sum(dice, dim=1)
        dice = dice / (self.nClasses - 1)
        self.total_dice += torch.sum(dice).item()
        self.total_num += batch_size
    
    def getMetric(self):
        
        epsilon = 1e-8
 
        return self.total_dice / (self.total_num + epsilon)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0 if (self.count < 1e-5) else (self.sum / self.count)