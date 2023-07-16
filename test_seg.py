import argparse
import os
import time
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import logging
import yaml
from pathlib import Path
import nibabel as nib
from models.resunet import ResUNet, DAResNet3d
from models.deepmedic import DeepMedic
from losses.loss import FCCELoss, FocalLoss, DiceLoss
from dataprocess.dataloader import TrainDataset, TestDataset
#from dataprocess.dualdata import TestDualDataset
from metrics.DiceEval import diceEval, AverageMeter

import pdb

NII_FOLDER = '/home/jhebu/dataset/CoronaryArtery/challengedata/'
task_path = '/home/jhebu/PycharmProject/ArterySeg_AAAI/experiment/da_resnet34_3d-ce_loss+dice_loss+mse_loss-2019-09-22_18-11/'
PATCH_SIZE = (32,32,32)

def parse_args():
    parser = argparse.ArgumentParser('ArterySeg')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='which gpu to select')
    return parser.parse_args()


def main(args):
    
    '''CONFIG LOAD'''
    config_path = './config.yaml'
    config_reader = open(config_path)
    cfg = yaml.load(config_reader, Loader=yaml.FullLoader)
    
    model_path = os.path.join(task_path,'models/seg_model_epoch_2672.pth')
    pred_path = os.path.join(task_path,'prediction')
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in args.gpu)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    
    '''model'''
    if cfg['MODEL']['NAME'] == 'da_resnet34_3d':
        model = DAResNet3d(cfg['MODEL']['NCLASS'], k = 32)
    elif cfg['MODEL']['NAME'] == 'da_resunet':
        model = DAResUNet(cfg['MODEL']['NCLASS'], k=16)
    elif cfg['MODEL']['NAME'] == 'resunet':
        model = ResUNet(cfg['MODEL']['NCLASS'], k = 32)
    elif cfg['MODEL']['NAME'] == 'deepmedic':
        model = DeepMedic(cfg['MODEL']['NCLASS'])
    else:
        pass
    
    device_ids = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.cuda(device_ids)
    model = torch.nn.DataParallel(model)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    if cfg['MODEL']['NAME'] == 'deepmedic':
        TEST_DATASET = TestDualDataset(train_lst=cfg['DATASET']['TRAIN_LIST'], 
                                 flip=True, patch_size = PATCH_SIZE)
    else:
        TEST_DATASET  = TestDataset(test_lst=NII_FOLDER+'test.lst', flip=True, patch_size = PATCH_SIZE)
    
    testdataloader  = DataLoader(TEST_DATASET, batch_size=1, 
                                 shuffle=False,num_workers=int(cfg['TEST']['WORKER']),drop_last = False)
    
    
    with torch.no_grad():
        for subject in TEST_DATASET.subjects:
            image_shape = TEST_DATASET.sample(subject)
            print("The subject ID is: " + subject)
            print("The number of test data is:", len(TEST_DATASET))
            
            full_pred_max = np.zeros(image_shape)
            full_pred_min = np.ones(image_shape)
            full_heatmap_max = np.zeros(image_shape)
            full_heatmap_min = np.ones(image_shape)*50
            for batch_id, (data, coord) in enumerate(testdataloader):
                if cfg['MODEL']['NAME'] == 'deepmedic':
                    image, sub_image = data
                    image, sub_image, mask = Variable(image.float()), Variable(sub_image.float())
                    image, sub_image, mask = image.cuda(), sub_image.cuda()
                    seg_pred = model((image,subimage))
                else:
                    image = Variable(data.float())
                    image = image.cuda()
                    image_pred = model(image)
                
                seg_pred = image_pred[:,0:2,:,:,:]
                heatmap_pred = image_pred[:,2,:,:,:]
                seg_pred = F.softmax(seg_pred, dim=1)
                seg_pred = seg_pred.cpu().numpy()
                heatmap_pred = heatmap_pred.cpu().numpy()
                
                patch_pred_max = full_pred_max[coord[0]:coord[0]+PATCH_SIZE[0],
                                               coord[1]:coord[1]+PATCH_SIZE[1],
                                               coord[2]:coord[2]+PATCH_SIZE[2]]
                patch_pred_max = np.maximum(patch_pred_max, seg_pred[0][1])
                full_pred_max[coord[0]:coord[0]+PATCH_SIZE[0],
                              coord[1]:coord[1]+PATCH_SIZE[1],
                              coord[2]:coord[2]+PATCH_SIZE[2]] = patch_pred_max

                patch_pred_min = full_pred_min[coord[0]:coord[0] + PATCH_SIZE[0],
                                               coord[1]:coord[1] + PATCH_SIZE[1],
                                               coord[2]:coord[2] + PATCH_SIZE[2]]
                patch_pred_min = np.minimum(patch_pred_min, seg_pred[0][1])
                full_pred_min[coord[0]:coord[0] + PATCH_SIZE[0],
                              coord[1]:coord[1] + PATCH_SIZE[1],
                              coord[2]:coord[2] + PATCH_SIZE[2]] = patch_pred_min
                
                
                patch_heatmap_max = full_heatmap_max[coord[0]:coord[0]+PATCH_SIZE[0],
                                               coord[1]:coord[1]+PATCH_SIZE[1],
                                               coord[2]:coord[2]+PATCH_SIZE[2]]
                patch_heatmap_max = np.maximum(patch_heatmap_max, heatmap_pred[0])
                full_heatmap_max[coord[0]:coord[0]+PATCH_SIZE[0],
                                 coord[1]:coord[1]+PATCH_SIZE[1],
                                 coord[2]:coord[2]+PATCH_SIZE[2]] = patch_heatmap_max

                patch_heatmap_min = full_heatmap_min[coord[0]:coord[0] + PATCH_SIZE[0],
                                               coord[1]:coord[1] + PATCH_SIZE[1],
                                               coord[2]:coord[2] + PATCH_SIZE[2]]
                patch_heatmap_min = np.minimum(patch_heatmap_min, heatmap_pred[0])
                full_heatmap_min[coord[0]:coord[0] + PATCH_SIZE[0],
                                 coord[1]:coord[1] + PATCH_SIZE[1],
                                 coord[2]:coord[2] + PATCH_SIZE[2]] = patch_heatmap_min
                
            full_pred_avg = (full_pred_min + full_pred_max)/2
            full_heatmap_avg = (full_heatmap_min + full_heatmap_max)/2
            
            
            
            #print(batch_id)
            object_path = os.path.join(NII_FOLDER, subject)
            img = nib.load(os.path.join(object_path,'image_resample.nii.gz'))
            #img = nib.load(os.path.join(NII_FOLDER, '%s.nii.gz' % subject))
            full_pred_max = np.transpose(full_pred_max, (1, 2, 0))
            full_pred_max = full_pred_max.astype('float32')
            affine = img.affine
            full_pred_img = nib.Nifti1Image(full_pred_max, affine)
            nib.save(full_pred_img, os.path.join(pred_path,'pred_max_'+subject+'.nii.gz'))

            full_pred_min = np.transpose(full_pred_min, (1, 2, 0))
            full_pred_min = full_pred_min.astype('float32')
            affine = img.affine
            full_pred_img = nib.Nifti1Image(full_pred_min, affine)
            nib.save(full_pred_img, os.path.join(pred_path,'pred_min_'+subject+'.nii.gz'))

            full_pred_avg = np.transpose(full_pred_avg, (1, 2, 0))
            full_pred_avg = full_pred_avg.astype('float32')
            affine = img.affine
            full_pred_img = nib.Nifti1Image(full_pred_avg, affine)
            nib.save(full_pred_img, os.path.join(pred_path,'pred_avg_'+subject+'.nii.gz'))

            full_heatmap_avg = np.transpose(full_heatmap_avg, (1, 2, 0))
            full_heatmap_avg = full_heatmap_avg.astype('float32')
            affine = img.affine
            full_heatmap_img = nib.Nifti1Image(full_heatmap_avg, affine)
            nib.save(full_heatmap_img, os.path.join(pred_path,'heatmap_avg_'+subject+'.nii.gz'))

            
            
if __name__ == '__main__':
    args = parse_args()
    main(args)