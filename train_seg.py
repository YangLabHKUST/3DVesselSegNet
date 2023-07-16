import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import logging
import yaml
from pathlib import Path
from models.resunet import ResUNet, DAResNet3d, DAResUNet
from models.deepmedic import DeepMedic
from losses.loss import FCCELoss, FocalLoss, DiceLoss, MixLoss, FocalDiceLoss, ELDiceLoss, MSELoss
from dataprocess.dataloader import TrainDataset, ValidateDataset
from dataprocess.dualdata import TrainDualDataset, ValidateDualDataset
from metrics.DiceEval import diceEval, AverageMeter
from metrics.AUCEval import aucEval
from metrics.IOUEval import iouEval
import pdb


def parse_args():
    parser = argparse.ArgumentParser('ArterySeg')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='which gpu to select')
    return parser.parse_args()

def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in args.gpu)
    
    '''CONFIG LOAD'''
    config_path = './config.yaml'
    config_reader = open(config_path)
    cfg = yaml.load(config_reader, Loader=yaml.FullLoader)

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%s'%cfg['MODEL']['NAME']+ '-%s-'%cfg['LOSS']['TYPE'] + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    model_dir = file_dir.joinpath('models/')
    model_dir.mkdir(exist_ok=True)
    
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(cfg['MODEL']['NAME'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%cfg['MODEL']['NAME'])
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('------------------TRANING---------------------------')
    logger.info('PARAMETER ...')
    logger.info(cfg)

    '''Dataset'''
    if cfg['MODEL']['NAME'] == 'deepmedic':
        TRAIN_DATASET = TrainDualDataset(train_lst=cfg['DATASET']['TRAIN_LIST'], 
                                 flip=True, patch_size = cfg['DATASET']['PATCH_SIZE'])
        VALIDATE_DATASET  = ValidateDualDataset(validate_lst=cfg['DATASET']['VALIDATE_LIST'], 
                                flip=True, patch_size = cfg['DATASET']['PATCH_SIZE'])
    else:
        TRAIN_DATASET = TrainDataset(train_lst=cfg['DATASET']['TRAIN_LIST'], 
                                 flip=False, patch_size = cfg['DATASET']['PATCH_SIZE'])
        VALIDATE_DATASET  = ValidateDataset(validate_lst=cfg['DATASET']['VALIDATE_LIST'], 
                                flip=False, patch_size = cfg['DATASET']['PATCH_SIZE'])
    TRAIN_DATASET.sample(cfg['DATASET']['SAMPLE_NUMBER'])
    VALIDATE_DATASET.sample()
    traindataloader = DataLoader(TRAIN_DATASET,batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                 shuffle=True,num_workers=int(cfg['TRAIN']['WORKER']),drop_last = True)
    validatedataloader  = DataLoader(VALIDATE_DATASET, batch_size=cfg['VALIDATE']['BATCH_SIZE'], 
                                 shuffle=True,num_workers=int(cfg['VALIDATE']['WORKER']),drop_last = False)
    

    print("The number of training data is:",len(TRAIN_DATASET))
    logger.info("The number of training data is:%d",len(TRAIN_DATASET))
    print("The number of validate data is:", len(VALIDATE_DATASET))
    logger.info("The number of validate data is:%d", len(VALIDATE_DATASET))
    
    
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
        
    
        
    '''Optimizer'''
    if cfg['SOLVER']['NAME'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif cfg['SOLVER']['NAME'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=cfg['SOLVER']['LEARN_RATE'], 
                                     betas=(0.9, 0.999),eps=1e-08, 
                                     weight_decay=cfg['SOLVER']['WEIGTH_DECAY'])
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['SOLVER']['LR_STEPS'], gamma=0.5)

    '''Loss'''
    if '+' in cfg['LOSS']['TYPE']:
        loss = nn.ModuleList([])
        loss_list = cfg['LOSS']['TYPE'].split('+')
        for t in loss_list:
            if t == 'ce_loss':
                loss.append(FCCELoss(cfg))
            elif t == 'focal_loss':
                loss.append(FocalLoss())
            elif t == 'dice_loss':
                loss.append(DiceLoss())
            elif t == 'focaldice_loss':
                loss.append(FocalDiceLoss())
            elif t == 'eldice_loss':
                loss.append(ELDiceLoss())
            elif t == 'mse_loss':
                loss.append(MSELoss())
            else:
                raise NameError('Unkown loss type')
        criterion = MixLoss(loss,cfg)
    else:
        if cfg['LOSS']['TYPE'] == 'ce_loss':
            criterion = FCCELoss(cfg)
        elif cfg['LOSS']['TYPE'] == 'focal_loss':
            criterion = FocalLoss()
        elif cfg['LOSS']['TYPE'] == 'dice_loss':
            criterion = DiceLoss()
        else:
            raise NameError('Unkown loss type')
    
    '''GPU selection and multi-GPU'''
    torch.backends.cudnn.benchmark = True
    device_ids = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda(device_ids)
    model = torch.nn.DataParallel(model)
    
    
    '''pretrain'''
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epoch = int(pretrain[-7:-4]) if args.pretrain is not None else 0
    init_epoch = 0
    
    '''Init'''
    acc = 0
    LEARNING_RATE_CLIP = 1e-5
    num_batch = int(len(TRAIN_DATASET) / cfg['TRAIN']['BATCH_SIZE'])
    blue = lambda x: '\033[94m' + x + '\033[0m'
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    dice_best = 0
    
    
    '''Epoch'''
    for epoch in range(init_epoch,cfg['SOLVER']['EPOCHS']):
        
        diceEvalTrain = diceEval(cfg['MODEL']['NCLASS'])
        #aucEvalTrain = aucEval(cfg['MODEL']['NCLASS'])
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        
        torch.cuda.empty_cache()
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate: %f' % lr)
        logger.info('Learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Train
        t0 = time.time()
        TRAIN_DATASET.sample(cfg['DATASET']['SAMPLE_NUMBER'])
        
        for batch_id, (data, mask, heatmap) in enumerate(traindataloader):
            if cfg['MODEL']['NAME'] == 'deepmedic':
                image, sub_image = data
                image, sub_image, mask = Variable(image.float()), Variable(sub_image.float()), Variable(mask.long())
                image, sub_image, mask = image.cuda(), sub_image.cuda(), mask.cuda()
                optimizer.zero_grad()
                
                seg_pred = model((image,sub_image))
            else:
                image, mask, heatmap = Variable(data.float()), Variable(mask.long()), Variable(mask.float())
                image, mask, heatmap = image.cuda(), mask.cuda(), heatmap.cuda()
                optimizer.zero_grad()
                
                seg_pred = model(image)

            loss = criterion(seg_pred, mask, heatmap)
            loss.backward()
            optimizer.step()
            
            if batch_id % 10 == 0:
                diceEvalTrain.addBatch(seg_pred.max(1)[1], mask)
                #aucEvalTrain.addBatch(seg_pred.max(1)[1], mask)
            t1 = time.time()
            meters['time'].update(t1-t0)
            meters['loss'].update(loss.item(), mask.size(0))
            
            # Print
            if batch_id % 50 == 0:
                dice = diceEvalTrain.getMetric()
                #acc, pos_recall, neg_recall, precision = aucEvalTrain.getMetric()
                logger.info('epoch=%03d, batch_id=%03d, loss=%.4f, dice=%.4f' % \
                             (epoch, batch_id, meters['loss'].avg, dice))
                print('epoch=%03d, batch_id=%03d, loss=%.4f, dice=%.4f' % \
                             (epoch, batch_id, meters['loss'].avg, dice))
            t0 = time.time()
            
        
        # Evaluation
        diceEvalVal = diceEval(cfg['MODEL']['NCLASS'])
        aucEvalVal = aucEval(cfg['MODEL']['NCLASS'])
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        
        VALIDATE_DATASET.sample()
        model = model.eval()
        t0 = time.time()
        with torch.no_grad():
            for i, (data, mask, heatmap) in enumerate(validatedataloader):
                if cfg['MODEL']['NAME'] == 'deepmedic':
                    image, sub_image = data
                    image, sub_image, mask = Variable(image.float()), Variable(sub_image.float()), Variable(mask.long())
                    image, sub_image, mask = image.cuda(), sub_image.cuda(), mask.cuda()
                    
                    seg_pred = model((image, sub_image))
                else:
                    image = data
                    image, mask, heatmap = Variable(image.float()), Variable(mask.long()), Variable(heatmap.float())
                    image, mask, heatmap = image.cuda(), mask.cuda(), heatmap.cuda()
                
                    seg_pred = model(image)
                    
                loss = criterion(seg_pred, mask, heatmap)
                
                diceEvalVal.addBatch(seg_pred.max(1)[1], mask)
                aucEvalVal.addBatch(seg_pred.max(1)[1], mask)
                t1 = time.time()
            
                meters['loss'].update(loss.item(), mask.size(0))
                meters['time'].update(t1-t0)
            
                t0 = time.time()
        
        dice = diceEvalVal.getMetric()
        acc, pos_recall, neg_recall, precision = aucEvalVal.getMetric()
        logger.info('Validate: Time=%.3fms/batch, Loss=%.4f, Dice=%.4f, recall=%.4f, precision=%.4f'% \
                      (meters['time'].avg * 100, meters['loss'].avg, dice, pos_recall, precision))
        print('\033[94m'+'Validate: Time=%.3fms/batch, Loss=%.4f, Dice=%.4f, recall=%.4f, precision=%.4f'% \
                      (meters['time'].avg * 100, meters['loss'].avg, dice, pos_recall, precision)+'\033[0m' )

        dice_temp = dice
        if dice_temp > dice_best:
            dice_best = dice_temp
            torch.save(model.state_dict(), '%s/seg_model_best_choice.pth' % (model_dir))
        print(blue('%s %f' % ('Best Dice:', dice_best)))
        logger.info('%s %f' % (blue('Best Dice:'), dice_best))

        #torch.save(model.state_dict(), '%s/seg_model_epoch_%03d.pth' % (model_dir,epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args)

