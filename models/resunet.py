
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from models.attention import PAM_Module, CAM_Module
import pdb

def norm(planes, mode='bn', groups=16):
    
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()
    
class CBR(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CBR, self).__init__()
        
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        self.bn = norm(nOut)
        self.act = nn.ReLU(True)
        
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        
        return output

class CB(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CB, self).__init__()
        
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        self.bn = norm(nOut)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        
        return output

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(C, self).__init__()
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
    def forward(self, input):
        return self.conv(input)
        
class BR(nn.Module):
    def __init__(self, nIn):
        super(BR, self).__init__()
        
        self.bn = norm(nIn)
        self.act = nn.ReLU(True)
    def forward(self, input):
        return self.act(self.bn(input))

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, nIn, nOut, kernel_size=(3,3,3), prob=0.03, stride=1, dilation=1):
        
        super(BasicBlock, self).__init__()
        
        self.c1 = CBR(nIn, nOut, kernel_size, stride, dilation)
        self.c2 = CB(nOut, nOut, kernel_size, 1, dilation)
        self.act = nn.ReLU(True)
        
        self.downsample=None
        if nIn != nOut or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(nIn, nOut, kernel_size=1, stride=stride, bias=False),
                norm(nOut)
            )
            
    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        if self.downsample is not None:
            input = self.downsample(input)

        output = output + input
        output = self.act(output)

        return output

class DownSample(nn.Module):
    def __init__(self, nIn, nOut, pool='max'):
        super(DownSample, self).__init__()
        
        if pool == 'conv':
            self.pool = CBR(nIn, nOut, 3, 2)
        elif pool == 'max':
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool = pool
            if nIn != nOut:
                self.pool = nn.Sequential(pool, CBR(nIn, nOut, 1, 1))
    
    def forward(self, input):
        output = self.pool(input)
        return output
    
class Upsample(nn.Module):
    
    def __init__(self, nIn, nOut):
        super(Upsample, self).__init__()
        self.conv = CBR(nIn, nOut)
        
    def forward(self, x):
        p = F.upsample(x, scale_factor=2, mode='trilinear')
        return self.conv(p)

class ResUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(ResUNet, self).__init__()
        
        
        
        self.layer0 = CBR(1, k, stride = 7, dilation = 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        #self.br_1 = BR(k+2*k)
        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k, dilation = 1)
        )
        
        self.pool2 = DownSample(2*k, 2*k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k, dilation = 1)
        )

        self.pool3 = DownSample(4*k, 4*k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k, dilation=1),
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4)
        )
        #self.br_3 = BR(7*k+8*k)
        sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
        self.class3 = PSPModule(8*k, 8*k, sizes) if psp else CBR(8*k, 8*k, dilation = 1)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):
        #pdb.set_trace()
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        
        output = self.class3(output3)
        output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return output


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1), 
                                  nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                  nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                  nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        return sasc_output
    
    
    
class DAResNet3d(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(DAResNet3d, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,   k, 3, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, kernel_size=(3,3,3), stride=2)
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, kernel_size=(3,3,3), stride=2)
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k, 8*k, kernel_size=(2,2,2), stride=(2,2,2)),
            norm(8*k),
            nn.ReLU(inplace=False)
        )
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 4*k, (3,3,3))
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=(2,2,2), stride=(2,2,2)),
            norm(4*k),
            nn.ReLU(inplace=False)
        )
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 2*k, (3,3,3)),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=False)
        )
        
        self.class1 = nn.Sequential(
            CBR(k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses+1, kernel_size=1, bias=False),
        )
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.class4(self.layer4(x3))

        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        x = self.class1(torch.cat([self.up1(x), x1], 1))
        
        x = F.interpolate(x, x_size[2:], mode='trilinear', align_corners=True)
        
        return x



    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    
class DAResUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(DAResUNet, self).__init__()
        
        self.layer0 = CBR(1, k, (7,7,7), 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        
        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k, (1,1,1))
        )
        
        self.pool2 = DownSample(2*k, 2*k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k, (1,1,1))
        )
        
        self.pool3 = DownSample(4*k, 4*k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k, dilation=1),
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4)
        )
        
        sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
        self.class3 = DANetHead(8*k, 8*k)
        

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')
    
        self._init_weight()
        
    def  _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        
        output = self.class3(output3)
        output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return output