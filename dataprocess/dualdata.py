#echo $OMP_NUM_THREADS
import os
import numpy as np
import torch
from torch.utils import data
import random
import nibabel as nib
from scipy import misc
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool, Process, Queue
import time
import pdb


NII_FOLDER = '/home/jhebu/dataset/CoronaryArtery/zjudata'
#NII_FOLDER = '/home/jhebu/dataset/CoronaryArtery/challengedata'
#BLOOD_FOLDER = '/home/hejiafa/blood_dataset/cta/mask'
patch_size = (25, 25, 25)
sub_patch_size = (19, 19, 19)
target_size = (9, 9, 9)

class TrainDualDataset(object):
    
    def __init__(self, train_lst, flip, patch_size):
        
        self.train_lst = train_lst
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.target_size = target_size
        self.flip = flip

        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]   
            
        self.data_list = {}
        for subject in self.subjects:

            self.data_list[subject] = {}
            object_path = os.path.join(NII_FOLDER, subject)
            img = nib.load(os.path.join(object_path,'image_resample.nii.gz')).get_data()
            blood = nib.load(os.path.join(object_path,'mask_artery_resample.nii.gz')).get_data()
            while len(img.shape) !=3:
                img = img[:,:,:,0]
            while len(blood.shape) != 3:
                blood = blood[:,:,:,0]
            #pdb.set_trace()
            img = np.transpose(img, (2, 0, 1))
            img = (1.0 * (img - np.min(img)) / (np.max(img) - np.min(img)))*2 - 1.0

            blood = np.transpose(blood, (2, 0, 1))
            blood[blood > 0] = 1
            
            self.data_list[subject]['image'] = img
            self.data_list[subject]['mask'] = blood
            self.data_list[subject]['points'] = np.asarray(np.where(blood == 1)).T
            
    def sample(self, num_sample):
        
        self.coords = []
        for subject in self.subjects:
            
            image_shape = self.data_list[subject]['image'].shape
            index = self.data_list[subject]['points']
            
            num = len(index)
            PX, PY, PZ = self.patch_size
            sucess = 0
            for t in range(10000):
                i = random.randint(0, num-1)
                delet_x = random.randint(-5,5)
                delet_y = random.randint(-5,5)
                delet_z = random.randint(-5,5)
                
                x0, x1 = index[i][0]-PX//2+delet_x, index[i][0]+PX-PX//2+delet_x
                y0, y1 = index[i][1]-PY//2+delet_y, index[i][1]+PY-PY//2+delet_y
                z0, z1 = index[i][2]-PZ//2+delet_z, index[i][2]+PZ-PZ//2+delet_z
                
                if x0 < 0 or x1 > image_shape[0]:
                    continue
                if y0 < 0 or y1 > image_shape[1]:
                    continue
                if z0 < 0 or z1 > image_shape[2]:
                    continue
                self.coords.append([int(subject), x0, y0, z0])
                sucess += 1
                if sucess >= num_sample:
                    break
        
    def __getitem__(self, index):
        
        subject_i = self.coords[index][0]
        coord_i = self.coords[index][1:4]
        if subject_i < 10:
            subject = '0' + str(subject_i)
        else:
            subject = str(subject_i)
        
        image = self.data_list[subject]['image']
        mask = self.data_list[subject]['mask']
        
        image_patch = image[coord_i[0]:coord_i[0]+self.patch_size[0],
                            coord_i[1]:coord_i[1]+self.patch_size[1],
                            coord_i[2]:coord_i[2]+self.patch_size[2]].copy()
        sub_image_patch = image[coord_i[0]:coord_i[0]+self.sub_patch_size[0],
                               coord_i[1]:coord_i[1]+self.sub_patch_size[1],
                               coord_i[2]:coord_i[2]+self.sub_patch_size[2]].copy()
        mask_patch = mask[coord_i[0]:coord_i[0]+self.target_size[0],
                          coord_i[1]:coord_i[1]+self.target_size[1],
                          coord_i[2]:coord_i[2]+self.target_size[2]].copy()

        '''
        if self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            image_patch = image_patch[::, ::flip_x, ::flip_y]
            mask_patch = mask_patch[::flip_x, ::flip_y]
        '''
        image_patch = image_patch.astype('float32')
        image_patch = image_patch[np.newaxis, :, :, :]
        image_patch = torch.from_numpy(image_patch).type(torch.FloatTensor)
        
        sub_image_patch = sub_image_patch.astype('float32')
        sub_image_patch = sub_image_patch[np.newaxis, :, :, :]
        sub_image_patch = torch.from_numpy(sub_image_patch).type(torch.FloatTensor)
        
        mask_patch = torch.from_numpy(mask_patch).type(torch.FloatTensor)
        
        return (image_patch, sub_image_patch), mask_patch

    def __len__(self):
        return len(self.coords)


class ValidateDualDataset(object):
    
    def __init__(self, validate_lst, flip, patch_size):
        
        self.validate_lst = validate_lst
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.target_size = target_size
        self.flip = flip
        
        with open(validate_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]   
            
        self.data_list = {}
        for subject in self.subjects:
            self.data_list[subject] = {}
            #img = nib.load(os.path.join(NII_FOLDER,'%s.nii.gz' % subject)).get_data()
            #blood = nib.load(os.path.join(BLOOD_FOLDER,'%s.nii.gz' % subject)).get_data()
            object_path = os.path.join(NII_FOLDER, subject)
            img = nib.load(os.path.join(object_path,'image_resample.nii.gz')).get_data()
            blood = nib.load(os.path.join(object_path,'mask_artery_resample.nii.gz')).get_data()
            while len(img.shape) !=3:
                img = img[:,:,:,0]
            while len(blood.shape) != 3:
                blood = blood[:,:,:,0]
            img = np.transpose(img, (2, 0, 1))
            img = (1.0 * (img - np.min(img)) / (np.max(img) - np.min(img))) * 2 - 1.0

            blood = np.transpose(blood, (2, 0, 1))
            blood[blood > 0] = 1
            
            self.data_list[subject]['image'] = img
            self.data_list[subject]['mask'] = blood
            self.data_list[subject]['points'] = np.asarray(np.where(blood == 1)).T
            
    def sample(self):
        
        PX, PY, PZ = self.patch_size
        self.coords = []
        for subject in self.subjects:
            
            image_shape = self.data_list[subject]['image'].shape
            index = self.data_list[subject]['points']
            x_min, x_max = np.min(index[:,0]), np.max(index[:,0])
            y_min, y_max = np.min(index[:,1]), np.max(index[:,1])
            z_min, z_max = np.min(index[:,2]), np.max(index[:,2])
            
            for x0 in range(0, image_shape[0], PX):
                for y0 in range(0, image_shape[1], PY):
                    for z0 in range(0, image_shape[2], PZ):
                        x1 = x0 + PX
                        y1 = y0 + PY
                        z1 = z0 + PZ
                        if x0 < 0 or x1 > image_shape[0]:
                            continue
                        if y0 < 0 or y1 > image_shape[1]:
                            continue
                        if z0 < 0 or z1 > image_shape[2]:
                            continue
                        self.coords.append([int(subject), x0, y0, z0])
        
    def __getitem__(self, index):
        
        subject_i = self.coords[index][0]
        coord_i = self.coords[index][1:4]
        if subject_i < 10:
            subject = '0' + str(subject_i)
        else:
            subject = str(subject_i)
        image = self.data_list[subject]['image']
        mask = self.data_list[subject]['mask']
        
        image_patch = image[coord_i[0]:coord_i[0]+self.patch_size[0],
                            coord_i[1]:coord_i[1]+self.patch_size[1],
                            coord_i[2]:coord_i[2]+self.patch_size[2]].copy()
        sub_image_patch = image[coord_i[0]:coord_i[0]+self.sub_patch_size[0],
                               coord_i[1]:coord_i[1]+self.sub_patch_size[1],
                               coord_i[2]:coord_i[2]+self.sub_patch_size[2]].copy()
        mask_patch = mask[coord_i[0]:coord_i[0]+self.target_size[0],
                          coord_i[1]:coord_i[1]+self.target_size[1],
                          coord_i[2]:coord_i[2]+self.target_size[2]].copy()


        image_patch = image_patch.astype('float32')
        image_patch = image_patch[np.newaxis, :, :, :]
        image_patch = torch.from_numpy(image_patch).type(torch.FloatTensor)
        sub_image_patch = sub_image_patch.astype('float32')
        sub_image_patch = sub_image_patch[np.newaxis, :, :, :]
        sub_image_patch = torch.from_numpy(sub_image_patch).type(torch.FloatTensor)
        mask_patch = torch.from_numpy(mask_patch).type(torch.FloatTensor)
        
        return (image_patch, sub_image_patch), mask_patch

    def __len__(self):
        return len(self.coords)


