import torch
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from scipy import ndimage
from monai.transforms import RandCropByPosNegLabeld,CenterSpatialCropd
from utils.overlap_no_pandding import *

def random_rot_flip(image, label):
    # print('use rot')
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    # print('use rotate')
    angle = np.random.randint(-25,25)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_shift(image, label):
    # print('use shift')
    image_out = np.empty_like(image)
    label_out = np.empty_like(label)
    shift_row = np.random.randint(-10, 10)
    shift_clu = np.random.randint(-10, 10)
    # print(shift_row,shift_clu)
    for i in range(image.shape[-1]):
        # Super slow but whatever...
        ndimage.shift(image[:,:,i], shift=[shift_row,shift_clu], output=image_out[:,:,i], order=0,cval=0)
    ndimage.shift(label, shift=[shift_row,shift_clu], output=label_out, order=0,cval=0)
    return image_out, label_out

def Randcrop(data,image_size):
    RrandCrop = RandCropByPosNegLabeld(
        keys=['image','label'],
        label_key = 'label',
        spatial_size = (image_size,image_size,-1),
        pos = 1,
        neg = 1,
        num_samples = 1,
        image_key = 'image',
        image_threshold = 0
    )
    return RrandCrop(data)

def Centercrop(data,image_size):
    CenterCrop = CenterSpatialCropd(
        keys=['image','label'],
       roi_size = (image_size,image_size,-1)
    )
    return CenterCrop(data)

def get_overlap(data,image_size):
    patch_size = [image_size,image_size]
    stride_size = [int(image_size/2),int(image_size/2)]
    image = extract_ordered_pathches(data, patch_size,stride_size)
    return image

class RandomGenerator(object):
    def __init__(self):
        self.init =0
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            image, label = random_shift(image, label)
        else:
            image, label = image, label
        # print(image.shape,label.shape)
        return image,label

class load_data(Dataset):
    def __init__(self, data_path,image_size,Type,transform=None):
        self.Type = Type  # using transform in torch!
        self.transform = transform
        self.image_size = image_size
        self.data_path = data_path
        self.image_path = self.data_path + 'images/'
        self.image_data_size = len(os.listdir(self.image_path))
        self.label_path = self.data_path + 'masks/'
        self.label_data_size = len(os.listdir(self.label_path))
        # self.sample_list = open(os.path.join(test_list_path, 'Sy_test_.txt')).readlines()
        assert self.image_data_size == self.label_data_size

    

    def __getitem__(self,index):
        if self.Type == 'train':
            # print(index)
            image_path = self.image_path + str(index)+'.npy'
            label_path = self.label_path+ str(index)+'.npy'
            # print(image_path)
            image = np.load(image_path)
            label = np.load(label_path)
            label = np.expand_dims(label,axis=-1)
            print(image.shape,label.shape)
            #overlap input shape:H,W,C
            sample_crop = {'image':image,'label':label}
            image,label = self.transform(sample_crop)
            image = image.transpose(2,0,1)
            # print('train',image.shape,label.shape)
            return image,label

        elif self.Type =='test':
            # slice_name = slice_name.zfill(4)
            image_path = self.image_path + str(index)+'.npy'
            label_path = self.label_path + str(index)+'.npy'
            # print(image_path)

            image = np.load(image_path).transpose(2,0,1)
            label = np.load(label_path)
            # print('test',image.shape,label.shape)
            return image,label
        
        else:
            assert 1 == 1 ,'error to train or test'
        
    def __len__(self):
            size = self.image_data_size
            #print(self.image_data_size-1)
            return size
