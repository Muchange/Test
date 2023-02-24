import torch
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from scipy import ndimage

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
    angle = np.random.randint(-35,35)
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
    for i in range(label.shape[0]):
        # Super slow but whatever...
        ndimage.shift(image[i,:,:], shift=[shift_row,shift_clu], output=image_out[i,:,:], order=0,cval=0)
    ndimage.shift(label[0,:,:], shift=[shift_row,shift_clu], output=label_out[0,:,:], order=0,cval=0)
    return image_out, label_out

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
        return image,label

class load_data(Dataset):
    def __init__(self, data_path,Type,transform=None):
        self.Type = Type  # using transform in torch!
        self.data_path = data_path
        self.image_path = self.data_path + '/images/'
        self.image_data_size = len(os.listdir(self.image_path))
        self.label_path = self.data_path +  '/masks/'
        self.label_data_size = len(os.listdir(self.label_path))

        if Type == 'train':
            self.transform = transform
        elif Type =='test':
            self.transform = False
        else:
            assert 1 == 1 ,'error to train or test'

        assert self.image_data_size == self.label_data_size

    def __len__(self):
        size = self.image_data_size
        #print(self.image_data_size-1)
        return size

    def __getitem__(self,index):
        image_path = self.image_path + str(index)+'.npy'
        label_path = self.label_path+ str(index)+'.npy'
        image = np.load(image_path)
        label = np.load(label_path)
        x,y,p = label.shape
        if p>2:
            data_image = image.transpose(2,0,1)
            data_label = label[:,:,int(p/2)]
            data_label = np.expand_dims(data_label,axis=0)
        else:
            data_image = image.transpose(2,0,1)
            data_label = label.transpose(2,0,1)
        sample = {'image': data_image, 'label': data_label}
        if self.transform:
            image,label = self.transform(sample)
            return image,label
        else:
            image, label = sample['image'], sample['label']
            return image,label
        

