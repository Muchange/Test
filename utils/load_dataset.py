import torch
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from matplotlib import pyplot as plt


class load_data(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_path = self.data_path + '/images/'
        self.image_data_size = len(os.listdir(self.image_path))
        self.label_path = self.data_path +  '/masks/'
        self.label_data_size = len(os.listdir(self.label_path))

        assert self.image_data_size == self.label_data_size
        #self.image_data = []
        #self.label_data = []


    def __getitem__(self,index):
        #print(index)
        image = np.load(self.image_path + str(index)+'.npy')
        if len(image.shape)>2:
            image = image.transpose(2,0,1)
        else:
            image = np.expand_dims(image,axis=0)
        label = np.load(self.label_path+ str(index)+'.npy')
        label = np.expand_dims(label,axis=0)
    
        '''
        patch_size = 5
        plt.figure()
        for i in range(patch_size):
            num = i
            x = image[num,:,:]
            plt.subplot(1,patch_size,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.figure()
        for i in range(1):
            num = i
            x = label[num,:,:]
            plt.subplot(1,1,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
        '''
        return image,label

    def __len__(self):
        size = self.image_data_size
        #print(self.image_data_size-1)
        return size
'''
if __name__ == "__main__":
    data_path = '/home/micc/train_data_mirror_256/'
    name_train = 'train'
    name_val = 'val'
    train_set = load_data(data_path, name_train)
    #train_set = np.array(train_set)
    val_set = load_data(data_path, name_val)
    #val_set = np.array(val_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=16,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=16,
                                             shuffle=True)
    for image,label in train_loader:
        print('train image size:',image.shape)
        break;
    for image,label in val_loader:
        print('val image size:', image.shape)
        break;
    print("train number：", len(train_loader))
    print("val number：", len(val_loader))
'''