from fnmatch import translate
import torch
import os
import glob
from torch.utils.data import Dataset
import random
from numpy import random
import numpy as np
from matplotlib import pyplot as plt

from monai.transforms import (
    LoadImaged,
    AddChanneld,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAffined,
    AddChannel,
    RandFlip,
    RandRotated,
    Rand3DElastic,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandAffine,
)

def Augm_flip(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道

    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 随机翻转，xy面
    randfilp = RandFlipd(keys=["image", "label"], prob=1, spatial_axis=(0, 1))
    data_dict = randfilp(data_dict)


    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_rotate90(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 随机90度旋转
    randrotate90 = RandRotate90d(keys=["image", "label"], prob=1, spatial_axes=(0, 1))
    data_dict = randrotate90(data_dict)

    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label

def Augm_rotate(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)
    rote = random.rand()
    # rote = 0.5
    # print(rote)
    # 随机90度旋转
    randrotate = RandRotated(keys=["image", "label"], range_x=rote,range_y=rote, prob=1)
    data_dict = randrotate(data_dict)

    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_Rand3DElasticd(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]
    x, y, z = image.shape

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 弹性变换
    randaelasticd = Rand3DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(300, 500),
        spatial_size=(x, y, z),
        translate_range=(14, 14, 0),
        rotate_range=(np.pi / 36, np.pi / 36, 0),
        scale_range=(0.15, 0.15, 0),
        padding_mode="border",
    )

    data_dict = randaelasticd(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_RandGaussianNoise(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 高斯噪声
    randgaussiannoise = RandGaussianNoised(keys=["image", "label"],prob=1, mean=0.0, std=0.05)

    data_dict = randgaussiannoise(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_RandGaussianSharpen(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 高斯锐化（使用高斯模糊）
    randgaussiansharpen = RandGaussianSharpend(
                        keys=["image", "label"],
                        prob=1,
                        sigma1_x=(0.5, 1.0),
                        sigma1_y=(0.5, 1.0),
                        sigma1_z=(0.5, 1.0),
                        sigma2_x=0.5,
                        sigma2_y=0.5,
                        sigma2_z=0.5,
                        alpha=(10.0, 30.0),
                        approx='erf')

    data_dict = randgaussiansharpen(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label

def Augm_RandAffined(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]
    x, y, z = image.shape

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)
    # print(data_dict)
    # 随机仿射变换
    randaffined = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(x, y, z),
        rotate_range=(np.pi /36, np.pi / 36, np.pi/4),
        scale_range=(0.3, 0.3, 0.15),
        padding_mode="border",
    )


    data_dict = randaffined(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label

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

if __name__ == "__main__":
    main_path = '/media/xd/date/muzhaoshan/MCM_Transformer/data/test_data/crop_256_patch_1/1GMQX2WE/images/'
    num_name = 183
    data_image_str = main_path + str(num_name)+ '.npy'
    data_masks_str = main_path.replace('images', 'masks') + str(num_name)+ '.npy'
    data_dicts = {'image': data_image_str, 'label': data_masks_str}
    image = np.load(data_image_str)
    
    label = np.load(data_masks_str)
    print(image.shape,label.shape)
    # h,w,p = image.shape
    print(label.shape,np.sum(label==1))
    # label = np.expand_dims(label,axis=0)
    a_image,a_label = Augm_RandAffined(data_dicts)
    # print(a_label.shape)
    # print(np.sum(a_image[:,:,0] == a_image[:,:,1]))
    # print(np.sum(a_label[:,:,0] == a_label[:,:,1]))
    # print(np.unique(a_image[:,:,0]))
    # print(np.unique(a_image[:,:,1]))
    
    # print(np.unique(a_image[:,:,2]))
    # print(a_image[:,:,int(p/2)].shape)
    # a_label = a_label.squeeze(-1)
    # Augm_flip,Augm_rotate,Augm_RandGaussianNoise,Augm_RandGaussianSharpen,Augm_RandAffined
    p = 1
    show = True
    if show:
        # show_num = int(p/2) * 2
        show_num = 1
        plt.figure()
        for i in range(0,p):
            num = i
            x = image[:,:,num]
            plt.subplot(show_num,show_num,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,p):
            num = i
            x = label[:,:,num]
            plt.subplot(show_num,show_num,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,p):
            num = i
            x = a_image[:,:,num]
            # x = a_image
            plt.subplot(show_num,show_num,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.figure()
        for i in range(0,p):
            num = i
            x = a_label[:,:,num]
            # print((a_label==1).all())
            plt.subplot(show_num,show_num,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()


