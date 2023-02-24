import torch
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

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

    # 随机90度旋转
    randrotate = RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, prob=1)
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

    # 弹性变换
    randaffined = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(x, y, z),
        rotate_range=(np.pi / 36, np.pi / 36, 0),
        scale_range=(0.15, 0.15, 0),
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

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-35,35)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_shift(image, label):
    image_out = np.empty_like(image)
    label_out = np.empty_like(label)
    shift_row = np.random.randint(-10, 10)
    shift_clu = np.random.randint(-10, 10)
    print(shift_row,shift_clu)
    for i in range(label.shape[0]):
        # Super slow but whatever...
        ndimage.shift(image[i,:,:], shift=[shift_row,shift_clu], output=image_out[i,:,:], order=0,cval=0)
        ndimage.shift(label[i,:,:], shift=[shift_row,shift_clu], output=label_out[i,:,:], order=0,cval=0)
    # image = ndimage.shift(image, shift=[shift_row,shift_clu], order=0,cval=0)
    # label = ndimage.shift(label, shift=[shift_row,shift_clu], order=0,cval=0)
    return image_out, label_out

if __name__ == "__main__":
    main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/train_data/crop_256_patch_3/images/'
    num_name = 180
    data_image_str = main_path + str(num_name)+ '.npy'
    data_masks_str = main_path.replace('images', 'masks') + str(num_name)+ '.npy'
    data_dicts = {'image': data_image_str, 'label': data_masks_str}
    image = np.load(data_image_str).transpose(2,0,1)
    p,h,w = image.shape
    label = np.load(data_masks_str).transpose(2,0,1)
    print(image.shape,label.shape)
    print(label.shape,np.unique(label))
    # label = np.expand_dims(label,axis=0)
    a_image,a_label = random_shift(image,label)
    print(a_label.shape,np.unique(a_label))
    # a_label = a_label.squeeze(-1)
    # Augm_flip,Augm_rotate,Augm_RandGaussianNoise,Augm_RandGaussianSharpen

    show = True
    if show:
        show_num = int(pow(p,0.5))
        plt.figure()
        for i in range(0,p):
            num = i
            x = image[num,:,:]
            plt.subplot(1,p,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,p):
            num = i
            x = label[num,:,:]
            plt.subplot(1,p,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,p):
            num = i
            x = a_image[num,:,:]
            # x = a_image
            plt.subplot(1,p,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.figure()
        for i in range(0,p):
            num = i
            x = a_label[num,:,:]
            plt.subplot(1,p,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()


