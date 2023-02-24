import torch
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-15,15)
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
    num_name = 9

    data_image_str = main_path + str(num_name)+ '.npy'
    data_masks_str = main_path.replace('images', 'masks') + str(num_name)+ '.npy'
    data_dicts = {'image': data_image_str, 'label': data_masks_str}
    image = np.load(data_image_str)
    h,w,p = image.shape
    label = np.load(data_masks_str)


    a_image,a_label = random_rot_flip(image,label)
    print(image.shape,label.shape)
    print(a_image.shape,a_label.shape)
    print(np.unique(a_image),np.unique(a_label))

    a_image = a_image.transpose(2,0,1)
    a_label = a_label.transpose(2,0,1)
    show = True
    if show:
        show_num = int(pow(p,0.5))
        plt.figure()
        for i in range(0,p):
            num = i
            x = image[:,:,num]
            plt.subplot(1,p,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,p):
            num = i
            x = label[:,:,num]
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


