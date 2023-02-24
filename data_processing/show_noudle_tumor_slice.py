from tkinter import image_names
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk
import torch


sum = 0
main_path = '/media/xd/date/muzhaoshan/LIDC-IDRI data/train_data/crop_256_patch_3/images/'
file_list = np.sort(os.listdir(main_path))
for i in range(len(file_list)):
    name = file_list[i]
    data_image = np.load(main_path+ name)
    data_masks = np.load(main_path.replace('images','masks')+ name)
    print(data_image.shape,data_masks.shape)
    show = True
    patch_size = data_image.shape[-1]
    show_num = round(pow(patch_size,0.5))
    if show_num * show_num < patch_size:
        show_num = show_num + 1 
    if show:
        plt.figure()
        for i in range(patch_size):
            num = i
            x = data_image[:,:,num]
            plt.subplot(show_num,show_num,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.figure()
        x = data_masks
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x,cmap='gray')
        plt.show()


