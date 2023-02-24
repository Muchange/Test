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


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

sum = 0
main_path = '/media/xd/date/muzhaoshan/LIDC-IDRI data/'
train_list = open(os.path.join(main_path, 'train_80.txt')).readlines()
for i in range(len(train_list)):
    name = train_list[i].strip('\n')
    print(name)
    data = np.load(main_path+'/LungNoduleLIDCTop100/01grdtUR/' + name+'.npy')
    for j in range(data.shape[-1]):
        if data[:,:,j].max() == 1:
            sum = sum + 1
print(sum)


