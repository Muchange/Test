from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAffined,
)
import SimpleITK as sitk

file_name = 'valid.txt'
file_list = open(os.path.join('D:/Medical Imaging/ADAC/ACDC/lists_ACDC/', file_name)).readlines()
name_list = []
for i in range(len(file_list)):
    patient_name = file_list[i].strip('\n')[5:8]
    name_list.append(patient_name)
    # print(patient_name)
new_name_list = list(set(name_list))
new_name_list.sort()
print(len(new_name_list))
print(new_name_list)
name = []
for file in new_name_list:
            #print(name_64)
        with open(file_name, 'a+') as f:  # 设置文件对象
            f.write(file)  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))

# file_name = 'acad_100.txt'
# file_list = 'D:/Medical Imaging/ADAC/ACDC_data/'
# file_name_list = os.listdir(file_list)
# name = []
# for file in file_name_list:
#             #print(name_64)
#         with open(file_name, 'a+') as f:  # 设置文件对象
#             f.write(file[-3:])  # 将字符串写入文件中
#             f.write('\n')
#         name.append(file)
# print(len(name))


