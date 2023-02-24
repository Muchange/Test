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

file_name = 'train.txt'
main_path = 'D:/Medical Imaging/ADAC/ACAD_data/'
file_list = open(os.path.join(main_path, file_name)).readlines()
for i in range(len(file_list)):
    patient_num = file_list[i].strip('\n')
    patient_path = main_path + 'patient'+patient_num+'/'
    file_patient_name = sorted(os.listdir(patient_path))
    print(file_patient_name)




# name = []
# for file in new_name_list:
#             #print(name_64)
#         with open(file_name, 'a+') as f:  # 设置文件对象
#             f.write(file)  # 将字符串写入文件中
#             f.write('\n')
#         name.append(file)
# print(len(name))


