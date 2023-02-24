from configparser import Interpolation
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
import cv2


main_path = 'D:/Medical Imaging/BrainTumour/Task01_BrainTumour/imagesTr/'
file_patient_name = sorted(os.listdir(main_path))
total = []
for x in file_patient_name:
    total.append(int(x[6:9]))
total_100 = sorted(random.sample(range(1, 484), 80))
test = sorted(random.sample(total_100, 12))
train = [i for i in total_100 if i not in test]
print('------',len(train),len(test))
#判断两个list是否有相同元素
set_c = set(train) & set(test)
list_c = list(set_c)
print(list_c)
name = []
for file in total_100:
            #print(name_64)
        with open('BT_80.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
name = []
for file in train:
            #print(name_64)
        with open('BT_train.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
name = []
for file in test:
            #print(name_64)
        with open('BT_test.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
